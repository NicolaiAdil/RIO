import numpy as np
import scipy

# =============================================================================
# ekf.py
#
# Extended Kalman Filter implementation in Python, with numerical Jacobians,
# system discretization (ZOH), and predict/correct stages.
# =============================================================================


class ErrorState_ExtendedKalmanFilter:
    """
    Extended Kalman Filter with Euler predict + ZOH discretization + numerical Jacobians.
    """

    def __init__(self, Q, R, T_acc, T_ars):
        """
        Error-state model:
        δx_dot = A(t)δx + E(t)w
        δy = Cδx + ε

        δx = [(δp^n)^T, (δv^n)^T, (δb_acc^b)^T, (δΘ_nb)^T, (δb_ars^b)^T]^T
        w = [(w_acc)^T, (w_b,acc)^T, (w_ars)^T, (w_b,ars)^T]^T

        Parameters
        A : State transition matrix.
        C : The measurement matrix.
        E, ε : Process and measurement noise matrices.
        dt : timestep
        """
        self.num_states = 15  # Number of states in the error state model
        self.num_states = self.num_states
        self.Q = Q
        self.R = R
        self.T_acc = T_acc
        self.T_ars = T_ars

        self.z = np.zeros((self.num_states, 1))  # measurement placeholder

        # State estimate
        # x_hat_ins = [((p_hat_ins)^n)^T, ((v_hat_ins)^n)^T, Θ^T, (Θ_hat_ins)^T, 0^T]^T
        self.x_hat_ins = np.zeros((self.num_states, 1))  # Also known as x_prior

        # INS propagation state
        self.p_hat_ins = np.zeros((3,1))
        self.v_hat_ins = np.zeros((3,1))
        self.b_acc_ins = np.zeros((3,1))  # Body frame accelerometer bias
        self.theta_hat_ins = np.zeros((3,1))  # Orientation in body frame
        self.b_ars_ins = np.zeros((3,1))  # Body frame angular rate bias

        # Error states
        self.delta_x_hat = np.zeros((self.num_states, 1))  # Also known as x_post
        self.P_hat = np.eye(self.num_states)  # Also known as P_post
        # placeholders
        self.delta_x_hat_prior = np.zeros((self.num_states, 1))
        self.P_hat_prior = np.eye(self.num_states)

    def predict(self, Ad, Ed):
        """
        Predictor update step, based on Fossen 2nd eq. 14.206, 14.207
        """

        # Predict the error state prior
        # δx_hat_prior[k+1] = Ad[k] * δx_hat[k]
        self.delta_x_hat_prior = Ad @ self.delta_x_hat

        # Predict the covariance
        # P_hat_prior[k+1] = Ad[k] * P_hat[k] * Ad[k].T + Ed[k] * Q * Ed[k].T
        self.P_hat_prior = Ad @ self.P_hat @ Ad.T + Ed @ self.Q @ Ed.T

        return self.delta_x_hat_prior, self.P_hat_prior

    def correct(self, z, Cd, R):
        """
        Update step: measurement z.
        """

        # KF gain: K[k]
        K = self.calculate_kalman_gain(Cd, R)
        IKC = np.eye(self.num_states) - K @ Cd

        
        innovation = z - Cd @ self.delta_x_hat_prior
        # print("SHAPE OF INNOVATIONS:", innovation.shape)
        self.delta_x_hat = self.delta_x_hat_prior + K @ innovation
        self.P_hat = IKC @ self.P_hat_prior @ IKC.T + K @ R @ K.T

        if self.delta_x_hat.shape != (self.num_states, 1):
            print(f"SHAPE OF DELTA_X_HAT_PRIOR: {self.delta_x_hat_prior.shape}")
            print(f"SHAPE OF K: {K.shape}")
            print(f"SHAPE OF INNOVATION: {innovation.shape}")
            print(f"SHAPE OF z: {z.shape}")
            print("SHAPE OF Cd:", Cd.shape)
            raise ValueError(
                f"Shape mismatch: delta_x_hat {self.delta_x_hat.shape} does not match expected {self.num_states, 1}"
            )

        return self.delta_x_hat, self.P_hat

    def update_state_estimate(self, delta_x_hat):
        """
        Update the state estimate based on the error state.
        """
        # print("Updating state estimate with delta_x_hat:", delta_x_hat)
        # print("Shape of delta_x_hat:", delta_x_hat.shape, "\n\n")
        # print("Current state estimate (x_hat_ins):", self.x_hat_ins)
        # print("Shape of x_hat_ins:", self.x_hat_ins.shape)
        if delta_x_hat.shape != self.x_hat_ins.shape:
            raise ValueError(
                f"Shape mismatch: delta_x_hat {delta_x_hat.shape} does not match x_hat_ins {self.x_hat_ins.shape}"
            )

        self.x_hat_ins += delta_x_hat
        self.delta_x_hat = np.zeros((self.num_states, 1))  # Reset error state after update

        return self.x_hat_ins

    def ins_propagation(self, x_hat, dt, R_bn, T_bn, f_imu_b, w_imu_b, g_n):
        """
        Propagate the INS state estimate using the error state.
        """
        # print("ENTERING INS propagation")
        # print("Type of x_hat:", type(x_hat))
        # print("x_hat shape:", x_hat.shape)
        # print(x_hat)
        # x_hat = x_hat.reshape(self.num_states, 1) 
        # f_imu_b = np.asarray(f_imu_b).reshape(3, 1)
        # w_imu_b = np.asarray(w_imu_b).reshape(3, 1)
        # g_n     = np.asarray(g_n).reshape(3, 1)
        #print("Shapes of inputs INS PROP:")
        #print("x_hat:", x_hat.shape)
        #print("f_imu_b:", f_imu_b.shape)
        #print("w_imu_b:", w_imu_b.shape)
        #print("g_n:", g_n.shape)

        b_acc_ins = x_hat[6:9]  # Body frame accelerometer bias
        b_ars_ins = x_hat[12:15]  # Body frame angular rate
        # p and v is in n-frame
        # p_hat_ins[k+1] = p_hat_ins[k] + dt * v_hat_ins[k]
        self.p_hat_ins = x_hat[:3] + dt * x_hat[3:6]  # position update

        # v_hat_ins[k+1] = v_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - 0) + g^n)
        self.v_hat_ins = x_hat[3:6] + dt * (
            R_bn @ (f_imu_b - b_acc_ins) + g_n
        )  # velocity update

        # theta_hat_ins[k+1] = theta_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - 0) + g^n)
        self.theta_hat_ins = x_hat[9:12] + dt * (
            T_bn @ (w_imu_b - b_ars_ins)
        )  # orientation update

        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                b_acc_ins,
                self.theta_hat_ins,
                b_ars_ins,
            ]
        )

        return self.x_hat_ins

    def calculate_kalman_gain(self, Cd, R):
        """
        Compute the Kalman gain.
        """
        # print("CALCULATING KALMAN GAIN")
        # print("P_hat_prior shape:", self.P_hat_prior.shape)
        # print("Cd.T shape:", Cd.T.shape)
        # print("Cd @ P_hat_prior shape:", (Cd @ self.P_hat_prior).shape)
        # print("R shape:", R.shape)
        return (
            self.P_hat_prior
            @ Cd.T
            @ np.linalg.inv(Cd @ self.P_hat_prior @ Cd.T + R)
        )

    def generate_A(self, R_bn, T_bn):
        """
        Generate the state transition matrix A.
        Defined in Fossen 2nd, eq. 14.192.
        """
        O3 = np.zeros((3, 3))  # 3x3 zero matrix
        I3 = np.eye(3)  # 3x3 identity matrix

        A = np.block(
            [
                [O3, I3, O3, O3, O3],
                [O3, O3, -R_bn, O3, O3],
                [O3, O3, -1 / self.T_acc * I3, O3, O3],
                [O3, O3, O3, O3, -T_bn],
                [O3, O3, O3, O3, -1 / self.T_ars * I3],
            ]
        )

        return A

    def generate_E(self, R_bn, T_bn):
        """
        Generate the process noise matrix E.
        Defined in Fossen 2nd, eq. 14.193.
        """
        O3 = np.zeros((3, 3))  # 3x3 zero matrix
        I3 = np.eye(3)  # 3x3 identity matrix

        E = np.block(
            [
                [O3, O3, O3, O3],
                [-R_bn, O3, O3, O3],
                [O3, I3, O3, O3],
                [O3, O3, -T_bn, O3],
                [O3, O3, O3, I3],
            ]
        )

        return E

    def generate_C(self):
        """
        Generate the measurement matrix C.
        Defined in Fossen 2nd, eq. 14.194.
        """
        O3 = np.zeros((3, 3))  # 3x3 zero matrix
        I3 = np.eye(3)  # 3x3 identity matrix

        C = np.block(
            [
                [I3, O3, O3, O3, O3],  # Position measurement
                [O3, I3, O3, O3, O3],  # Velocity measurement
                [O3, O3, O3, I3, O3],  # Orientation measurement
            ]
        )

        return C


def Rzyx(phi, theta, psi):
    """
    Compute the 3x3 rotation matrix for Z-Y-X Euler angles.

    Parameters
    ----------
    phi : float or array_like
        Rotation about the x-axis (roll).
    theta : float or array_like
        Rotation about the y-axis (pitch).
    psi : float or array_like
        Rotation about the z-axis (yaw).

    Returns
    -------
    R : ndarray, shape (3,3)
        Rotation matrix in SO(3) corresponding to the sequence
        R = R_z(psi) @ R_y(theta) @ R_x(phi).
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    R = np.array(
        [
            [
                cpsi * cth,
                -spsi * cphi + cpsi * sth * sphi,
                spsi * sphi + cpsi * cphi * sth,
            ],
            [
                spsi * cth,
                cpsi * cphi + sphi * sth * spsi,
                -cpsi * sphi + sth * spsi * cphi,
            ],
            [-sth, cth * sphi, cth * cphi],
        ]
    )
    return R


def Tzyx(phi, theta):
    """
    Compute the 3x3 kinematic transformation matrix T for Z-Y-X Euler angles.

    This maps the Euler-angle rate vector [phi_dot; theta_dot; psi_dot]
    into the body-fixed angular velocity vector [p; q; r] via:
        omega = Tzyx(phi, theta) @ [phi_dot; theta_dot; psi_dot]

    Parameters
    ----------
    phi : float or array_like
        Roll angle (rotation about the x-axis), in radians.
    theta : float or array_like
        Pitch angle (rotation about the y-axis), in radians.

    Returns
    -------
    T : ndarray, shape (3,3)
        The kinematic transformation matrix.

    Raises
    ------
    ValueError
        If cos(theta) is (close to) zero, since the matrix is singular at theta = ±90°.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)

    if np.isclose(cth, 0.0):
        raise ValueError("Tzyx is singular for theta = ±90° (cos(theta) = 0)")

    T = np.array(
        [
            [1.0, sphi * sth / cth, cphi * sth / cth],
            [0.0, cphi, -sphi],
            [0.0, sphi / cth, cphi / cth],
        ]
    )
    return T


def ssa(angle, unit="rad"):
    """
    Smallest-Signed Angle: wrap angle to the interval
      [-π, π) if unit=='rad' (default), or
      [-180°, 180°) if unit=='deg'.

    Parameters
    ----------
    angle : array_like or scalar
        Input angle(s), in radians or degrees.
    unit : {'rad', 'deg'}, optional
        Unit of the input and output angle. Defaults to 'rad'.

    Returns
    -------
    wrapped : ndarray or scalar
        Angle(s) wrapped to the smallest signed representation.
    """
    angle = np.asarray(angle)

    if unit == "rad":
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    elif unit == "deg":
        wrapped = (angle + 180.0) % 360.0 - 180.0
    else:
        raise ValueError(f"Invalid unit '{unit}'. Must be 'rad' or 'deg'.")

    # If input was a scalar, return a scalar
    if np.isscalar(angle):
        return wrapped.item()
    return wrapped

def gravity(lat):
    """
    g = gravity(latitude) computes the acceleration of gravity (m/s^2) as a function
    of latitude lat (rad) using the WGS-84 ellipsoid parameters

    Parameters
    ----------
    latitude : float
        Latitude in degrees.

    Returns
    -------
    g : float
        Gravitational acceleration in m/s² (NED)
    """
    # Convert latitude to radians
    lat_rad = np.radians(lat)

    # Gravitational constant at sea level (m/s²)
    g0 = 9.7803253359

    # Gravity formula based on latitude
    g = g0 * (1 + 0.001931850400 * np.sin(lat_rad)**2) / np.sqrt(1 - 0.006694384442 * np.sin(lat_rad)**2)

    return g
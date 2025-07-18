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

    def __init__(self, dt, Q, R, T_acc, T_ars):
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
        self.dt = dt
        self.num_states = self.num_states
        self.Q = Q
        self.R = R
        self.T_acc = T_acc
        self.T_ars = T_ars

        self.z = np.zeros(self.num_states)  # measurement placeholder

        # State estimate
        # x_hat_ins = [((p_hat_ins)^n)^T, ((v_hat_ins)^n)^T, Θ^T, (Θ_hat_ins)^T, 0^T]^T
        self.x_hat_ins = np.zeros(self.num_states)  # Also known as x_prior

        # INS propagation state
        self.p_hat_ins = np.zeros(3)
        self.v_hat_ins = np.zeros(3)
        self.theta_hat_ins = np.zeros(3)  # Orientation in body frame

        # Error states
        self.delta_x_hat = np.zeros(self.num_states)  # Also known as x_post
        self.P_hat = np.eye(self.num_states)  # Also known as P_post
        # placeholders
        self.delta_x_hat_prior = np.zeros(self.num_states)
        self.P_hat_prior = np.eye(self.num_states)

    def predict(self):
        """
        Predictor update step, based on Fossen 2nd eq. 14.206, 14.207
        """

        # Discretization
        Ad, Ed = np.eye(self.num_states) + self.A * self.dt, self.E * self.dt

        # Predict the error state prior
        # δx_hat_prior[k+1] = Ad[k] * δx_hat[k]
        self.delta_x_hat_prior = Ad @ self.delta_x_hat

        # Predict the covariance
        # P_hat_prior[k+1] = Ad[k] * P_hat[k] * Ad[k].T + Ed[k] * Q * Ed[k].T
        self.P_hat_prior = Ad @ self.P_hat @ Ad.T + Ed @ self.Q @ Ed.T

        return self.delta_x_hat_prior, self.P_hat_prior

    def correct(self, delta_x_hat_prior, z, C):
        """
        Update step: measurement z.
        """
        K = self.calculate_kalman_gain()
        self.delta_x_hat = delta_x_hat_prior + K @ (z - C @ delta_x_hat_prior)
        self.P_hat = (np.eye(self.num_states) - K @ C) @ self.P_hat_prior @ (
            np.eye(self.num_states) - K @ C
        ).T + K @ self.R @ K.T

        return self.delta_x_hat, self.P_hat

    def update_state_estimate(self, delta_x_hat):
        """
        Update the state estimate based on the error state.
        """
        self.x_hat_ins += delta_x_hat
        self.delta_x_hat = 0  # Reset error state after update

        return self.x_hat_ins

    def ins_propagation(self, x_hat, R_bn, T_bn, f_imu_b, w_imu_b, g_n):
        """
        Propagate the INS state estimate using the error state.
        """
        b_acc_ins = x_hat[6:9]  # Body frame accelerometer bias
        b_ars_ins = x_hat[12:15]  # Body frame angular rate
        # p and v is in n-frame
        # p_hat_ins[k+1] = p_hat_ins[k] + dt * v_hat_ins[k]
        self.p_hat_ins = x_hat[:3] + self.dt * x_hat[3:6]  # position update

        # v_hat_ins[k+1] = v_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - 0) + g^n)
        self.v_hat_ins = x_hat[3:6] + self.dt * (
            R_bn @ (f_imu_b - b_acc_ins) + g_n
        )  # velocity update

        # theta_hat_ins[k+1] = theta_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - 0) + g^n)
        self.theta_hat_ins = x_hat[9:12] + self.dt * (
            T_bn @ (w_imu_b - b_ars_ins)
        )  # orientation update

        return self.p_hat_ins, self.v_hat_ins, self.theta_hat_ins

    def calculate_kalman_gain(self):
        """
        Compute the Kalman gain.
        """
        return (
            self.P_hat_prior
            @ self.C.T
            @ np.linalg.inv(self.C @ self.P_hat_prior @ self.C.T + self.R)
        )
    
    def generate_A(self, R_bn, T_bn):
        """
        Generate the state transition matrix A.
        Defined in Fossen 2nd, eq. 14.192.
        """
        O3 = np.zeros((3, 3))  # 3x3 zero matrix
        I3 = np.eye(3)  # 3x3 identity matrix

        A = np.block([
            [O3, I3, O3                , O3, O3              ], 
            [O3, O3, -R_bn             , O3, O3              ],
            [O3, O3, -1/self.T_acc * I3, O3, O3              ],
            [O3, O3, O3                , O3, -T_bn           ],
            [O3, O3, O3                , O3, -1/self.T_ars*I3]
        ])

    def generate_E(self):
        """
        Generate the process noise matrix E.
        """
        O3 = np.zeros((3, 3))
    
    def generate_C(self):


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

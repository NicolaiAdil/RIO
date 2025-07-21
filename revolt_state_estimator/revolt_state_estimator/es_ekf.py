import numpy as np

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

    def __init__(self, Q, T_acc, T_ars):
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
        self.T_acc = T_acc
        self.T_ars = T_ars

        self.z = np.zeros((self.num_states, 1))  # measurement placeholder

        # State estimate
        # x_hat_ins = [((p_hat_ins)^n)^T, ((v_hat_ins)^n)^T, ((b_hat_acc,ins)^b)^T, (Θ_hat_ins)^T, ((b_hat_ars,ins)^b)^T]^T (Eq. 14.213 in Fossen 2nd)
        self.x_hat_ins = np.zeros((self.num_states, 1))  # Also known as x_prior

        # INS propagation state
        self.p_hat_ins = np.zeros((3,1)) # Position in NED frame
        self.v_hat_ins = np.zeros((3,1)) # Velocity in NED frame
        self.b_acc_ins = np.zeros((3,1))  # Body frame accelerometer bias
        self.theta_hat_ins = np.zeros((3,1))  # Attitude in body frame
        self.b_ars_ins = np.zeros((3,1))  # Body frame angular rate bias

        # Error states
        # Posteri error states
        self.delta_x_hat = np.zeros((self.num_states, 1))  # Also known as x_post
        self.P_hat = np.eye(self.num_states)  # Also known as P_post

        # Prior error states
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
  
        self.delta_x_hat = self.delta_x_hat_prior + K @ innovation
        self.P_hat = IKC @ self.P_hat_prior @ IKC.T + K @ R @ K.T

        if self.delta_x_hat.shape != (self.num_states, 1):
            raise ValueError(
                f"Shape mismatch: delta_x_hat {self.delta_x_hat.shape} does not match expected {self.num_states, 1}"
            )

        return self.delta_x_hat, self.P_hat

    def update_state_estimate(self, delta_x_hat):
        """
        Update the state estimate based on the error state.
        """
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
        b_acc_ins = x_hat[6:9]  # Body frame accelerometer bias
        b_ars_ins = x_hat[12:15]  # Body frame angular rate
        # p and v is in n-frame
        # p_hat_ins[k+1] = p_hat_ins[k] + dt * v_hat_ins[k]
        self.p_hat_ins = x_hat[:3] + dt * x_hat[3:6]  # position update

        # v_hat_ins[k+1] = v_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - b_acc,ins^b[k]) + g^n)
        self.v_hat_ins = x_hat[3:6] + dt * (
            R_bn @ (f_imu_b - b_acc_ins) + g_n
        )  # velocity update

        # theta_hat_ins[k+1] = theta_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - b_ars,ins^b) + g^n)
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



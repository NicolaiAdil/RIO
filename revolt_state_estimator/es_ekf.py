import numpy as np
from revolt_state_estimator.revolt_sensor_transforms import Tzyx, Rzyx
from revolt_state_estimator.utils import ssa, _skew, _exp_so3

# =============================================================================
# es_ekf.py
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
        self.Q = Q
        self.T_acc = T_acc
        self.T_ars = T_ars

        self.z = np.zeros((self.num_states, 1))  # measurement placeholder

        # State estimate
        # x_hat_ins = [((p_hat_ins)^n)^T, ((v_hat_ins)^n)^T, ((b_hat_acc,ins)^b)^T, (Θ_hat_ins)^T, ((b_hat_ars,ins)^b)^T]^T (Eq. 14.213 in Fossen 2nd)
        self.x_hat_ins = np.zeros((self.num_states, 1))  # Also known as x_prior

        # INS propagation state
        self.p_hat_ins = np.zeros((3, 1))  # Position in NED frame
        self.v_hat_ins = np.zeros((3, 1))  # Velocity in NED frame
        self.b_acc_ins = np.zeros((3, 1))  # Body frame accelerometer bias
        self.theta_hat_ins = np.zeros((3, 1))  # Attitude from body to NED frame
        self.b_ars_ins = np.zeros((3, 1))  # Body frame angular rate bias

        # Error states
        # Posteri error states
        self.delta_x_hat = np.zeros((self.num_states, 1))  # Also known as x_post
        self.P_hat = np.eye(self.num_states)  # Also known as P_post

        # Prior error states
        self.delta_x_hat_prior = np.zeros((self.num_states, 1))
        self.P_hat_prior = np.eye(self.num_states)

    def predict(self, Ad, Qd):
        """
        Predictor update step, based on Fossen 2nd eq. 14.206, 14.207
        """

        # Predict the error state prior
        # δx_hat_prior[k+1] = Ad[k] * δx_hat[k]
        self.delta_x_hat_prior = Ad @ self.delta_x_hat

        # Predict the covariance
        # P_hat_prior[k+1] = Ad[k] * P_hat[k] * Ad[k].T + Ed[k] * Q * Ed[k].T
        self.P_hat_prior = Ad @ self.P_hat @ Ad.T + Qd

        return self.delta_x_hat_prior, self.P_hat_prior

    def correct(self, e, H, R_meas):
        """
        Update step: measurement z.
        """

        # KF gain: K[k]
        K = self.calculate_kalman_gain(H, R_meas)
        IKC = np.eye(self.num_states) - K @ H
        # innovation = z - H @ self.delta_x_hat_prior

        self.delta_x_hat = self.delta_x_hat_prior + K @ e
        self.P_hat = IKC @ self.P_hat_prior @ IKC.T + K @ R_meas @ K.T # Joseph form

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

        # nominal additive parts
        self.x_hat_ins[:6]  += delta_x_hat[:6]
        self.x_hat_ins[6:9] += delta_x_hat[6:9]     # b_a
        self.x_hat_ins[12:15] += delta_x_hat[12:15] # b_g

        # multiplicative attitude on SO(3)
        roll, pitch, yaw = self.theta_hat_ins.flatten()
        R_nb = Rzyx(roll, pitch, yaw)
        dth = delta_x_hat[9:12, 0]
        R_nb_next = R_nb @ _exp_so3(dth)            # compose rotation
        # re-extract wrapped Euler only for storage/output
        roll  = np.arctan2(R_nb_next[2,1], R_nb_next[2,2])
        pitch = -np.arcsin(np.clip(R_nb_next[2,0], -1.0, 1.0))
        yaw   = np.arctan2(R_nb_next[1,0], R_nb_next[0,0])
        self.theta_hat_ins[:] = np.array([ssa(roll), ssa(pitch), ssa(yaw)]).reshape(3,1)

        # rebuild x_hat_ins angles
        self.x_hat_ins[9:12] = self.theta_hat_ins

        G = np.eye(15)
        G[9:12,9:12] = np.eye(3) - _skew(0.5 * dth)  # attitude correction
        self.P_hat = G @ self.P_hat @ G.T  # Update covariance with attitude correction

        self.delta_x_hat = np.zeros(
            (self.num_states, 1)
        )  # Reset error state after update

        return self.x_hat_ins

    def ins_propagation(self, x_hat, dt, R_bn, T_bn, f_imu_b, w_imu_b, g_n):
        """
        Propagate the INS state estimate using the error state.
        """

        self.b_acc_ins = x_hat[6:9]  # Body frame accelerometer bias
        self.b_ars_ins = x_hat[12:15]  # Body frame angular rate
        # p and v is in n-frame
        # p_hat_ins[k+1] = p_hat_ins[k] + dt * v_hat_ins[k]
        self.p_hat_ins = x_hat[:3] + dt * x_hat[3:6] + 0.5 * dt**2 * (R_bn @ (f_imu_b) + g_n)  # position update

        # v_hat_ins[k+1] = v_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - b_acc,ins^b[k]) + g^n)
        self.v_hat_ins = x_hat[3:6] + dt * (
            R_bn @ (f_imu_b) + g_n
        )  # velocity update

        # Euler angles to rotation matrix
        dR = _exp_so3(w_imu_b * dt)
        R_bn_next = R_bn @ dR 
        roll  = np.arctan2(R_bn_next[2,1], R_bn_next[2,2])
        pitch = -np.arcsin(np.clip(R_bn_next[2,0], -1.0, 1.0))
        yaw   = np.arctan2(R_bn_next[1,0], R_bn_next[0,0])
        theta_next = np.array([ssa(roll), ssa(pitch), ssa(yaw)])
        self.theta_hat_ins = theta_next.reshape(3,1)

        # # v_hat_ins[k+1] = v_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - b_acc,ins^b[k]) + g^n)
        # self.v_hat_ins = x_hat[3:6] + dt * (
        #     R_bn_next @ (f_imu_b) + g_n
        # )  # velocity update

        # # p and v is in n-frame
        # # p_hat_ins[k+1] = p_hat_ins[k] + dt * v_hat_ins[k]
        # self.p_hat_ins = x_hat[:3] + dt * self.v_hat_ins  # position update

        # theta_hat_ins[k+1] = theta_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - b_ars,ins^b) + g^n)
        # theta_hat_unwrapped = x_hat[9:12] + dt * (
        #     T_bn @ (w_imu_b)
        # )  # orientation update
        # roll_wrapped = ssa(theta_hat_unwrapped[0])
        # pitch_wrapped = ssa(theta_hat_unwrapped[1])
        # yaw_wrapped = ssa(theta_hat_unwrapped[2])

        # self.theta_hat_ins = np.array([[roll_wrapped], [pitch_wrapped], [yaw_wrapped]]).reshape(3, 1)

        
        # Alternatively, using tf_transformations (commented out)
        # roll, pitch, yaw = tf_transformations.euler_from_matrix(R_bn_next, axes='sxyz')

        # self.theta_hat_ins = np.array([[ssa(theta_hat_unwrapped[0,0])],
        #                                [ssa(theta_hat_unwrapped[1,0])],
        #                                [ssa(theta_hat_unwrapped[2,0])]])

        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                self.b_acc_ins,
                self.theta_hat_ins,
                self.b_ars_ins,
            ]
        )

        return self.x_hat_ins

    def calculate_kalman_gain(self, H, R):
        """
        Compute the Kalman gain.
        """
        return self.P_hat_prior @ H.T @ np.linalg.inv(H @ self.P_hat_prior @ H.T + R)
    
    # Solas implementation

    def generate_A(self, R_nb, T_nb, f_b_nom, w_b_nom):
        O3 = np.zeros((3, 3)); I3 = np.eye(3)
        A = np.block([
            [O3,  I3,                 O3,                     O3,                 O3],
            [O3,  O3,              -R_nb, -R_nb @ _skew(f_b_nom),                 O3],
            [O3,  O3, -(1/self.T_acc)*I3,                     O3,                 O3],
            [O3,  O3,                 O3,        -_skew(w_b_nom),                -I3],
            [O3,  O3,                 O3,                     O3, -(1/self.T_ars)*I3],
        ])
        return A

    def generate_E(self, R_nb, T_bn):
        O3 = np.zeros((3, 3)); I3 = np.eye(3)
        E = np.block([
            [   O3,  O3,    O3, O3],
            [-R_nb,  O3,    O3, O3],
            [   O3,  I3,    O3, O3], 
            [   O3,  O3,   -I3, O3],
            [   O3,  O3,    O3, I3],
        ])
        return E

    #Fossens implementation

    # def generate_A(self, R_nb, T_nb, f_b_nom):
    #     """
    #     Linearized error dynamics (Fossen 2nd, §14.4.2):
    #     δv̇^n = -R_nb δb_a^b  - R_nb [f_b_nom]_x δΘ
    #     """
    #     O3 = np.zeros((3, 3)); I3 = np.eye(3)
    #     A = np.block([
    #         [O3, I3,                 O3,                  O3,               O3],
    #         [O3, O3,               -R_nb,        -R_nb @ _skew(f_b_nom),    O3],
    #         [O3, O3, -(1/self.T_acc)*I3,          O3,                       O3],
    #         [O3, O3,                 O3,          O3,                     -T_nb],
    #         [O3, O3,                 O3,          O3,     -(1/self.T_ars)*I3],
    #     ])
    #     return A

    # def generate_E(self, R_bn, T_bn):
    #     """
    #     Generate the process noise matrix E.
    #     Defined in Fossen 2nd, eq. 14.193.
    #     """
    #     O3 = np.zeros((3, 3))  # 3x3 zero matrix
    #     I3 = np.eye(3)  # 3x3 identity matrix

    #     E = np.block(
    #         [
    #             [O3, O3, O3, O3],
    #             [-R_bn, O3, O3, O3],
    #             [O3, I3, O3, O3],
    #             [O3, O3, -T_bn, O3],
    #             [O3, O3, O3, I3],
    #         ]
    #     )

    #     return E


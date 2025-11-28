import numpy as np
import tf_transformations
from revolt_state_estimator.utils import _skew, _exp_so3, _project_to_SO3

# =============================================================================
# es_ekf.py
# =============================================================================


class ErrorState_ExtendedKalmanFilter:
    """
    Extended Kalman Filter with quaternion nominal attitude and 15D error state:
      δx = [δp^n, δv^n, δb_acc^b, δθ_nb, δb_ars^b]^T
    """

    def __init__(self, Q, P_initial, T_acc, T_ars):
        """
        Error-state model:
        δx_dot = A(t)δx + E(t)w
        δy     = Cδx   + ε

        δx = [(δp^n)^T, (δv^n)^T, (δb_acc^b)^T, (δΘ_nb)^T, (δb_ars^b)^T]^T
        w  = [(w_acc)^T, (w_b,acc)^T, (w_ars)^T, (w_b,ars)^T]^T
        """
        # Error-state dimension
        self.num_states = 15

        self.Q      = Q
        self.T_acc  = T_acc
        self.T_ars  = T_ars

        # Nominal INS state components (16 states total)
        self.p_hat_ins  = np.zeros((3, 1))           # position in NED
        self.v_hat_ins  = np.zeros((3, 1))           # velocity in NED
        self.b_acc_ins  = np.zeros((3, 1))           # accel bias (body)
        self.q_hat_ins  = np.array([[0.0],
                                    [0.0],
                                    [0.0],
                                    [1.0]])         # quaternion [x,y,z,w], body→NED
        self.b_ars_ins  = np.zeros((3, 1))           # gyro bias (body)

        # Nominal state vector: 16x1
        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                self.b_acc_ins,
                self.q_hat_ins,
                self.b_ars_ins,
            ]
        )
        self.num_ins_states = self.x_hat_ins.shape[0]   # 16

        # Error state and covariance
        self.delta_x_hat       = np.zeros((self.num_states, 1))
        self.delta_x_hat_prior = np.zeros((self.num_states, 1))

        self.P_hat       = P_initial.copy()
        self.P_hat_prior = P_initial.copy()

    # -------------------------------------------------------------------------
    # Kalman filter core
    # -------------------------------------------------------------------------
    def predict(self, Ad, Qd):
        """
        Predictor update (Fossen 2nd, eq. 14.206, 14.207)
        """
        # δx_hat_prior[k+1] = Ad[k] * δx_hat[k]
        self.delta_x_hat_prior = Ad @ self.delta_x_hat

        # P_hat_prior[k+1] = Ad[k] * P_hat[k] * Ad[k].T + Qd[k]
        self.P_hat_prior = Ad @ self.P_hat @ Ad.T + Qd

        return self.delta_x_hat_prior, self.P_hat_prior

    def calculate_kalman_gain(self, H, R):
        """
        Compute the Kalman gain.
        """
        S = H @ self.P_hat_prior @ H.T + R
        K = self.P_hat_prior @ H.T @ np.linalg.inv(S)
        return K

    def correct(self, e, H, R_meas):
        """
        Measurement update using residual e (scalar or vector).

        e = z - h(x)
        H = ∂h/∂(δx)
        """
        K = self.calculate_kalman_gain(H, R_meas)
        IKH = np.eye(self.num_states) - K @ H

        # Error-state update
        self.delta_x_hat = K @ e

        # Covariance update (Joseph form)
        self.P_hat = IKH @ self.P_hat_prior @ IKH.T + K @ R_meas @ K.T

        if self.delta_x_hat.shape != (self.num_states, 1):
            raise ValueError(
                f"Shape mismatch: delta_x_hat {self.delta_x_hat.shape} "
                f"does not match expected {(self.num_states, 1)}"
            )

        return self.delta_x_hat, self.P_hat

    # -------------------------------------------------------------------------
    # INS reset / nominal state update from error state
    # -------------------------------------------------------------------------
    def update_state_estimate(self, delta_x_hat):
        """
        Apply error state δx to nominal state x_hat_ins using
        additive corrections for p, v, biases and multiplicative
        correction on SO(3) for attitude.
        """
        if delta_x_hat.shape != (self.num_states, 1):
            raise ValueError("Invalid shape for delta_x_hat")

        # Additive parts
        self.p_hat_ins  += delta_x_hat[0:3]
        self.v_hat_ins  += delta_x_hat[3:6]
        self.b_acc_ins  += delta_x_hat[6:9]
        self.b_ars_ins  += delta_x_hat[12:15]

        # Attitude error δθ (3x1)
        dth = delta_x_hat[9:12, 0]  # roll, pitch, yaw error (small angles)

        # Current R_nb from quaternion (body→NED)
        q = self.q_hat_ins.flatten()
        R_nb = tf_transformations.quaternion_matrix(q)[:3, :3]

        # Multiplicative correction: R_nb_next = R_nb @ exp([δθ]x)
        dR = _exp_so3(dth)
        R_nb_next = R_nb @ dR
        R_nb_next = _project_to_SO3(R_nb_next)

        # Back to quaternion
        R4 = np.eye(4)
        R4[:3, :3] = R_nb_next
        q_next = tf_transformations.quaternion_from_matrix(R4)
        self.q_hat_ins = np.asarray(q_next, dtype=float).reshape(4, 1)

        # Rebuild nominal state vector (16x1)
        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                self.b_acc_ins,
                self.q_hat_ins,
                self.b_ars_ins,
            ]
        )

        # Covariance correction for attitude block
        G = np.eye(self.num_states)
        G[9:12, 9:12] = np.eye(3) - _skew(0.5 * dth)
        self.P_hat = G @ self.P_hat @ G.T

        # Reset error state after applying the correction
        self.delta_x_hat = np.zeros((self.num_states, 1))

        return self.x_hat_ins

    # -------------------------------------------------------------------------
    # INS propagation
    # -------------------------------------------------------------------------
    def ins_propagation(self, x_hat, dt, f_imu_b, w_imu_b, g_w):
        """
        Propagate nominal INS state with IMU measurements (bias-corrected already).

        x_hat: current nominal state (16x1)
        dt:    timestep
        f_imu_b: specific force in body frame (3x1), bias-corrected
        w_imu_b: angular rate in body frame (3x1), bias-corrected
        g_w: gravity vector in NED (3x1)
        """
        # Read nominal from input vector
        self.p_hat_ins = x_hat[0:3]
        self.v_hat_ins = x_hat[3:6]
        self.b_acc_ins = x_hat[6:9]
        self.q_hat_ins = x_hat[9:13]   # 4x1
        self.b_ars_ins = x_hat[13:16]

        # Rotation body→NED from quaternion
        q = self.q_hat_ins.flatten()
        R_nb = tf_transformations.quaternion_matrix(q)[:3, :3]

        # Linear acceleration in NED
        a_n = R_nb @ f_imu_b + g_w

        # Position & velocity update
        self.p_hat_ins = self.p_hat_ins + dt * self.v_hat_ins + 0.5 * dt**2 * a_n
        self.v_hat_ins = self.v_hat_ins + dt * a_n

        # Attitude update: R_nb_next = R_nb @ exp([w_b]x dt)
        dR = _exp_so3((w_imu_b.reshape(3)) * dt)
        R_nb_next = R_nb @ dR
        R_nb_next = _project_to_SO3(R_nb_next)

        # Back to quaternion
        R4 = np.eye(4)
        R4[:3, :3] = R_nb_next
        q_next = tf_transformations.quaternion_from_matrix(R4)
        self.q_hat_ins = np.asarray(q_next, dtype=float).reshape(4, 1)

        # Rebuild nominal state (16x1)
        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                self.b_acc_ins,
                self.q_hat_ins,
                self.b_ars_ins,
            ]
        )

        return self.x_hat_ins

    # -------------------------------------------------------------------------
    # Linearization matrices
    # -------------------------------------------------------------------------
    def generate_A(self, R_nb, f_b_nom, w_b_nom):
        """
        Continuous-time A matrix for the 15D error state.
        """
        O3 = np.zeros((3, 3))
        I3 = np.eye(3)
        A = np.block([
            [O3,  I3,                 O3,                     O3,                 O3],
            [O3,  O3,              -R_nb, -R_nb @ _skew(f_b_nom),                 O3],
            [O3,  O3, -(1/self.T_acc)*I3,                     O3,                 O3],
            [O3,  O3,                 O3,        -_skew(w_b_nom),                -I3],
            [O3,  O3,                 O3,                     O3, -(1/self.T_ars)*I3],
        ])
        return A

    def generate_E(self, R_nb):
        """
        Continuous-time E matrix for process noise mapping.
        """
        O3 = np.zeros((3, 3))
        I3 = np.eye(3)
        E = np.block([
            [   O3,  O3,    O3, O3],
            [-R_nb,  O3,    O3, O3],
            [   O3,  I3,    O3, O3],
            [   O3,  O3,   -I3, O3],
            [   O3,  O3,    O3, I3],
        ])
        return E

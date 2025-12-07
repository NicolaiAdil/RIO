#!/usr/bin/env python3
"""
ROS2 node that runs an Error-state Extended Kalman Filter using IMU + radar Doppler.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import (
    TransformStamped, 
    Quaternion,
    Vector3Stamped,
    PoseStamped,
)
import tf_transformations
import tf2_ros
import numpy as np

from state_estimator.eskf import ErrorState_KalmanFilter
from state_estimator.utils import _skew


def quat_xyzw_to_R(qx, qy, qz, qw):
    return tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]


class StateEstimator(Node):
    # Initialize EKF system (START) ==================================================
    def __init__(self):
        super().__init__("state_estimator")

        # EKF Parameters
        self.declare_parameter("parameters.Q", [0.0] * 12)  # []
        self.declare_parameter("parameters.radar_sigma_vr", 0.038)
        self.declare_parameter("parameters.T_acc", 1000.0)
        self.declare_parameter("parameters.T_ars", 500.0)

        # Gating parameters
        self.declare_parameter("parameters.radar_gating_enable", True)
        self.declare_parameter("parameters.radar_gate_nsigma", 3.0)

        # IKF parameters (iterated EKF on measurement update)
        self.declare_parameter("parameters.ikf_enable", False)
        self.declare_parameter("parameters.ikf_max_iters", 3)
        self.declare_parameter("parameters.ikf_tol", 1e-4)

        # Extrinsic transformation from radar to IMU
        self.declare_parameter("parameters.l_BR_B", [0.0, 0.0, 0.0])
        self.declare_parameter("parameters.q_R_B", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("radar_vr_sign", +1)

        _P_initial = self.get_initial_P()

        _Q              = self.get_parameter("parameters.Q").value
        _radar_sigma_vr = self.get_parameter("parameters.radar_sigma_vr").value
        _T_acc          = self.get_parameter("parameters.T_acc").value
        _T_ars          = self.get_parameter("parameters.T_ars").value
        _l_BR_B         = self.get_parameter("parameters.l_BR_B").value
        _q_R_B          = self.get_parameter("parameters.q_R_B").value
        _radar_vr_sign  = self.get_parameter("radar_vr_sign").value
        _gating_enable  = self.get_parameter("parameters.radar_gating_enable").value
        _gate_nsigma    = self.get_parameter("parameters.radar_gate_nsigma").value
        _ikf_enable     = self.get_parameter("parameters.ikf_enable").value
        _ikf_max_iters  = self.get_parameter("parameters.ikf_max_iters").value
        _ikf_tol        = self.get_parameter("parameters.ikf_tol").value

        # ROS 2 Parameters
        self.declare_parameter("parameters.state_estimate_topic", "/rio/pose")
        self.declare_parameter("parameters.imu_topic", "/imu/data")
        self.declare_parameter("parameters.radar_topic", "/vel")

        _state_estimate_topic = self.get_parameter("parameters.state_estimate_topic").value
        _imu_topic            = self.get_parameter("parameters.imu_topic").value
        _radar_topic          = self.get_parameter("parameters.radar_topic").value

        l_BR_B = np.array(_l_BR_B, dtype=float).reshape(3, 1)
        qx, qy, qz, qw = _q_R_B
        q_R_B = np.array([qx, qy, qz, qw], dtype=float).reshape(4, 1)

        self.vr_sign  = int(_radar_vr_sign)
        self.sigma_vr = float(_radar_sigma_vr)

        self.gating_enable = bool(_gating_enable)
        self.gate_nsigma   = float(_gate_nsigma)

        self.ikf_enable    = bool(_ikf_enable)
        self.ikf_max_iters = int(_ikf_max_iters)
        self.ikf_tol       = float(_ikf_tol)

        # EKF Setup ----------
        self.new_velocity_measurement = False
        self.initialized              = False
        self.imu_last_stamp           = None

        self.MU_R    = None
        self.VR_meas = None

        self.Q             = np.diag(_Q)
        self.P_hat_initial = _P_initial

        # Error-state Kalman Filter
        self.eskf = ErrorState_KalmanFilter(
            Q=self.Q,
            P_initial=self.P_hat_initial,
            T_acc=_T_acc,
            T_ars=_T_ars,
            p_IR_init=l_BR_B,
            q_IR_init=q_R_B,
        )

        # ROS2 Interfaces ----------
        self.radar_sub = self.create_subscription(
            PointCloud2, _radar_topic, self.update_radar, 1
        )
        self.imu_sub = self.create_subscription(Imu, _imu_topic, self.imu_callback, 1)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state_pub = self.create_publisher(Odometry, _state_estimate_topic, 10)
        # Bias publishers (accel & gyro)
        self.accel_bias_pub = self.create_publisher(Vector3Stamped, "/rio/accel_bias", 10)
        self.gyro_bias_pub  = self.create_publisher(Vector3Stamped, "/rio/gyro_bias", 10)
        # Radar extrinsics publisher
        self.radar_pose_pub = self.create_publisher(PoseStamped, "/radar/extrinsics", 10)

        # Debugging ----------
        np.set_printoptions(
            linewidth=200,
            precision=6,
            suppress=True,
        )

        self.e_mean = 0.0
        self.e_std  = 0.0
        self.i      = 0

        self.get_logger().info(
            f"\nEKF Parameters:\n"
            f" Q:\n{self.Q}\n"
            f" sigma_vr: {self.sigma_vr}\n"
            f" T_acc: {_T_acc}\n"
            f" T_ars: {_T_ars}\n\n"
            f"Gating enabled: {self.gating_enable}\n"
            f" Gate n-sigma: {self.gate_nsigma}\n\n"
            f"IKF enabled: {self.ikf_enable}\n"
            f" IKF max iterations: {self.ikf_max_iters}\n"
            f" IKF tolerance: {self.ikf_tol}\n\n"
            f"Publisher topics:\n"
            f" State estimate topic: {_state_estimate_topic}\n"
            f"Subscribe topics:\n"
            f" IMU topic: {_imu_topic}\n"
            f" Radar topic: {_radar_topic}\n"
            f"Extrinsic transformation (l_BR_B): {_l_BR_B}\n"
            f"Extrinsic transformation (q_R_B): {_q_R_B}\n"
            f"Radar vr sign: {_radar_vr_sign}\n"
        )
        self.get_logger().info("EKF waiting for IMU and radar measurements to start...")

    # Initialize EKF system (STOP) ==================================================

    def _publish_state(self, x, P, time_stamp):
        """
        x: nominal state (16x1)
        P: 15x15 error covariance
        """
        x_pos, y_pos, z_pos = x[0], x[1], x[2]
        v_x, v_y, v_z       = x[3], x[4], x[5]
        b_ax, b_ay, b_az    = x[6], x[7], x[8]
        qx, qy, qz, qw      = x[9], x[10], x[11], x[12]
        b_gx, b_gy, b_gz    = x[13], x[14], x[15]
        p_ir_x, p_ir_y, p_ir_z = x[16], x[17], x[18]
        q_ir_x, q_ir_y, q_ir_z, q_ir_w = x[19], x[20], x[21], x[22]
        
        t = TransformStamped()
        t.header.stamp = time_stamp
        t.header.frame_id = "ned"
        t.child_frame_id  = "body"
        t.transform.translation.x = float(x_pos)
        t.transform.translation.y = float(y_pos)
        t.transform.translation.z = float(z_pos)
        t.transform.rotation = Quaternion(x=float(qx),
                                          y=float(qy),
                                          z=float(qz),
                                          w=float(qw))
        self.tf_broadcaster.sendTransform(t)

        # Publish as Odometry
        state = Odometry()
        state.header.stamp = t.header.stamp
        state.header.frame_id = "ned"
        state.child_frame_id  = "body"

        state.pose.pose.position.x = float(x_pos)
        state.pose.pose.position.y = float(y_pos)
        state.pose.pose.position.z = float(z_pos)

        state.pose.pose.orientation.x = float(qx)
        state.pose.pose.orientation.y = float(qy)
        state.pose.pose.orientation.z = float(qz)
        state.pose.pose.orientation.w = float(qw)

        pose_cov = np.zeros((6, 6))
        pose_cov[0:3, 0:3] = P[0:3, 0:3]     # position
        pose_cov[3:6, 3:6] = P[9:12, 9:12]   # attitude error
        state.pose.covariance = pose_cov.flatten().tolist()

        state.twist.twist.linear.x = float(v_x)
        state.twist.twist.linear.y = float(v_y)
        state.twist.twist.linear.z = float(v_z)

        velocity_cov = np.zeros((6, 6))
        velocity_cov[0:3, 0:3] = P[3:6, 3:6]  # velocity
        state.twist.covariance = velocity_cov.flatten().tolist()

        self.state_pub.publish(state)

        accel_bias_msg = Vector3Stamped()
        accel_bias_msg.header = state.header
        accel_bias_msg.vector.x = float(b_ax)
        accel_bias_msg.vector.y = float(b_ay)
        accel_bias_msg.vector.z = float(b_az)
        self.accel_bias_pub.publish(accel_bias_msg)

        gyro_bias_msg = Vector3Stamped()
        gyro_bias_msg.header = state.header
        gyro_bias_msg.vector.x = float(b_gx)
        gyro_bias_msg.vector.y = float(b_gy)
        gyro_bias_msg.vector.z = float(b_gz)
        self.gyro_bias_pub.publish(gyro_bias_msg)

        radar_pose_msg = PoseStamped()
        radar_pose_msg.header = state.header
        radar_pose_msg.pose.position.x = float(p_ir_x)
        radar_pose_msg.pose.position.y = float(p_ir_y)
        radar_pose_msg.pose.position.z = float(p_ir_z)
        radar_pose_msg.pose.orientation.x = float(q_ir_x)
        radar_pose_msg.pose.orientation.y = float(q_ir_y)
        radar_pose_msg.pose.orientation.z = float(q_ir_z)
        radar_pose_msg.pose.orientation.w = float(q_ir_w)
        self.radar_pose_pub.publish(radar_pose_msg)

    # EKF main loop. The ES-EKF runs at the frequency of the IMU.
    def imu_callback(self, msg: Imu):
        # Ensure no NaNs in orientation or yaw-rate
        if (
            np.isnan(msg.orientation.x)
            or np.isnan(msg.orientation.y)
            or np.isnan(msg.orientation.z)
            or np.isnan(msg.orientation.w)
            or np.isnan(msg.angular_velocity.z)
        ):
            return

        if not self.initialized:
            return

        t0 = self.get_clock().now()

        # IMU data
        f_b = np.array(
            [[msg.linear_acceleration.x],
             [msg.linear_acceleration.y],
             [msg.linear_acceleration.z]],
            dtype=float,
        )
        w_b = np.array(
            [[msg.angular_velocity.x],
             [msg.angular_velocity.y],
             [msg.angular_velocity.z]],
            dtype=float,
        )

        # Initialize attitude from gravity if not yet done
        if not hasattr(self, "initialized_att") or not self.initialized_att:
            if np.linalg.norm(f_b) < 9.0 or np.linalg.norm(f_b) > 10.5:
                self.get_logger().warn("Drone must be level and not moving for attitude init.")
                return

            self.get_logger().info(f"Initializing attitude from f_b: {f_b.flatten()}")

            fb = f_b / max(1e-6, np.linalg.norm(f_b))
            gb = fb
            roll  = np.arctan2(gb[1, 0], gb[2, 0])
            pitch = np.arctan2(-gb[0, 0], np.sqrt(gb[1, 0]**2 + gb[2, 0]**2))
            yaw   = 0.0  # arbitrary without compass

            q0 = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
            q0 = np.asarray(q0, dtype=float).reshape(4, 1)

            self.eskf.q_hat_ins         = q0
            self.eskf.x_hat_ins[9:13,0] = q0[:,0]
            self.initialized_att          = True

            self.get_logger().info(f"Initialized attitude from gravity: roll={roll:.3f}, pitch={pitch:.3f}")

        # Time update
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.imu_last_stamp is None:
            self.imu_last_stamp = t
            return

        dt = t - self.imu_last_stamp
        self.imu_last_stamp = t
        if dt <= 0.0 or dt > 0.1:
            return

        # Nominal rotation from quaternion (body→NED)
        q = self.eskf.q_hat_ins.flatten()
        R_nb = tf_transformations.quaternion_matrix(q)[:3, :3]

        # Error-state dynamics
        f_nom = f_b - self.eskf.b_acc_ins
        w_nom = w_b - self.eskf.b_ars_ins

        A = self.eskf.generate_A(R_nb, f_nom, w_nom)
        E = self.eskf.generate_E(R_nb)

        Ad = np.eye(self.eskf.num_states) + A * dt
        Qd = (E @ self.Q @ E.T) * dt

        # Predict error state and covariance
        self.eskf.predict(Ad, Qd)

        # INS propagation (nominal)
        g_w = np.array([[0.0], [0.0], [-9.81]])
        self.eskf.ins_propagation(
            self.eskf.x_hat_ins,
            dt,
            f_nom,
            w_nom,
            g_w=g_w,
        )

        # Radar velocity updates
        if self.new_velocity_measurement:
            any_update = False
            for vr, mu_r in zip(self.VR_meas, self.MU_R):
                # Use same R_nb for radar model
                
                v_WI = self.eskf.v_hat_ins.reshape(3, 1)
                R_WI = self.eskf.quaternion_to_rotation_matrix(self.eskf.q_hat_ins)
                p_IR = self.eskf.p_IR
                R_IR = self.eskf.quaternion_to_rotation_matrix(self.eskf.q_IR)
                R_RI = R_IR.T
                w_nom = w_b - self.eskf.b_ars_ins

                H = self.calculate_radar_H(mu_r, R_WI, v_WI, w_nom, p_IR, R_IR)
                h = self.calculate_radar_h(mu_r, R_WI, v_WI, w_nom, p_IR, R_IR)

                e = np.array([[self.vr_sign * vr]], dtype=np.float64) - h

                # self.e_mean += e.item()
                # self.e_std  += e.item() ** 2

                R_meas = np.array([[self.sigma_vr**2]], dtype=np.float64)

                if self.gating_enable:
                    S = float(H @ self.eskf.P_hat_prior @ H.T + R_meas)
                    if S <= 0.0:
                        continue  # degenerate, skip

                    e_scalar = float(e)  # e is 1x1 array
                    gamma = e_scalar * e_scalar / S
                    gate_threshold = self.gate_nsigma ** 2

                    if gamma > gate_threshold:
                        self.get_logger().warn(
                            f"Radar velocity measurement rejected by gating: "
                            f"gamma={gamma:.2f} > threshold={gate_threshold:.2f}"
                        )
                        continue
                if not self.ikf_enable:
                    delta_x_hat_i, _ = self.eskf.correct(e, H, R_meas)
                    self.eskf.update_state_estimate(delta_x_hat_i)
                    any_update = True

                    self.eskf.P_hat_prior       = self.eskf.P_hat.copy()
                    self.eskf.delta_x_hat_prior = self.eskf.delta_x_hat.copy()
                    continue
                # ------------------------------
                # IKF: iterated EKF on this measurement
                # ------------------------------
                # Store covariance prior for this measurement
                P_prior_meas = self.eskf.P_hat_prior.copy()
                delta_norm_prev = np.inf

                for it in range(self.ikf_max_iters):
                    # Recompute linearization around current state
                    v_WI = self.eskf.v_hat_ins.reshape(3, 1)
                    R_WI = self.eskf.quaternion_to_rotation_matrix(self.eskf.q_hat_ins)
                    p_IR = self.eskf.p_IR
                    R_IR = self.eskf.quaternion_to_rotation_matrix(self.eskf.q_IR)
                    w_nom = w_b - self.eskf.b_ars_ins

                    H = self.calculate_radar_H(mu_r, R_WI, v_WI, w_nom, p_IR, R_IR)
                    h = self.calculate_radar_h(mu_r, R_WI, v_WI, w_nom, p_IR, R_IR)

                    e_it = np.array([[self.vr_sign * vr]], dtype=np.float64) - h

                    # For a “true” IKF, covariance is based on the same prior each iteration.
                    # Reset prior before calling correct:
                    self.eskf.P_hat_prior       = P_prior_meas
                    self.eskf.delta_x_hat_prior = np.zeros_like(self.eskf.delta_x_hat_prior)

                    delta_x_hat_i, _ = self.eskf.correct(e_it, H, R_meas)
                    self.eskf.update_state_estimate(delta_x_hat_i)

                    delta_norm = float(np.linalg.norm(delta_x_hat_i))
                    if delta_norm < self.ikf_tol:
                        break
                    if delta_norm > delta_norm_prev:
                        # divergence, stop iterating
                        break
                    delta_norm_prev = delta_norm

                any_update = True

                # After IKF, set prior for the next measurement in this scan
                self.eskf.P_hat_prior       = self.eskf.P_hat.copy()
                self.eskf.delta_x_hat_prior = self.eskf.delta_x_hat.copy()

            # If no measurement was accepted, we still need to advance P_hat
            if not any_update:
                self.eskf.P_hat       = self.eskf.P_hat_prior
                self.eskf.delta_x_hat = self.eskf.delta_x_hat_prior

            # self.i += 1
            # if self.i == 200:
            #     self.e_mean /= self.i
            #     self.e_std = np.sqrt(self.e_std / self.i - self.e_mean**2)
            #     self.get_logger().info(
            #         f"Radar vel residuals mean: {self.e_mean:.4f}, std: {self.e_std:.4f}"
            #     )

            self.new_velocity_measurement = False

        else:
            # No aiding measurements
            self.eskf.P_hat       = self.eskf.P_hat_prior
            self.eskf.delta_x_hat = self.eskf.delta_x_hat_prior

        # Publish state
        t1 = self.get_clock().now()
        dt = (t1 - t0).nanoseconds * 1e-9
        t_msg = msg.header.stamp
        t_ros = rclpy.time.Time.from_msg(t_msg)
        t_ros = t_ros + rclpy.time.Duration(seconds=dt)
        time_stamp = t_ros.to_msg()
        self._publish_state(self.eskf.x_hat_ins, self.eskf.P_hat, time_stamp)

    def update_radar(self, msg: PointCloud2, min_range=1e-2):
        """Extract bearing unit vectors (in radar frame R) and per-return radial speeds."""
        U_list, vr_list = [], []
        for x, y, z, v in pc2.read_points(
            msg, field_names=("x", "y", "z", "velocity"), skip_nans=True
        ):
            r = np.linalg.norm([x, y, z])
            if r < min_range:
                continue
            mu = np.array([x, y, z], dtype=np.float64).reshape(3, 1) / r
            U_list.append(mu)
            vr_list.append(v)

        if not U_list:
            return

        if (self.MU_R is None) or (self.VR_meas is None):
            self.initialized = True
            self.get_logger().info("EKF initialized with first radar measurement.")

        self.MU_R    = np.asarray(U_list, dtype=np.float64)
        self.VR_meas = np.asarray(vr_list, dtype=np.float64)

        self.new_velocity_measurement = True

    def calculate_radar_H(self, mu_r, R_WI, v_WI, w_nom, p_IR, R_IR):
        """
        Build 1x21 Jacobian for Doppler measurement:
        State order: [p(0:3), v(3:6), b_a(6:9), δθ(9:12), b_g(12:15), p_IR(15:18), q_IR(18:21)]
        """
        R_IW = R_WI.T
        R_RI = R_IR.T
        assert np.allclose(R_IR @ R_RI, np.eye(3), atol=1e-6)

        H = np.zeros((1, 21), dtype=np.float64)

        # d e / d v_W = - μ^T R_RI R_IW
        H[0, 3:6] = -(mu_r.reshape(1, 3) @ (R_RI @ R_IW))

        # d e / d δθ = - μ^T R_RI [R_IW v_W]_x
        v_I = R_IW @ v_WI
        S   = -(mu_r.reshape(1, 3) @ (R_RI @ _skew(v_I.flatten())))
        H[0, 9:12] = S

        # d e / d b_g = - μ^T R_RI [p_IR]_x
        H[0, 12:15] = -(mu_r.reshape(1, 3) @ (R_RI @ _skew(p_IR.flatten())))

        # d e / d p_IR 
        H[0, 15:18] = -(mu_r.reshape(1, 3) @ R_RI) @ _skew(w_nom.flatten())

        # d e / d δθ_IR 
        w_nom_skew = _skew(w_nom.flatten())
        s_I = R_IW @ v_WI + w_nom_skew @ p_IR
        v_R_nom = R_RI @ s_I
        H[0, 18:21] = -(mu_r.reshape(1, 3) @ R_RI) @ _skew(v_R_nom.flatten())

        return H

    def calculate_radar_h(self, mu_r, R_WI, v_WI, w_nom, p_IR, R_IR):
        R_IW = R_WI.T
        v_I  = R_IW @ v_WI
        spin = np.cross(w_nom.flatten(), p_IR.flatten()).reshape(3, 1)
        v_R  = R_IR.T @ (v_I + spin)
        return float(-(mu_r.reshape(1, 3) @ v_R))

    def get_initial_P(self):
        self.declare_parameter("parameters.initial_sigma.attitude_deg", [6.0, 6.0, 1e-6])
        self.declare_parameter("parameters.initial_sigma.position",     [1e-6, 1e-6, 1e-6])
        self.declare_parameter("parameters.initial_sigma.velocity",     [1e-1, 1e-1, 1e-1])
        self.declare_parameter("parameters.initial_sigma.accel_bias",   [1e-2, 1e-2, 1e-2])
        self.declare_parameter("parameters.initial_sigma.gyro_bias",    [1e-2, 1e-2, 1e-2])
        self.declare_parameter("parameters.initial_sigma.radar_position",    [0.1, 0.1, 0.1])
        self.declare_parameter("parameters.initial_sigma.radar_attitude_deg", [1.0, 1.0, 1.0])

        sig_att_deg = np.array(
            self.get_parameter("parameters.initial_sigma.attitude_deg").value,
            dtype=float,
        )
        sig_pos = np.array(
            self.get_parameter("parameters.initial_sigma.position").value,
            dtype=float,
        )
        sig_vel = np.array(
            self.get_parameter("parameters.initial_sigma.velocity").value,
            dtype=float,
        )
        sig_ba = np.array(
            self.get_parameter("parameters.initial_sigma.accel_bias").value,
            dtype=float,
        )
        sig_bg = np.array(
            self.get_parameter("parameters.initial_sigma.gyro_bias").value,
            dtype=float,
        )
        sig_radar_pos = np.array(
            self.get_parameter("parameters.initial_sigma.radar_position").value,
            dtype=float,
        )
        sig_radar_att_deg = np.array(
            self.get_parameter("parameters.initial_sigma.radar_attitude_deg").value,
            dtype=float,
        )

        sig_att = np.deg2rad(sig_att_deg)
        sig_radar_att = np.deg2rad(sig_radar_att_deg)

        P_init = np.zeros((21, 21), dtype=np.float64)
        P_init[0:3,   0:3]   = np.diag(sig_pos**2)
        P_init[3:6,   3:6]   = np.diag(sig_vel**2)
        P_init[6:9,   6:9]   = np.diag(sig_ba**2)
        P_init[9:12,  9:12]  = np.diag(sig_att**2)
        P_init[12:15, 12:15] = np.diag(sig_bg**2)
        P_init[15:18, 15:18] = np.diag(sig_radar_pos**2)
        P_init[18:21, 18:21] = np.diag(sig_radar_att**2)

        return P_init


def main():
    rclpy.init()
    node = StateEstimator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

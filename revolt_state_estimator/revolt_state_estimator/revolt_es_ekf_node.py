#!/usr/bin/env python3
"""
ROS2 node that runs an Error-state Extended Kalman Filter using INS and an IMU with an AHRS system.
This is based on the Fossen 2nd edition book, chapter 14.4.2 - Error-state Kalman Filter Using Attitude Measurements.
Fuses:
 - GNSS position (/fix)
 - GNSS heading (/heading)
 - GNSS velocity (/vel)
 - IMU data (/imu/data)
    + yaw
    + yaw rate
    + linear velocity (integrated from linear acceleration)
Publishes:
 - ned → body TF
 - nav_msgs/Odometry with the state estimate
"""

import rclpy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import (
    TransformStamped,
    Quaternion,
)
import tf_transformations
import tf2_ros
import numpy as np
import scipy.linalg

from revolt_state_estimator.es_ekf import ErrorState_ExtendedKalmanFilter
from revolt_state_estimator.revolt_sensor_transforms import Tzyx, Rzyx
from revolt_state_estimator.utils import _skew, ssa, gravity

def quat_xyzw_to_R(qx,qy,qz,qw):
    return tf_transformations.quaternion_matrix([qx,qy,qz,qw])[:3,:3]


class RevoltEKF(Node):
    # Initialize EKF system (START) ==================================================
    def __init__(self):
        # ROS2 Node Setup ----------
        super().__init__("revolt_ekf")

        # EKF Parameters
        self.declare_parameter("revolt_ekf.Q", [0.0] * 12)  # []
        self.declare_parameter(
            "revolt_ekf.R_vel", [0.0, 0.0, 0.0]
        )  # [v_x, v_y, v_z (m/s)]
        self.declare_parameter(
            "revolt_ekf.T_acc", 1000.0
        )  # Eq. 14.195 in Fossen 2nd edition
        self.declare_parameter(
            "revolt_ekf.T_ars", 500.0
        )  # 14.196 in Fossen 2nd edition

        # Extrinsic transformation from radar to IMU
        self.declare_parameter(
            "l_BR_B", [0.0, 0.0, 0.0]
        )
        self.declare_parameter(
            "q_R_B", [0.0, 0.0, 0.0, 1.0]
        )

        _Q = self.get_parameter("revolt_ekf.Q").value
        # _R_head = self.get_parameter("revolt_ekf.R_head").value
        _R_vel = self.get_parameter("revolt_ekf.R_vel").value
        _T_acc = self.get_parameter("revolt_ekf.T_acc").value
        _T_ars = self.get_parameter("revolt_ekf.T_ars").value
        _l_BR_B = self.get_parameter("l_BR_B").value
        _q_R_B = self.get_parameter("q_R_B").value

        # ROS 2 Parameters
        # Publish topics
        self.declare_parameter(
            "revolt_ekf.state_estimate_topic", "/state_estimate/revolt"
        )
        # Subscribe topics
        self.declare_parameter("revolt_ekf.imu_topic", "/imu/data")
        # self.declare_parameter("revolt_ekf.fix_topic", "/fix")
        # self.declare_parameter("revolt_ekf.heading_topic", "/heading")
        self.declare_parameter("revolt_ekf.velocity_topic", "/vel")
        _state_estimate_topic = self.get_parameter(
            "revolt_ekf.state_estimate_topic"
        ).value
        _imu_topic = self.get_parameter("revolt_ekf.imu_topic").value
        _radar_topic = self.get_parameter("revolt_ekf.radar_topic").value

        self.l_BR_B = np.array(_l_BR_B, dtype=float).reshape(3,1)
        qx,qy,qz,qw = _q_R_B
        self.R_RI = quat_xyzw_to_R(qx,qy,qz,qw)   # IMU->Radar
        self.R_IR = self.R_RI.T                   # Radar->IMU

        self.declare_parameter("radar_vr_sign", +1)
        self.vr_sign = int(self.get_parameter("radar_vr_sign").value)

        self.declare_parameter("radar_sigma_vr", 0.35)
        self.sigma_vr = float(self.get_parameter("radar_sigma_vr").value)

        # EKF Setup ----------
        self.new_velocity_measurement = False

        # EKF Initialization variables
        self.initialized = False  # don’t run EKF until first GNSS fix arrives
        self.imu_last_stamp = None  # last IMU timestamp

        self.MU_R = None
        self.VR_meas = None

        # Noise models
        self.Q = np.diag(_Q)
        self.R_vel = np.diag(_R_vel)

        # Latest measurements
        self.latest_velocity = None

        # Error-state Extended Kalman Filter setup
        self.es_ekf = ErrorState_ExtendedKalmanFilter(
            Q=self.Q, T_acc=_T_acc, T_ars=_T_ars
        )

        # ROS2 Interfaces Setup ----------
        # Sensor data subscribers
        self.radar_sub = self.create_subscription(
            PointCloud2, _radar_topic, self.update_radar, 1
        )
        # IMU
        self.imu_sub = self.create_subscription(Imu, _imu_topic, self.imu_callback, 1)
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # State publisher
        self.state_pub = self.create_publisher(Odometry, _state_estimate_topic, 10)

        # Debugging ----------
        np.set_printoptions(
            linewidth=200,
            precision=6,  # adjust as you like
            suppress=True,  # so small floats don’t go to scientific notation
        )

        self.get_logger().info(
            f"                                   \n"
            f"EKF Parameters:                    \n"
            f" Q:                                \n"
            f" {self.Q}                          \n"
            f" R_head:                           \n"
            f" {self.R_head}                     \n"
            f" R_vel:                            \n"
            f" {self.R_vel}                      \n"
            f" T_acc: {_T_acc}               \n"
            f" T_ars: {_T_ars}               \n"
            f"                                   \n"
            f"Publisher topics:                     \n"
            f"State estimate topic: {_state_estimate_topic} \n"
            f"Subscribe topics:                  \n"
            f"IMU topic: {_imu_topic} \n"
            f"Radar topic: {_radar_topic} \n"
            f"Extrinsic transformation (l_BR_B): {_l_BR_B} \n"
            f"Extrinsic transformation (q_R_B): {_q_R_B} \n"
            f"                                   \n"
        )
        self.get_logger().info(
            "EKF waiting for IMU and radar measurements to start..."
        )

    # Initialize EKF system (STOP) ==================================================

    def _publish_state(self, x, P):

        # State is a 15‑element vector:
        # x_hat_ins = [
        #   p_x,      p_y,      p_z,        # position in navigation frame (m)
        #   v_x,      v_y,      v_z,        # velocity in navigation frame (m/s)
        #   b_acc_x,  b_acc_y,  b_acc_z,    # accelerometer biases (m/s²)
        #   ϕ,        θ,        ψ,          # attitude Euler angles: roll, pitch, yaw (rad)
        #   b_gyro_x, b_gyro_y, b_gyro_z    # gyroscope biases (rad/s)
        # ]
        # P is the covariance matrix of the full state estimate.

        x_pos, y_pos, z_pos = x[0], x[1], x[2]  # position in NED frame (m)
        v_x, v_y, v_z = x[3], x[4], x[5]  # velocity in NED frame (m/s)
        roll, pitch, yaw = (
            x[9],
            x[10],
            x[11],
        )  # attitude Euler angles: roll, pitch, yaw (rad)
        # Broadcast NED → imu TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "ned"
        t.child_frame_id = "body"
        t.transform.translation.x = float(x_pos)
        t.transform.translation.y = float(y_pos)
        t.transform.translation.z = float(z_pos)
        # yaw → quaternion
        q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.tf_broadcaster.sendTransform(t)

        # 3) Publish ekf/state as Odometry 
        state = Odometry()
        state.header.stamp = t.header.stamp
        state.header.frame_id = "ned"
        state.child_frame_id = "body"

        state.pose.pose.position.x = float(x_pos)
        state.pose.pose.position.y = float(y_pos)
        state.pose.pose.position.z = float(z_pos)

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        state.pose.pose.orientation.x  = float(qx)
        state.pose.pose.orientation.y  = float(qy)
        state.pose.pose.orientation.z  = float(qz)
        state.pose.pose.orientation.w  = float(qw)

        # Covariance matrix
        pose_cov = np.zeros((6, 6))
        pose_cov[0:3, 0:3] = P[0:3, 0:3]
        pose_cov[3:6, 3:6] = P[9:12, 9:12]
        state.pose.covariance = pose_cov.flatten().tolist()

        state.twist.twist.linear.x = float(v_x)  # North or X velocity (m/s)
        state.twist.twist.linear.y = float(v_y)  # East or Y velocity (m/s)
        state.twist.twist.linear.z = float(v_z)  # Down or Z velocity (m/s)

        # Could feed forward the imu angular velocity, but will not do it to avoid confusion
        # state.twist.twist.angular.x = float(self.latest_roll_rate)  # Roll rate (rad/s)
        # state.twist.twist.angular.y = float(self.latest_pitch_rate)  # Pitch rate (rad/s)
        # state.twist.twist.angular.z = float(self.latest_yaw_rate)  # Yaw rate (rad/s)
        state.twist.twist.angular.x = 0.0  # Roll rate (rad/s)
        state.twist.twist.angular.y = 0.0  # Pitch rate (rad/s)
        state.twist.twist.angular.z = 0.0  # Yaw rate (rad/s)

        velocity_cov = np.zeros((6, 6))
        velocity_cov[0:3, 0:3] = P[3:6, 3:6]  # v_x variance
        state.twist.covariance = velocity_cov.flatten().tolist()

        self.state_pub.publish(state)

    # EKF main loop. The ES-EKF runs at the frequency of the IMU.
    def imu_callback(self, msg: Imu):
        # Ensure the value we get is NOT NaN
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

        # Update listeners
        # self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=3600))
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # AHRS angles. These are not being used atm, but according to Fossens implementation it should.
        # q = msg.orientation
        # roll_imu, pitch_imu, yaw_imu = tf_transformations.euler_from_quaternion(
        #     [q.x, q.y, q.z, q.w]
        # )
        #  # Force rpy to be in [-pi, pi)
        # roll_imu = ssa(roll_imu) 
        # pitch_imu = ssa(pitch_imu)
        # yaw_imu = ssa(yaw_imu)

        # Specific force (acceleration in body frame)
        f_msg = msg.linear_acceleration
        f_vec = np.array([f_msg.x, f_msg.y, f_msg.z]).reshape(3, 1)
        f_imu = f_vec - self.es_ekf.b_acc_ins

        # Attitude rate (body frame)
        w_msg = msg.angular_velocity
        w_vec = np.array([w_msg.x, w_msg.y, w_msg.z]).reshape(3, 1)
        w_imu = w_vec - self.es_ekf.b_ars_ins
        self._last_w_imu = w_imu.copy()

        # Current time
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not hasattr(self, "imu_last_stamp") or self.imu_last_stamp is None:
            self.imu_last_stamp = t
            return

        dt = t - self.imu_last_stamp
        self.imu_last_stamp = t

        # Rotation and transformation matrices from body to NED
        # R_bn = self.get_rotation_and_translation_from_tf("body", "ned")
        # Use EKF's own nominal attitude for linearization & mechanization to allow feedback
        # According to fossen we should use the AHRS measurements, but this leads to numerical instability.
        roll_est, pitch_est, yaw_est = self.es_ekf.theta_hat_ins.flatten()
        R = Rzyx(roll_est, pitch_est, yaw_est)   # body -> NED
        T = Tzyx(roll_est, pitch_est)            # Euler kinematics


        # System dynamics to implement the 15-state error-state model
        # ∂x_dot = A(t) * ∂x + E(t) * w (Eq. 14.188 in Fossen 2nd ed.)
        # ∂y = C * ∂x + ε (Eq. 14.189 in Fossen 2nd ed.)
        A = self.es_ekf.generate_A(R, T, f_imu)  # Eq. 14.192 in Fossen 2nd ed.
        E = self.es_ekf.generate_E(R, T)  # Eq. 14.193 in Fossen 2nd ed.

        # Discretization according to Fossen 2nd ed. Eq. 14.201
        Ad = np.eye(self.es_ekf.num_states) + A * dt
        Ed = E * dt

        # Checking which aiding measurements we have
        O3 = np.zeros((3, 3))
        I3 = np.eye(3)
        zs, Cs, Rs = [], [], []

        # Radar velocity
        if self.new_velocity_measurement:
            e, H = self.calculate_radar_velocity_error_and_H()

            print("Radar velocity error:", e)

            # C_vel = np.hstack([O3, I3, O3, O3, O3])
            zs.append(e)
            Cs.append(H)
            Rs.append(self.R_vel)
            self.new_velocity_measurement = False

        # If we have any aiding measurements, we perform the correction step, if not we just propagate the state.
        if zs:
            z_total = np.vstack(zs)  # combined measurement vector
            Cd_total = np.vstack(Cs)  # combined measurement matrix, Cd = C
            R_noise = scipy.linalg.block_diag(
                *Rs
            )  # combined measurement noise covariance matrix

            # Corrector: delta_x_hat[k] and P_hat[k]
            delta_x_hat, P_hat = self.es_ekf.correct(z_total, Cd_total, R_noise)

            # INS reset: x_ins[k]
            x_hat_ins = self.es_ekf.update_state_estimate(delta_x_hat)

            # Publish the state estimate
            self._publish_state(x_hat_ins, P_hat)

        else:
            # No aiding measurements
            self.es_ekf.P_hat = self.es_ekf.P_hat_prior

        # Predictor: P_hat_prior[k+1]
        delta_x_hat_prior, P_hat_prior = self.es_ekf.predict(Ad, Ed)

        # INS propagation: x_hat_ins[k+1]
        g_n = (np.array([0.0, 0.0, gravity(self.latest_latitude)])).reshape(
            3, 1
        )  # gravity vector in navigation frame
        x_hat_ins = self.es_ekf.ins_propagation(
            self.es_ekf.x_hat_ins,
            dt,
            R,
            T,
            f_imu,
            w_imu,
            g_n=g_n,
        )
        # Publish the state estimate
        self._publish_state(x_hat_ins, P_hat_prior)

    def update_radar(self, msg: PointCloud2, min_range=1e-3):
        """Extract LOS unit vectors (in radar frame R) and per-return radial speeds."""
        U_list, vr_list = [], []
        for x, y, z, v in pc2.read_points(msg, field_names=("x", "y", "z", "velocity"), skip_nans=True):
            r = np.sqrt(x*x + y*y + z*z)
            if r < min_range:
                continue
            U_list.append([x/r, y/r, z/r])
            vr_list.append(v)
        if not U_list:
            return  # No valid points

        if (self.MU_R is None) or (self.VR_meas is None):
            self.initialized = True
            self.get_logger().info("EKF initialized with first radar measurement.")
        
        self.MU_R = np.asarray(U_list, dtype=np.float64)
        self.VR_meas = np.asarray(vr_list, dtype=np.float64)

        self.new_velocity_measurement = True

    def calculate_radar_velocity_error_and_H(self):
        # 2) State rotations
        roll, pitch, yaw = self.es_ekf.theta_hat_ins.flatten()
        R_WI = Rzyx(roll, pitch, yaw)      # WRI
        R_IW = R_WI.T                      # IRW
        R_RI = self.R_RI                   # RRI
        p_IR = self.l_BR_B                 # IpIR, expressed in I
        w_I  = getattr(self, "_last_w_imu", np.zeros((3,1)))  # IωWI (bias-corrected)

        # 3) Predicted radar linear velocity per Eq. (8): v_R = R_RI( R_IW v_W + (w_I)× p_IR )
        v_W  = self.es_ekf.v_hat_ins.reshape(3,1)        # WvWI (expressed in W)
        v_I  = R_IW @ v_W                                # WR^T_I WvWI  == R_IW v_W
        v_R_pred = R_RI @ ( v_I + np.cross(w_I.flatten(), p_IR.flatten()).reshape(3,1) )

        # 4) Residuals per Eq. (9): e_i = - μ_i^T v_R_pred - \tilde v_{r,i}
        # If your driver flips sign, vr_sign handles it.
        e = (- self.MU_R @ v_R_pred).flatten() - self.vr_sign * self.VR_meas   # shape (N,)
        N = e.size

        # 5) Build stacked Jacobian H per Eq. (10)
        # State order: [p(0:3), v(3:6), b_a(6:9), eul(9:12), b_g(12:15)]
        H = np.zeros((N, 15), dtype=np.float64)

        # d e / d v_W = - μ^T R_RI R_IW    (Eq. 10)
        H[:, 3:6] = - (self.MU_R @ (R_RI @ R_IW))

        # d e / d b_g = - μ^T R_RI [p_IR]_x   (Eq. 10)
        H[:, 12:15] = - (self.MU_R @ (R_RI @ _skew(p_IR.flatten())))


        term = R_IW @ v_W                     # (IRW WvWI)
        S = - (self.MU_R @ (R_RI @ _skew(term.flatten())))   # shape (N,3)
        H[:, 9:12] = S

        return e, H


def main():
    rclpy.init()
    node = RevoltEKF()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

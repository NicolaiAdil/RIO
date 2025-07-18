#!/usr/bin/env python3
"""
ROS2 node that runs an Extended Kalman Filter for a simple ship model.
Fuses:
 - GNSS position (/fix)
 - GNSS heading (/heading)
 - GNSS velocity (/vel)
 - IMU data (/imu/data) 
    + yaw
    + yaw rate
    + linear velocity (integrated from linear acceleration) 
Publishes:
 - world → body TF
 - ekf/state Odometry (x, y, yaw, v, yaw_rate)
"""

import rclpy
import rclpy.logging
from rclpy.node import Node
from custom_msgs.msg import StateEstimate
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64
from geometry_msgs.msg import TransformStamped, Quaternion, QuaternionStamped, TwistStamped
import tf_transformations
import tf2_ros
import numpy as np
import pymap3d as pm

from revolt_state_estimator.revolt_model import ReVoltModel
from revolt_state_estimator.revolt_sensor_transforms import h_fix, h_head, h_vel, h_imu
from revolt_state_estimator.es_ekf import ErrorState_ExtendedKalmanFilter, ssa, Tzyx, Rzyx, gravity
from revolt_state_estimator.utils import unwrap



class RevoltEKF(Node):
    # Initialize EKF system (START) ==================================================
    def __init__(self):
        # ROS2 Node Setup ----------
        super().__init__('revolt_ekf')

        # Ship Model Parameters
        self.declare_parameter('revolt_model.m',                    0.0) # mass [kg]
        self.declare_parameter('revolt_model.dimensions',          [0.0, 0.0]) # [radius (m), length (m)]
        self.declare_parameter('revolt_model.thruster_placement',  [0.0, 0.0]) # [x,y] in body frame [m]
        self.declare_parameter('revolt_model.velocity_linear_max',  0.0) # max forward speed [m/s]
        self.declare_parameter('revolt_model.velocity_angular_max', 0.0) # max yaw rate [rad/s]
        _m                    = self.get_parameter('revolt_model.m').value
        _dimensions           = self.get_parameter('revolt_model.dimensions').value
        _thruster_placement   = self.get_parameter('revolt_model.thruster_placement').value
        _velocity_linear_max  = self.get_parameter('revolt_model.velocity_linear_max').value
        _velocity_angular_max = self.get_parameter('revolt_model.velocity_angular_max').value

        # EKF Parameters
        self.declare_parameter('revolt_ekf.Q',      [0.0]*5) # [x (m), y (m), yaw (rad), v (m/s), w_yaw (rad/s)]
        self.declare_parameter('revolt_ekf.R_fix',  [0.0, 0.0]) # [x (m), y (m)]
        self.declare_parameter('revolt_ekf.R_head', [0.0]) # [yaw (rad)]
        self.declare_parameter('revolt_ekf.R_vel',  [0.0]) # [v (m/s)]
        self.declare_parameter('revolt_ekf.R_imu',  [0.0, 0.0, 0.0]) # [yaw (rad), w_yaw (rad/s)]
        self.declare_parameter('revolt_ekf.pred_freq', 0.0) # Hz
        _Q         = self.get_parameter('revolt_ekf.Q').value
        _R_fix     = self.get_parameter('revolt_ekf.R_fix').value # NOTE: Not using the static one as the GNSS has dynamic covariance matrix inbuilt in sensor itself
        _R_head    = self.get_parameter('revolt_ekf.R_head').value
        _R_vel     = self.get_parameter('revolt_ekf.R_vel').value
        _R_imu     = self.get_parameter('revolt_ekf.R_imu').value
        _pred_freq = self.get_parameter('revolt_ekf.pred_freq').value

        # ReVolt ship model setup ----------
        self.revolt_model = ReVoltModel(
            m=_m,
            r=_dimensions[0],
            l=_dimensions[1],
            thruster_pos=_thruster_placement,
            v_lin_max=_velocity_linear_max,
            v_ang_max=_velocity_angular_max,
        )

        # EKF Setup ----------
        self.new_fix_measurement = False
        self.new_heading_measurement = False
        self.new_velocity_measurement = False

        # EKF Initialization variables
        self.initialized = False           # don’t run EKF until first GNSS fix arrives
        self.imu_last_stamp = None  # last IMU timestamp
        self.u           = np.zeros(2)     # control input [effort, angle]
        # self.dt          = 1.0/_pred_freq  # prediction timestep

        # Unwrapped yaw states for heading continuity
        self.yaw_measured_unwrapped_head = 0.0
        self.yaw_measured_unwrapped_imu  = 0.0

        # One-shot IMU↔GNSS yaw alignment, 
        self.gnss_yaw_offset = None
        self.gnss_yaw_last   = 0.0
        
        self.imu_yaw_offset  = None
        self.imu_yaw_last    = 0.0

        # IMU Integrates acceleration
        # When we get new GNSS velocity measurement, we will reset the velocity IMU has to estimate
        # If GNSS is away for to long, IMU can start dead reconing and then it is important to limit its speed
        self.v_integrated = 0.0
        self.imu_update_last = 0.0
        self.imu_max_v = 5.0 # [m/s]

        # Noise models
        self.Q = np.diag(_Q)
        self.R_fix  = np.diag(_R_fix) # NOTE: Not using the static one as the GNSS has dynamic covariance matrix inbuilt in sensor itself
        self.R_head = np.diag(_R_head)
        self.R_vel  = np.diag(_R_vel)
        self.R_imu  = np.diag(_R_imu)

        # Latest measurements
        self.latest_fix = None
        self.latest_velocity = None
        self.latest_heading = None 
        self.latest_latitude = None # For calculating gravity
        
        # Error-state Extended Kalman Filter setup
        self.es_ekf = ErrorState_ExtendedKalmanFilter(
            Q=self.Q,
            R=self.R_fix,
            T_acc = 1000,
            T_ars = 500
        )
        
        # ROS2 Interfaces Setup ----------
        # Control signal subscribers
        self.u_effort_sub = self.create_subscription(Float64, '/tau_m',     self.u_effort_cb, 1)
        self.u_angle_sub  = self.create_subscription(Float64, '/tau_delta', self.u_angle_cb,  1)
        # Sensor data subscribers
        # GNSS
        self.fix_sub  = self.create_subscription(NavSatFix,         '/fix',      self.update_fix,     1)
        self.head_sub = self.create_subscription(QuaternionStamped, '/heading',  self.update_heading, 1)
        self.vel_sub  = self.create_subscription(TwistStamped,      '/vel',      self.update_vel,     1)
        # IMU
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.update_imu, 1)
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Timer for predict()
        # self.timer = self.create_timer(self.dt, self.estimate_state)
        # State publisher
        self.state_pub = self.create_publisher(StateEstimate, '/state_estimate/revolt', 10)

        # Debugging ----------
        np.set_printoptions(
            linewidth=200, 
            precision=6, # adjust as you like
            suppress=True # so small floats don’t go to scientific notation
        )  

        self.get_logger().info(
            f"                                   \n" \
            f"EKF Parameters:                    \n" \
            f" Q:                                \n" \
            f" {self.Q}                          \n" \
            f" R_fix:               {self.R_fix} \n" \
            f" R_head:              {self.R_head}\n" \
            f" R_imu:               {self.R_imu} \n" \
            f"                                   \n" \
        )
        self.get_logger().info("EKF waiting for first GNSS '/fix' position topic before predictions...")
    # Initialize EKF system (STOP) ==================================================



    # Control signal callback functions (START) ==================================================
    def u_effort_cb(self, msg: Float64):
        # Ensure the value we get is NOT NaN
        if np.isnan(msg.data):
            return
        
        self.u[0] = msg.data

    def u_angle_cb(self, msg: Float64):
        # Ensure the value we get is NOT NaN
        if np.isnan(msg.data):
            return
        
        self.u[1] = msg.data
    # Control signal callback functions (STOP) ==================================================

    def _publish_state(self, x):

        # State is a 15‑element vector:
        # x_hat_ins = [
        #   p_x,      p_y,      p_z,        # position in navigation frame (m)
        #   v_x,      v_y,      v_z,        # velocity in navigation frame (m/s)
        #   b_acc_x,  b_acc_y,  b_acc_z,    # accelerometer biases (m/s²)
        #   ϕ,        θ,        ψ,          # attitude Euler angles: roll, pitch, yaw (rad)
        #   b_gyro_x, b_gyro_y, b_gyro_z    # gyroscope biases (rad/s)
        # ] 

        x_pos, y_pos, z_pos = x[0], x[1], x[2]  # position in navigation frame (m)
        roll, pitch, yaw = x[6], x[7], x[8]  # attitude Euler angles: roll, pitch, yaw (rad)
        v_x, v_y, v_z = x[3], x[4], x[5]  # velocity in navigation frame (m/s)
        linear_velocity = np.sqrt(v_x**2 + v_y**2)  # magnitude of velocity (ignoring z)

        # Broadcast world → body TF
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id  = 'body'
        t.transform.translation.x = float(x_pos)
        t.transform.translation.y = float(y_pos)
        t.transform.translation.z = float(z_pos)
        # yaw → quaternion
        q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.tf_broadcaster.sendTransform(t)

        # 3) Publish ekf/state as Odometry
        state = StateEstimate()
        state.header.stamp     = t.header.stamp
        state.header.frame_id  = 'world'
        state.child_frame_id   = 'body'

        # extract from your state vector x_pred = [x, y, yaw, v, w]
        state.x                = float(x_pos)  # world‐frame X
        state.y                = float(y_pos)  # world‐frame Y
        state.yaw              = float(z_pos)  # heading (rad)

        state.linear_velocity  = float(linear_velocity)  # forward speed (m/s)
        state.angular_velocity = 0.0  # yaw rate   (rad/s)
        self.state_pub.publish(state)

    # EKF callback functions (START) ==================================================
    def update_imu(self, msg: Imu):
        # Ensure the value we get is NOT NaN
        if (
            np.isnan(msg.orientation.x) or 
            np.isnan(msg.orientation.y) or
            np.isnan(msg.orientation.z) or
            np.isnan(msg.orientation.w) or
            np.isnan(msg.angular_velocity.z)
            ):
            return
        
        if not self.initialized:
            return

        # AHRS angles
        q = msg.orientation
        roll_imu, pitch_imu, yaw_imu = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Specific force (acceleration in body frame)
        f_msg = msg.linear_acceleration
        f_imu = np.array([f_msg.x, f_msg.y, f_msg.z]) - self.es_ekf.b_acc_ins

        # Attitude rate
        w_msg = msg.angular_velocity
        w_imu = np.array([w_msg.x, w_msg.y, w_msg.z]) - self.es_ekf.b_ars_ins

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not hasattr(self, 'imu_last_stamp') or self.imu_last_stamp is None:
            self.imu_last_stamp = t
            return
        
        dt = t - self.imu_last_stamp
        self.imu_last_stamp = t

        R = Rzyx(roll_imu, pitch_imu, yaw_imu)  # Rotation matrix from body to navigation frame
        T = Tzyx(roll_imu, pitch_imu)  # Transformation matrix from body to navigation frame

        A = self.es_ekf.generate_A(R, T)  # State transition matrix
        Ad = np.eye(self.es_ekf.num_states) + A * dt

        E = self.es_ekf.generate_E(R, T)
        Ed = E * dt

        O3 = np.zeros((3, 3))
        I3 = np.eye(3)

        ys, Cs, Rs = [], [], []

        # GNSS position (first 3 states)
        if self.new_fix_measurement:
            y_fix = self.latest_fix
            C_fix = np.hstack([I3, O3, O3, O3, O3])
            R_fix = self.R_fix
            ys.append(y_fix); Cs.append(C_fix); Rs.append(R_fix)
            self.new_fix_measurement = False

        # GNSS velocity (states 3–5)
        if self.new_velocity_measurement:
            y_vel = self.latest_velocity
            C_vel = np.hstack([O3, I3, O3, O3, O3])
            R_vel = self.R_vel
            ys.append(y_vel); Cs.append(C_vel); Rs.append(R_vel)
            self.new_velocity_measurement = False

        # GNSS heading (state 11)
        if self.new_heading_measurement:
            y_head = self.latest_heading 
            C_head = np.zeros((3, 15))
            C_head[2, 11] = 1
            R_head = self.R_head

            ys.append(y_head); Cs.append(C_head); Rs.append(R_head)
            self.new_heading_measurement = False
        
        if ys:
            z = np.vstack(ys)    # combined measurement vector
            Cd = np.vstack(Cs)    # combined measurement matrix, Cd = C
            R_noise = np.block([[R] for R in Rs])  # combined measurement noise covariance matrix


            self.get_logger().info(f"IMU update → z={z.flatten()}")
            self.get_logger().info(f"IMU update → C={Cd}")            

            # Corrector: delta_x_hat[k] and P_hat[k]
            delta_x_hat, P_hat = self.es_ekf.correct(z, Cd)

            # INS reset: x_ins[k]
            x_hat_ins = self.es_ekf.update_state_estimate(delta_x_hat)

            # Publish the state estimate
            self._publish_state(x_hat_ins)

        else:
            # No aiding measurements
            self.es_ekf.P_hat = self.es_ekf.P_hat_prior

        # Predictor: P_hat_prior[k+1]
        delta_x_hat_prior, P_hat_prior = self.es_ekf.predict(Ad, Ed)

        # INS propagation: x_hat_ins[k+1]
        x_hat_ins = self.es_ekf.ins_propagation(
            delta_x_hat_prior, dt, R, T, f_imu, w_imu, g_n=np.array([0.0, 0.0, gravity(self.latest_latitude)])
        )
        # Publish the state estimate
        self._publish_state(x_hat_ins)

        # Debugging ----------
        #self.get_logger().info(f"IMU update → z=[yaw:{imu_yaw_offset:.3f}, v_integrated: {self.v_integrated:.3f} w:{w_imu:.3f}], x_post={x_post}")

    def update_fix(self, msg: NavSatFix):
        # Ensure the value we get is NOT NaN
        if np.isnan(msg.latitude) or np.isnan(msg.longitude):
            return

        if not self.initialized:
            # set ENU origin
            self.ref_lat, self.ref_lon = msg.latitude, msg.longitude
            self.initialized = True
            self.get_logger().info("EKF initialized at first GNSS position")
            self.get_logger().info("EKF now waiting for GNSS and IMU heading alignment")
            return

        # Convert to ENU measurement (east, north, up), ignore up
        self.latest_latitude = msg.latitude

        e, n, u = pm.geodetic2enu(
            msg.latitude, msg.longitude, 0.0,
            self.ref_lat, self.ref_lon, 0.0
        )

        # 2) configure EKF measurement model & noise
        cov = msg.position_covariance
        print(f"GNSS covariance: {cov}")
        var_e = cov[0]  # latitude variance but because we're in ENU this maps to north; swap if needed
        var_n = cov[4]  # longitude variance mapping to east
        var_u = cov[8]  # up variance, not used here
        R_fix_msg = np.array([[var_e, 0.0, 0.0],
                              [0.0, var_n, 0.0],
                              [0.0, 0.0, var_u]])
        #self.ekf.R = self.R_fix # NOTE: Not using the static one as the GNSS has dynamic covariance matrix inbuilt in sensor itself
        self.es_ekf.R = R_fix_msg
        self.es_ekf.h = h_fix

        # 3) Perform the correction step
        z = np.zeros(15)
        z[0] = e
        z[1] = n
        z[2] = u 
        self.latest_fix = z
        self.new_fix_measurement = True

        # Debugging ----------
        #self.get_logger().info(f"Fix update → z=[{e:.2f}, {n:.2f}], x_post={x_post}")

    def update_heading(self, msg: QuaternionStamped):
        # Ensure the value we get is NOT NaN
        if (
            np.isnan(msg.quaternion.x) or 
            np.isnan(msg.quaternion.y) or
            np.isnan(msg.quaternion.z) or
            np.isnan(msg.quaternion.w)
            ):
            return
        
        if not self.initialized:
            return

        # Get GNSS yaw
        q = msg.quaternion
        _, _, yaw_gnss = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])


        z = np.zeros(15)
        z[11] = yaw_gnss 
        
        self.latest_heading = z
        self.new_heading_measurement = True

        # x_post, P_post = self.ekf.update(z)

        # Debugging ----------
        #self.get_logger().info(f"Heading update → z=[{gnss_yaw_offset:.3f}], x_post={x_post}")

    def update_vel(self, msg: TwistStamped):
        # Ensure the value we get is NOT NaN
        if (
            np.isnan(msg.twist.linear.x) or 
            np.isnan(msg.twist.linear.y) or
            np.isnan(msg.twist.angular.z)
            ):
            return
        
        # reject all‐zero bursts (bad data that sometimes comes up from linear velocity)
        if (
            msg.twist.linear.x  == 0.0 or 
            msg.twist.linear.y  == 0.0 
            ):
            return
    
        if not self.initialized:
            return

        # Extract GNSS‐reported velocity in world frame
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z 

        # 3) Perform the correction step
        z = np.zeros(15)
        z[3] = vx
        z[4] = vy
        z[5] = vz
        self.latest_velocity = z
        self.new_velocity_measurement = True

        # Debugging ----------
        #self.get_logger().info(f"Vel update → z=[{v_body:.3f}], x_post={x_post}")
    
    # EKF callback functions (STOP) ==================================================

def main():
    rclpy.init()
    node = RevoltEKF()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

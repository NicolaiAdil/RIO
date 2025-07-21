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
import scipy.linalg

from revolt_state_estimator.revolt_model import ReVoltModel
from revolt_state_estimator.revolt_sensor_transforms import h_fix, h_head, h_vel, h_imu
from revolt_state_estimator.es_ekf import ErrorState_ExtendedKalmanFilter, ssa, Tzyx, Rzyx, gravity
from revolt_state_estimator.utils import unwrap



class RevoltEKF(Node):
    # Initialize EKF system (START) ==================================================
    def __init__(self):
        # ROS2 Node Setup ----------
        super().__init__('revolt_ekf')

        # EKF Parameters
        self.declare_parameter('revolt_ekf.Q',      [0.0]*12) # []
        self.declare_parameter('revolt_ekf.R_fix',  [0.0, 0.0, 0.0]) # [x (m), y (m), z (m)]
        self.declare_parameter('revolt_ekf.R_head', [0.0]) # [yaw (rad)]
        self.declare_parameter('revolt_ekf.R_vel',  [0.0, 0.0, 0.0]) # [v_x, v_y, v_z (m/s)]
        self.declare_parameter('revolt_ekf.T_acc', 1000.0) # Eq. 14.195 in Fossen 2nd edition
        self.declare_parameter('revolt_ekf.T_ars', 500.0) # 14.196 in Fossen 2nd edition
        _Q         = self.get_parameter('revolt_ekf.Q').value
        _R_fix     = self.get_parameter('revolt_ekf.R_fix').value # NOTE: Not using the static one as the GNSS has dynamic covariance matrix inbuilt in sensor itself
        _R_head    = self.get_parameter('revolt_ekf.R_head').value
        _R_vel     = self.get_parameter('revolt_ekf.R_vel').value
        _T_acc     = self.get_parameter('revolt_ekf.T_acc').value
        _T_ars     = self.get_parameter('revolt_ekf.T_ars').value

        # ROS 2 Parameters
        # Publish topics
        self.declare_parameter('revolt_ekf.state_estimate_topic', '/state_estimate/revolt')
        # Subscribe topics
        self.declare_parameter('revolt_ekf.imu_topic', '/imu/data')
        self.declare_parameter('revolt_ekf.fix_topic', '/fix')
        self.declare_parameter('revolt_ekf.heading_topic', '/heading')
        self.declare_parameter('revolt_ekf.velocity_topic', '/vel')
        _state_estimate_topic = self.get_parameter('revolt_ekf.state_estimate_topic').value
        _imu_topic = self.get_parameter('revolt_ekf.imu_topic').value
        _fix_topic = self.get_parameter('revolt_ekf.fix_topic').value
        _heading_topic = self.get_parameter('revolt_ekf.heading_topic').value
        _velocity_topic = self.get_parameter('revolt_ekf.velocity_topic').value
    

        # EKF Setup ----------
        self.new_fix_measurement = False
        self.new_heading_measurement = False
        self.new_velocity_measurement = False

        # EKF Initialization variables
        self.initialized = False           # don’t run EKF until first GNSS fix arrives
        self.imu_last_stamp = None  # last IMU timestamp

        # Noise models
        self.Q = np.diag(_Q)
        self.R_fix  = np.diag(_R_fix) # NOTE: This is just a current guess. We will update it with the GNSS covariance matrix
        self.R_head = np.diag(_R_head)
        self.R_vel  = np.diag(_R_vel)

        # Latest measurements
        self.latest_fix = None
        self.latest_velocity = None
        self.latest_heading = None 
        self.latest_yaw_rate = None
        self.latest_latitude = None # For calculating gravity
        
        # Error-state Extended Kalman Filter setup
        self.es_ekf = ErrorState_ExtendedKalmanFilter(
            Q=self.Q,
            T_acc = _T_acc,
            T_ars = _T_ars
        )
        
        # ROS2 Interfaces Setup ----------
        # Sensor data subscribers
        # GNSS
        self.fix_sub  = self.create_subscription(NavSatFix,         _fix_topic,      self.update_fix,     1)
        self.head_sub = self.create_subscription(QuaternionStamped, _heading_topic,  self.update_heading, 1)
        self.vel_sub  = self.create_subscription(TwistStamped,      _velocity_topic,      self.update_vel,     1)
        # IMU
        self.imu_sub = self.create_subscription(Imu, _imu_topic, self.imu_callback, 1)
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # State publisher
        self.state_pub = self.create_publisher(StateEstimate, _state_estimate_topic, 10)

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
            f" R_fix:                            \n" \
            f" {self.R_fix}                      \n" \
            f" R_head:                           \n" \
            f" {self.R_head}                     \n" \
            f" R_vel:                            \n" \
            f" {self.R_vel}                      \n" \
            f" T_acc: {_T_acc}               \n" \
            f" T_ars: {_T_ars}               \n" \
            f"                                   \n" \
            f"State estimate topic: {_state_estimate_topic} \n" \
            f"IMU topic: {_imu_topic} \n" \
            f"GNSS fix topic: {_fix_topic} \n" \
            f"GNSS heading topic: {_heading_topic} \n" \
            f"GNSS velocity topic: {_velocity_topic} \n" \
            f"                                   \n" \
        )
        self.get_logger().info("EKF waiting for first GNSS '/fix' position topic before predictions...")
    # Initialize EKF system (STOP) ==================================================

    def _publish_state(self, x):

        # State is a 15‑element vector:
        # x_hat_ins = [
        #   p_x,      p_y,      p_z,        # position in navigation frame (m)
        #   v_x,      v_y,      v_z,        # velocity in navigation frame (m/s)
        #   b_acc_x,  b_acc_y,  b_acc_z,    # accelerometer biases (m/s²)
        #   ϕ,        θ,        ψ,          # attitude Euler angles: roll, pitch, yaw (rad)
        #   b_gyro_x, b_gyro_y, b_gyro_z    # gyroscope biases (rad/s)
        # ] 
        # print(f"State estimate: {x}, \n Dimensions: {type(x)}")

        x_pos, y_pos, z_pos = x[0], x[1], x[2]  # position in navigation frame (m)
        roll, pitch, yaw = x[9], x[10], x[11]  # attitude Euler angles: roll, pitch, yaw (rad)
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
        state.yaw              = float(yaw)  # heading (rad)

        state.linear_velocity  = float(linear_velocity)  # forward speed (m/s)
        state.angular_velocity = float(self.latest_yaw_rate)  # yaw rate   (rad/s)
        self.state_pub.publish(state)

    # EKF main loop. The ES-EKF runs at the frequency of the IMU.
    def imu_callback(self, msg: Imu):
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
        f_vec = np.array([f_msg.x, f_msg.y, f_msg.z]).reshape(3, 1) 
        f_imu = f_vec - self.es_ekf.b_acc_ins                       

        # Attitude rate (body frame)
        w_msg = msg.angular_velocity
        w_vec = np.array([w_msg.x, w_msg.y, w_msg.z]).reshape(3, 1)   
        w_imu = w_vec - self.es_ekf.b_ars_ins 

        # This implementation does not estimate the attitude rate, so we use the IMU ARS directly.
        self.latest_yaw_rate = w_imu[2, 0]       
           
        # Current time
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not hasattr(self, 'imu_last_stamp') or self.imu_last_stamp is None:
            self.imu_last_stamp = t
            return
        
        dt = t - self.imu_last_stamp
        self.imu_last_stamp = t

        # Rotation and transformation matrices from body to NED
        R = Rzyx(roll_imu, pitch_imu, yaw_imu)
        T = Tzyx(roll_imu, pitch_imu)

        # System dynamics to implement the 15-state error-state model
        # ∂x_dot = A(t) * ∂x + E(t) * w (Eq. 14.188 in Fossen 2nd ed.)
        # ∂y = C * ∂x + ε (Eq. 14.189 in Fossen 2nd ed.)
        A = self.es_ekf.generate_A(R, T)  # Eq. 14.192 in Fossen 2nd ed.
        E = self.es_ekf.generate_E(R, T)  # Eq. 14.193 in Fossen 2nd ed.

        # Discretization according to Fossen 2nd ed. Eq. 14.201
        Ad = np.eye(self.es_ekf.num_states) + A * dt 
        Ed = E * dt

        # Checking which aiding measurements we have

        O3 = np.zeros((3, 3))
        I3 = np.eye(3)
        zs, Cs, Rs = [], [], []
        # GNSS position
        if self.new_fix_measurement:
            p_meas = self.latest_fix[:3].reshape(3,1)
            p_ins  = self.es_ekf.p_hat_ins
            z_fix  = p_meas - p_ins # Calculate the position error                 
            C_fix = np.hstack([I3, O3, O3, O3, O3])

            zs.append( z_fix )
            Cs.append( C_fix )
            Rs.append( self.R_fix )

            self.new_fix_measurement = False

        # GNSS velocity
        if self.new_velocity_measurement:
            v_meas = self.latest_velocity[3:6].reshape(3,1) 
            v_ins  = self.es_ekf.v_hat_ins                   
            z_vel  = v_meas - v_ins # Calculate the velocity error             
            C_vel  = np.hstack([O3, I3, O3, O3, O3])     
            zs.append( z_vel )
            Cs.append( C_vel )
            Rs.append( self.R_vel )
            self.new_velocity_measurement = False

        # GNSS heading
        if self.new_heading_measurement:
            psi_meas = self.latest_heading[11] 
            psi_ins  = self.es_ekf.x_hat_ins[11,0]
            z_head   = np.array([[ psi_meas - psi_ins ]]) # Calculate the heading error
            C_head = np.zeros((1, 15))
            C_head[0, 11] = 1.0

            zs.append( z_head )
            Cs.append( C_head )
            Rs.append( self.R_head )

            self.new_heading_measurement = False
        
        # If we have any aiding measurements, we perform the correction step, if not we just propagate the state.
        if zs:
            z_total  = np.vstack(zs)    # combined measurement vector
            Cd_total  = np.vstack(Cs)    # combined measurement matrix, Cd = C
            R_noise  = scipy.linalg.block_diag(*Rs)  # combined measurement noise covariance matrix

            # Corrector: delta_x_hat[k] and P_hat[k]
            delta_x_hat, P_hat = self.es_ekf.correct(z_total, Cd_total, R_noise)

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
        g_n = (np.array([0.0, 0.0, gravity(self.latest_latitude)])).reshape(3, 1)  # gravity vector in navigation frame
        x_hat_ins = self.es_ekf.ins_propagation(
            self.es_ekf.x_hat_ins, dt, R, T, f_imu, w_imu, g_n=g_n,
        )
        # Publish the state estimate
        self._publish_state(x_hat_ins)


    # Callback functions for aiding measurements
    def update_fix(self, msg: NavSatFix):
        # Ensure the value we get is NOT NaN
        if np.isnan(msg.latitude) or np.isnan(msg.longitude):
            return

        if not self.initialized:
            # set ENU origin
            self.ref_lat, self.ref_lon = msg.latitude, msg.longitude
            self.latest_latitude = msg.latitude
            self.initialized = True
            self.get_logger().info("ES-EKF successfully initialized at first GNSS position")
            return

        # Convert to ENU measurement (east, north, up), ignore up
        self.latest_latitude = msg.latitude

        e, n, u = pm.geodetic2enu(
            msg.latitude, msg.longitude, 0.0,
            self.ref_lat, self.ref_lon, 0.0
        )

        # 2) configure EKF measurement model & noise
        cov = msg.position_covariance
        #print(f"GNSS covariance: {cov}")
        var_e = cov[0]  # latitude variance but because we're in ENU this maps to north; swap if needed
        var_n = cov[4]  # longitude variance mapping to east
        var_u = cov[8]  # up variance, not used here
        R_fix_msg = np.array([[var_e, 0.0, 0.0],
                              [0.0, var_n, 0.0],
                              [0.0, 0.0, var_u]])
        #self.ekf.R = self.R_fix # NOTE: Not using the static one as the GNSS has dynamic covariance matrix inbuilt in sensor itself
        self.R_fix = R_fix_msg
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

        print(f"Yaw before ssa: {yaw_gnss}")
        yaw_gnss = ssa(yaw_gnss)  # Force yaw to be in [-pi, pi)
        print(f"Yaw after ssa: {yaw_gnss}")

        z = np.zeros(15)
        z[11] = yaw_gnss 
        
        self.latest_heading = z
        self.new_heading_measurement = True

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

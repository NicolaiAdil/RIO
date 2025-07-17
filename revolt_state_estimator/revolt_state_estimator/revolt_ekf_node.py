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
from revolt_state_estimator.ekf import ExtendedKalmanFilter
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
        # EKF Initialization variables
        self.initialized = False           # don’t run EKF until first GNSS fix arrives
        self.u           = np.zeros(2)     # control input [effort, angle]
        self.dt          = 1.0/_pred_freq  # prediction timestep

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
        
        """
        State vector: 
        Vector with estimated states of the ReVolt ship, like position and velocity
         - x     (east)             (m)
         - y     (north)            (m)
         - yaw   (angle)            (rad)
         - v     (forward speed)    (m/s)
         - w_yaw (angular velocity) (rad/s)

        Measurement vector:
        Since different sensors give measurements at different rates we must chose measurement vector as the biggest sensor data vector
        In our case that is the IMU with 3 data points, thus dim_z = 3
        """
        self.ekf = ExtendedKalmanFilter(
            f=self.revolt_model.f,
            h=h_fix,
            dim_x=5,
            dim_z=3,
            dt=self.dt,
            Q=self.Q,
            R=self.R_fix,
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
        self.timer = self.create_timer(self.dt, self.predict)
        # State publisher
        self.state_pub = self.create_publisher(StateEstimate, '/state_estimate/revolt', 10)

        # Debugging ----------
        np.set_printoptions(
            linewidth=200, 
            precision=6, # adjust as you like
            suppress=True # so small floats don’t go to scientific notation
        )  
        self.get_logger().info(
            f"                                       \n" \
            f"ReVolt Model Params:                   \n" \
            f" m:            {_m}                    \n" \
            f" r:            {_dimensions[0]}        \n" \
            f" l:            {_dimensions[1]}        \n" \
            f" thruster_pos: {_thruster_placement}   \n" \
            f" v_lin_max:    {_velocity_linear_max}  \n" \
            f" v_ang_max:    {_velocity_angular_max} \n" \
            f"                                       \n" \
        )
        self.get_logger().info(
            f"                                   \n" \
            f"EKF Parameters:                    \n" \
            f" Q:                                \n" \
            f" {self.Q}                          \n" \
            f" R_fix:               {self.R_fix} \n" \
            f" R_head:              {self.R_head}\n" \
            f" R_imu:               {self.R_imu} \n" \
            f" Prediction interval: {self.dt}    \n" \
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



    # EKF callback functions (START) ==================================================
    def predict(self):
        if not self.initialized:
            return
        
        # 1) Run EKF predict step with current control u
        x_pred, P_pred = self.ekf.predict(self.u)

        # 2) Broadcast world → body TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'body'
        t.transform.translation.x = float(x_pred[0])
        t.transform.translation.y = float(x_pred[1])
        t.transform.translation.z = 0.0
        # yaw → quaternion
        q = tf_transformations.quaternion_from_euler(0, 0, float(x_pred[2]))
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.tf_broadcaster.sendTransform(t)

        # 3) Publish ekf/state as Odometry
        state = StateEstimate()
        state.header.stamp     = t.header.stamp
        state.header.frame_id  = 'world'
        state.child_frame_id   = 'body'

        # extract from your state vector x_pred = [x, y, yaw, v, w]
        state.x                = float(x_pred[0])  # world‐frame X
        state.y                = float(x_pred[1])  # world‐frame Y
        state.yaw              = float(x_pred[2])  # heading (rad)

        state.linear_velocity  = float(x_pred[3])  # forward speed (m/s)
        state.angular_velocity = float(x_pred[4])  # yaw rate   (rad/s)

        self.state_pub.publish(state)

        # Debugging ----------
        #self.get_logger().info(f"x_pred: {x_pred}")
        #self.get_logger().info(f"P_pred: {P_pred}")

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

        # 1) convert to ENU measurement (east, north, up), ignore up
        e, n, _ = pm.geodetic2enu(
            msg.latitude, msg.longitude, 0.0,
            self.ref_lat, self.ref_lon, 0.0
        )

        # 2) configure EKF measurement model & noise
        cov = msg.position_covariance
        var_e = cov[0]  # latitude variance but because we're in ENU this maps to north; swap if needed
        var_n = cov[4]  # longitude variance mapping to east
        R_fix_msg = np.array([[var_e, 0.0],
                          [0.0, var_n]])
        #self.ekf.R = self.R_fix # NOTE: Not using the static one as the GNSS has dynamic covariance matrix inbuilt in sensor itself
        self.ekf.R = R_fix_msg
        self.ekf.h = h_fix

        # 3) Perform the correction step
        z = np.array([e, n])
        x_post, P_post = self.ekf.update(z)

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

        # 1) Get GNSS yaw
        q = msg.quaternion
        _, _, raw_gnss = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # If first time we must find what the IMU start position is and add that as offset
        # This ensures the prediction yaw angle and the IMU yaw angle start at the same place
        if self.gnss_yaw_offset is None:
            self.gnss_yaw_offset = raw_gnss - self.ekf.x_post[2]
            self.gnss_yaw_last = raw_gnss
            self.get_logger().info("EKF Aligned GNSS heading")
            return

        gnss_yaw = unwrap(self.gnss_yaw_last, raw_gnss) # Unwrap so we don't get sharp edges
        self.gnss_yaw_last = gnss_yaw
        gnss_yaw_offset = self.imu_yaw_offset - gnss_yaw

        # 4) EKF‐correct with raw GNSS in world‐frame
        z = np.array([gnss_yaw_offset])
        self.ekf.h = h_head
        self.ekf.R = self.R_head
        x_post, P_post = self.ekf.update(z)

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

        # 1) extract GNSS‐reported velocity in world frame
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y

        # Transform vx and vy into body frame
        yaw = self.ekf.x_prior[2]
        v_body = vx*np.cos(yaw) + vy*np.sin(yaw)

        # 2) configure EKF measurement model & noise
        self.ekf.h = h_vel
        self.ekf.R = self.R_vel

        # 3) Perform the correction step
        z = np.array([v_body])
        x_post, P_post = self.ekf.update(z)

        # 4) Update IMU integrated velocity so it doesn't drift that much  
        self.v_integrated = x_post[3]

        # Debugging ----------
        #self.get_logger().info(f"Vel update → z=[{v_body:.3f}], x_post={x_post}")
        
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

        # 1) Get IMU yaw
        q = msg.orientation
        _, _, raw_imu = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # If first time we must find what the IMU start position is and add that as offset
        # This ensures the prediction yaw angle and the IMU yaw angle start at the same place
        if self.imu_yaw_offset is None:
            self.imu_yaw_offset = raw_imu - self.ekf.x_post[2]
            self.imu_yaw_last = raw_imu
            self.get_logger().info("EKF Aligned IMU heading")
            return

        imu_yaw = unwrap(self.imu_yaw_last, raw_imu) # Unwrap so we don't get sharp edges
        self.imu_yaw_last = imu_yaw
        imu_yaw_offset = self.imu_yaw_offset - imu_yaw

        # 2) Get IMU acceleration and integrate into velocity
        # Integrate acceleration to get velocity, a lot of noise so drift, GNSS reset the drift
        # however if GNSS fails, then we begin dead reconing
        # To avoid to drastic jumps limit realistic velocity of the imu
        # timestamp from message
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not hasattr(self, 'imu_last_stamp') or self.imu_last_stamp is None:
            self.imu_last_stamp = t
            self.v_integrated = 0.0
            return
        
        dt = t - self.imu_last_stamp
        self.imu_last_stamp = t
        self.v_integrated += msg.linear_acceleration.x * dt

        if abs(self.v_integrated) > self.imu_max_v:
            v_norm = self.v_integrated/abs(self.v_integrated)
            self.v_integrated = v_norm * self.imu_max_v

        # 3) Get IMU angular velocity
        w_imu = msg.angular_velocity.z

        # 4) EKF Correct
        z = np.array([imu_yaw_offset, self.v_integrated, w_imu])
        self.ekf.h = h_imu
        self.ekf.R = self.R_imu
        x_post, P_post = self.ekf.update(z)

        # Debugging ----------
        #self.get_logger().info(f"IMU update → z=[yaw:{imu_yaw_offset:.3f}, v_integrated: {self.v_integrated:.3f} w:{w_imu:.3f}], x_post={x_post}")
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

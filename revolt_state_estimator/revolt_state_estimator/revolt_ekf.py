#!/usr/bin/env python3
"""
revolt_ekf.py

ROS2 node that runs an Extended Kalman Filter for a simple ship model.
Fuses:
 - GNSS position (/fix)
 - GNSS heading (/heading)
 - GNSS velocity (/vel)
 - IMU linear acceleration & yaw rate (/imu/data)
 imu/dq	geometry_msgs/QuaternionStamped	integrated angular velocity from sensor (in quaternion representation)	1-400Hz(MTi-600 and MTi-100 series), 1-100Hz(MTi-1 series)
imu/dv	geometry_msgs/Vector3Stamped	
Publishes:
 - map → base_link TF
 - ekf/state Odometry (x, y, yaw, v, yaw_rate)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu, MagneticField
from std_msgs.msg import Float64
from geometry_msgs.msg import TransformStamped, Quaternion, QuaternionStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
import tf_transformations
import tf2_ros
import numpy as np
import pymap3d as pm

from revolt_state_estimator.revolt_model import ReVoltModel
from revolt_state_estimator.revolt_sensor_transforms import h_fix, h_head, h_vel, h_imu, h_dv
from revolt_state_estimator.ekf import ExtendedKalmanFilter
from revolt_state_estimator.utils import unwrap, HeadingAligner



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
        self.declare_parameter('revolt_ekf.R_vel',  [0.0, 0.0, 0.0]) # [v_x (m/s), v_y (m/s), w_yaw (rad/s)]
        self.declare_parameter('revolt_ekf.R_imu',  [0.0, 0.0]) # [yaw (rad), w_yaw (rad/s)]
        self.declare_parameter('revolt_ekf.R_dv',   [0.0, 0.0]) # [v_x (m/s), v_y (m/s)]
        self.declare_parameter('revolt_ekf.pred_freq', 0.0) # Hz
        _Q         = self.get_parameter('revolt_ekf.Q').value
        _R_fix     = self.get_parameter('revolt_ekf.R_fix').value
        _R_head    = self.get_parameter('revolt_ekf.R_head').value
        _R_vel     = self.get_parameter('revolt_ekf.R_vel').value
        _R_imu     = self.get_parameter('revolt_ekf.R_imu').value
        _R_dv      = self.get_parameter('revolt_ekf.R_dv').value
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

        # One-shot IMU↔GNSS yaw alignment
        self.heading_aligner = HeadingAligner()  # will compute constant yaw offset
        self.yaw_unwrap_head = 0.0               # last unwrapped GNSS yaw
        self.yaw_unwrap_imu  = 0.0               # last unwrapped IMU yaw
        self.last_imu_yaw    = None              # store most recent IMU yaw
        self.last_gnss_yaw   = None              # store most recent GNSS yaw

        # Noise models
        self.Q = np.diag(_Q)
        self.R_fix  = np.diag(_R_fix)
        self.R_head = np.diag(_R_head)
        self.R_vel  = np.diag(_R_vel)
        self.R_imu  = np.diag(_R_imu)
        self.R_dv   = np.diag(_R_dv)
        
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
        self.imu_sub = self.create_subscription(Imu,             '/imu/data', self.update_imu, 1)
        self.dv_sub  = self.create_subscription(Vector3Stamped,  '/imu/dv',   self.update_dv,  1) # IMU Has internal method to track velocity with NO drift from what data sheet says (sus stuff, might be marketing but who knows is good stuff either way jesjes)
        # TF broadcaster
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Timer for predict()
        self.timer = self.create_timer(self.dt, self.predict)
        # State publisher
        self.state_pub = self.create_publisher(Odometry, 'ekf/state', 10)

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

        # 2) Broadcast map → base_link TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(x_pred[0])
        t.transform.translation.y = float(x_pred[1])
        t.transform.translation.z = 0.0
        # yaw → quaternion
        q = tf_transformations.quaternion_from_euler(0, 0, float(x_pred[2]))
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self._tf_broadcaster.sendTransform(t)

        # 3) Publish ekf/state as Odometry
        odom = Odometry()
        odom.header.stamp = t.header.stamp
        odom.header.frame_id    = 'map'
        odom.child_frame_id     = 'base_link'
        odom.pose.pose.position.x    = float(x_pred[0])
        odom.pose.pose.position.y    = float(x_pred[1])
        odom.pose.pose.position.z    = 0.0
        odom.pose.pose.orientation   = t.transform.rotation

        # forward speed and yaw rate
        odom.twist.twist.linear.x   = float(x_pred[3])
        odom.twist.twist.linear.y   = 0.0
        odom.twist.twist.linear.z   = 0.0
        odom.twist.twist.angular.x  = 0.0
        odom.twist.twist.angular.y  = 0.0
        odom.twist.twist.angular.z  = float(x_pred[4])

        self.state_pub.publish(odom)

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
            self.get_logger().info("EKF initialized at first GNSS '/fix' position topic")
            self.get_logger().info("EKF now waiting for '/heading' and '/imu/data' topics for heading alignment")
            return

        # 1) convert to ENU measurement (east, north, up), ignore up
        e, n, _ = pm.geodetic2enu(
            msg.latitude, msg.longitude, 0.0,
            self.ref_lat, self.ref_lon, 0.0
        )

        # 2) configure EKF measurement model & noise
        self.ekf.h = h_fix
        self.ekf.R = self.R_fix

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

        # 1) unwrap GNSS yaw
        q = msg.quaternion
        _, _, raw_gnss = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        gnss_yaw = unwrap(self.yaw_unwrap_head, raw_gnss)
        self.yaw_unwrap_head = gnss_yaw
        self.last_gnss_yaw = gnss_yaw # Updating this, signals to heading realignment that we got GNSS heading, wait for IMU data

        # 2) try to calibrate if we also have IMU
        if ((self.last_imu_yaw is not None) and (not self.heading_aligner.is_calibrated())):
            self.heading_aligner.add_sample(self.last_imu_yaw, gnss_yaw)
            self.get_logger().info("EKF Aligned '/heading'")

        # 3) if not yet calibrated, bail
        if not self.heading_aligner.is_calibrated():
            return

        # 4) EKF‐correct with raw GNSS in world‐frame
        z = np.array([gnss_yaw])
        self.ekf.h = h_head
        self.ekf.R = self.R_head
        x_post, P_post = self.ekf.update(z)

        # Debugging ----------
        #self.get_logger().info(f"Heading update → z=[{gnss_yaw:.3f}], x_post={x_post}")

    def update_vel(self, msg: TwistStamped):
        # Ensure the value we get is NOT NaN
        if (
            np.isnan(msg.twist.linear.x) or 
            np.isnan(msg.twist.linear.y) or
            np.isnan(msg.twist.angular.z)
            ):
            return
        
        if not self.initialized:
            return

        # 1) extract GNSS‐reported velocity in world frame
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        wz = msg.twist.angular.z

        # 2) configure EKF measurement model & noise
        self.ekf.h = h_vel
        self.ekf.R = self.R_vel

        # 3) Perform the correction step
        z = np.array([vx, vy, wz])
        x_post, P_post = self.ekf.update(z)

        # Debugging ----------
        #self.get_logger().info(f"Vel update → z=[{vx:.3f}, {vy:.3f}, {wz:.3f}], x_post={x_post}")
        
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

        # 1) unwrap IMU yaw
        q = msg.orientation
        _, _, raw_imu = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        imu_yaw = unwrap(self.yaw_unwrap_imu, raw_imu)
        self.yaw_unwrap_imu = imu_yaw
        self.last_imu_yaw   = imu_yaw # Updating this, signals to heading realignment that we got IMU heading, wait for GNSS data

        # 2) calibrate if we’ve seen GNSS already
        if ((self.last_gnss_yaw is not None) and (not self.heading_aligner.is_calibrated())):
            self.heading_aligner.add_sample(imu_yaw, self.last_gnss_yaw)
            self.get_logger().info("EKF Aligned '/imu/data'")

        # 3) only correct once calibrated
        if not self.heading_aligner.is_calibrated():
            return

        # 4) align IMU into GNSS frame and EKF‐correct
        aligned_imu = self.heading_aligner.align_imu(imu_yaw)
        w_imu = msg.angular_velocity.z

        z = np.array([aligned_imu, w_imu])
        self.ekf.h = h_imu
        self.ekf.R = self.R_imu
        x_post, P_post = self.ekf.update(z)

        # Debugging ----------
        #self.get_logger().info(f"IMU update → z=[yaw:{aligned_imu:.3f}, w:{w_imu:.3f}], x_post={x_post}")

    def update_dv(self, msg: Vector3Stamped):
        dvx, dvy, dvz = msg.vector.x, msg.vector.y, msg.vector.z
        if np.isnan(dvx) or np.isnan(dvy) or np.isnan(dvz):
            return
        if not self.initialized:
            return

        self.ekf.h = h_dv
        self.ekf.R = self.R_dv

        z = np.array([dvx, dvy])
        x_post, P_post = self.ekf.update(z)

        # Debugging ----------
        self.get_logger().info(f"dv update → z=[{dvx:.3f},{dvy:.3f}], x_post={x_post}")
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

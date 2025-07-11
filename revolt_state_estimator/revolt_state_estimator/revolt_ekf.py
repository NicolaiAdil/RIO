#!/usr/bin/env python3
"""
revolt_ekf.py

ROS2 node that runs an Extended Kalman Filter for a simple ship model.
Fuses:
 - GNSS position (/fix)
 - GNSS heading (/heading)
 - IMU forward acceleration & yaw rate (/imu/data)
Publishes:
 - map → base_link TF
 - ekf/state Odometry (x, y, yaw, v, yaw_rate)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import tf_transformations
import tf2_ros
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import pymap3d as pm


class RevoltEKF(Node):
    def __init__(self):
        super().__init__('revolt_ekf')

        # -- EKF state and covariances ------------------------------------------
        # State vector: [x (east), y (north), yaw, v (forward speed)]
        self.ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
        self.ekf.x = np.zeros(4)               # Initialize at origin
        self.ekf.P = np.eye(4) * 1.0           # Initial state covariance
        # Process noise: adjust to match ship dynamics uncertainty
        self.ekf.Q = np.diag([0.1, 0.1, 0.01, 0.1])

        # Measurement noise covariances
        self.R_fix  = np.diag([5.0, 5.0])   # GNSS position noise (m^2)
        self.R_head = np.array([[0.05]])    # GNSS heading noise (rad^2)
        self.R_imu  = np.diag([0.2, 0.1])   # [accel noise, yaw-rate noise]

        # Prediction interval (seconds)
        self.dt = 0.1
        # Flag & origin for ENU conversion
        self.initialized = False
        self.ref_lat = self.ref_lon = 0.0

        # -- ROS interfaces ----------------------------------------------------
        # Subscribers for sensors
        self.fix_sub  = self.create_subscription(NavSatFix,  '/fix', self.fix_cb, 10)
        self.head_sub = self.create_subscription(Float64,    '/heading', self.heading_cb, 10)
        self.imu_sub  = self.create_subscription(Imu,        '/imu/data', self.imu_cb, 10)

        # Timer triggers predict() at fixed rate
        self.timer = self.create_timer(self.dt, self.predict)

        # TF broadcaster for map→base_link
        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        # Publisher for state as Odometry message
        self.state_pub = self.create_publisher(Odometry, 'ekf/state', 10)

    def predict(self):
        """Perform EKF predict step and broadcast TF + odometry."""
        if not self.initialized:
            return

        x, y, psi, v = self.ekf.x  # unpack state
        # State transition Jacobian F for constant-velocity motion
        F = np.eye(4)
        F[0,2] = -v * np.sin(psi) * self.dt
        F[0,3] =  np.cos(psi) * self.dt
        F[1,2] =  v * np.cos(psi) * self.dt
        F[1,3] =  np.sin(psi) * self.dt

        self.ekf.F = F
        self.ekf.predict()

        # -- Broadcast TF -----------------------------------------------------
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(self.ekf.x[0])
        t.transform.translation.y = float(self.ekf.x[1])
        # convert yaw to quaternion
        q = tf_transformations.quaternion_from_euler(0, 0, float(self.ekf.x[2]))
        (t.transform.rotation.x,
         t.transform.rotation.y,
         t.transform.rotation.z,
         t.transform.rotation.w) = q
        self.broadcaster.sendTransform(t)

        # -- Publish Odometry ------------------------------------------------
        odom = Odometry()
        odom.header.stamp = t.header.stamp
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(self.ekf.x[0])
        odom.pose.pose.position.y = float(self.ekf.x[1])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        # forward speed and yaw rate
        odom.twist.twist.linear.x  = float(self.ekf.x[3])
        # yaw-rate stored from last IMU callback
        odom.twist.twist.angular.z = getattr(self, 'last_yaw_rate', 0.0)
        self.state_pub.publish(odom)

    def fix_cb(self, msg: NavSatFix):
        """GNSS position update. Initialize or fuse east/north."""
        if not self.initialized:
            # set ENU origin
            self.ref_lat, self.ref_lon = msg.latitude, msg.longitude
            self.initialized = True
            self.get_logger().info('EKF initialized at first GNSS fix')
            return

        # Convert geodetic → ENU relative to origin
        e, n, _ = pm.geodetic2enu(
            msg.latitude, msg.longitude, msg.altitude,
            self.ref_lat, self.ref_lon, 0.0
        )
        z = np.array([e, n])
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        # update() needs functions for Jacobian and measurement model
        self.ekf.update(
            z,
            HJacobian=lambda x: H,
            Hx=lambda x: x[:2],
            R=self.R_fix
        )

    def heading_cb(self, msg: Float64):
        """GNSS heading update. Fuse yaw angle."""
        if not self.initialized:
            return
        z = np.array([msg.data])
        H = np.array([[0, 0, 1, 0]])
        self.ekf.update(
            z,
            HJacobian=lambda x: H,
            Hx=lambda x: np.array([x[2]]),
            R=self.R_head
        )

    def imu_cb(self, msg: Imu):
        """IMU update. Fuse forward accel and yaw rate."""
        if not self.initialized:
            return
        # store yaw rate for odometry
        self.last_yaw_rate = msg.angular_velocity.z

        # project accel into ship frame
        psi = self.ekf.x[2]
        ax, ay = msg.linear_acceleration.x, msg.linear_acceleration.y
        a_fwd = ax * np.cos(psi) + ay * np.sin(psi)
        z = np.array([a_fwd, self.last_yaw_rate])

        # only yaw rate maps directly (accel is indirect)
        H = np.zeros((2, 4))
        H[1,2] = 1.0 / self.dt
        def hx(x): return np.array([0.0, x[2] / self.dt])

        self.ekf.update(
            z,
            HJacobian=lambda x: H,
            Hx=hx,
            R=self.R_imu
        )


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

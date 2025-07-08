# Revolt State Estimator

launch using
ros2 launch revolt_state_estimator revolt_state_estimator.launch.py

you can see topics:
ros2 run tf2_ros tf2_echo odom base_link
ros2 topic echo /odometry/filtered

Enable filter if neede ususally its alreday enabled tho by defaukt, sanity check jesjes:
ros2 service call /enable std_srvs/srv/Empty "{}"

ros2 service call /reset std_srvs/srv/Empty "{}"


/odometry/filtered IS THE MOST importnat topic that MUST be used for controls and navigation further

We now see what we have The TF and the topics
The TF must be braodcasted in a good time before EKF can start, thsi way TF has time to propagate through teh ssytem before EKF can use it 


ALso shoudl exlain what internal_sensors_static_tf.yaml is and what it does

Also the utm zone 32 stuff fr Norway very important

martynas@MARTYNAS-PC:~/Desktop/MartynasLinux/DNV/GUI_ws$ 


ALl the TF is launched through launch file:
map
 └ odom
     └ base_link
         ├ imu_link
         └ gps_link
Seperate config file for this revolt_tf.yaml that describes how ship is related to world at first, how the sensors are related to the ship and ect. We use TF beacuse it simplifies Kinematics equations and is handled by the TF2 toolbox.


Make the workspace
my_ws/src/SensorFusion and Harwdare also (optionally if you intend to run full harwdware) Also have RevoltMSGS

## Prerequisites

Before building, install the **robot_localization** package for your ROS 2 distribution (for example, Jazzy):
```bash
sudo apt update
sudo apt install ros-<ros2-distro>-robot-localization
```
for us its 
sudo apt install ros-jazzy-robot-localization

Then ensure your GNSS and IMU drivers are installed, the esiest way to ensure this is just to go to Hardware repo, then go to xsens and nmea_navsat ROS2 packages under teh repo, read their respective README.md files and set them up acordingly.


# TOPICS IN
/fix              [sensor_msgs/NavSatFix]
/vel              [geometry_msgs/TwistStamped]
/heading          [geometry_msgs/QuaternionStamped]
imu/data	        [sensor_msgs/Imu]	               quaternion, calibrated angular velocity and acceleration	1-400Hz(MTi-600 and MTi-100 series), 1-100Hz(MTi-1 series)




NOTE: SOme topics are remaped from hardware because of robot_localization package that has strict topic requirements to be:
write them down 

Subscribed Topics NOTE need to remap my /fix, /heading, /vel and /imu/data
imu/data A sensor_msgs/Imu message with orientation data
odometry/filtered A nav_msgs/Odometry message of your robot’s current position. This is needed in the event that your first GPS reading comes after your robot has attained some non-zero pose.
gps/fix A sensor_msgs/NavSatFix message containing your robot’s GPS coordinates
Published Topics
odometry/gps A nav_msgs/Odometry message containing the GPS coordinates of your robot, transformed into its world coordinate frame. This message can be directly fused into robot_localization’s state estimation nodes.
gps/filtered (optional) A sensor_msgs/NavSatFix message containing your robot’s world frame position, transformed into GPS coordinates
Published Transforms
world_frame->utm (optional) - If the broadcast_utm_transform parameter is set to true, navsat_transform_node calculates a transform from the utm frame to the frame_id of the input odometry data. By default, the utm frame is published as a child of the odometry frame by using the inverse transform. With use of the broadcast_utm_transform_as_parent_frame parameter, the utm frame will be published as a parent of the odometry frame. This is useful if you have multiple robots within one TF tree.

Published Topics
odometry/filtered (nav_msgs/Odometry)
accel/filtered (geometry_msgs/AccelWithCovarianceStamped) (if enabled)
Published Transforms
If the user’s world_frame parameter is set to the value of odom_frame, a transform is published from the frame given by the odom_frame parameter to the frame given by the base_link_frame parameter.
If the user’s world_frame parameter is set to the value of map_frame, a transform is published from the frame given by the map_frame parameter to the frame given by the odom_frame parameter.
Note This mode assumes that another node is broadcasting the transform from the frame given by the odom_frame parameter to the frame given by the base_link_frame parameter. This can be another instance of a robot_localization state estimation node.
Services
set_pose - By issuing a geometry_msgs/PoseWithCovarianceStamped message to the set_pose topic, users can manually set the state of the filter. This is useful for resetting the filter during testing, and allows for interaction with rviz. Alternatively, the state estimation nodes advertise a SetPose service, whose type is robot_localization/SetPose.




**Purpose**

The Revolt State Estimator package fuses GNSS (GPS) and IMU data into a smooth, drift‑compensated pose and velocity estimate. It relies entirely on the pre‑built ROS 2 nodes provided by the **robot_localization** toolbox—**navsat_transform_node** and **ekf_node**—so you never write filter code yourself. Instead, you configure behavior via simple YAML files and a minimal launch script.

---

## Overview of How It Works

1. **navsat_transform_node**
   - Listens to:
     - `/fix` (sensor_msgs/NavSatFix)
     - (optionally) `/vel` (geometry_msgs/TwistStamped)
     - (optionally) `/heading` (geometry_msgs/QuaternionStamped)
   - On first valid fix it picks that latitude/longitude as the local origin (datum).
   - Converts every incoming fix into X/Y meters relative to the origin.
   - Optionally incorporates GNSS velocity and compass heading into the output.
   - Publishes `nav_msgs/Odometry` on `/odometry/gps`, with pose in the `map` frame and child frame `odom`.

2. **ekf_node**
   - Listens to:
     - `/imu/data` (sensor_msgs/Imu) for orientation, angular velocity, and linear acceleration
     - `/odometry/gps` (nav_msgs/Odometry) produced by the transform node
     - (optionally) `/vel` for GNSS‑reported velocity
     - (optionally) `/heading` for GNSS‑provided absolute yaw
   - Uses a 15‑state Extended Kalman Filter (from robot_localization) to fuse all inputs:
     - **Predict** step on IMU messages (integrates gyro + accel)
     - **Update** step on GNSS‑derived odometry, velocity, and heading messages
   - Publishes fused `nav_msgs/Odometry` on `/odometry/filtered` and broadcasts the TF transform `odom → base_link` at a fixed rate.

---

## Why We Use robot_localization

- **No custom filter code**: Both `navsat_transform_node` and `ekf_node` are fully implemented in C++ in the robot_localization package.
- **YAML‑driven**: You control topics, frames, data masks, and covariances entirely via YAML parameter files. No recompilation needed to adjust tuning.
- **Battle‑tested**: robot_localization is widely used across ROS projects for mobile robots, drones, and marine vessels.

---

## Configuration Files

### `config/navsat.yaml`
Contains parameters for `navsat_transform_node`:

```yaml
navsat_transform_node:
  ros__parameters:
    frequency: 5.0            # How often to publish /odometry/gps (Hz)
    delay: 0.0                # Seconds before picking GNSS origin
    wait_for_datum: false     # Don’t block if no fix yet

    world_frame: map
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link

    gps0: /fix                # Raw NavSatFix topic
    gps0_config: [ true, true, false, false, false, false, false, false, false, false, false, false, false, false, false ]

    vel0: /vel                # GNSS velocity (optional)
    vel0_config: [ false, false, false, false, false, false, true, true, false, false, false, false, false, false, false ]

    head0: /heading           # GNSS heading (optional)
    head0_config: [ false, false, false, false, false, true, false, false, false, false, false, false, false, false, false ]
```

### `config/ekf.yaml`
Contains parameters for `ekf_node`:

```yaml
ekf_filter_node:
  ros__parameters:
    frequency: 30.0           # Filter output rate in Hz
    two_d_mode: true          # Planar operation: ignore Z/roll/pitch
    publish_tf: true          # Broadcast odom→base_link TF

    world_frame: odom
    odom_frame: odom
    base_link_frame: base_link

    imu0: /imu/data           # IMU topic
    imu0_config: [ false, false, false, false, false, false, false, false, false, true, true, true, true, true, true ]
    imu0_remove_gravitational_acceleration: true
    gravitational_acceleration: 9.80665

    pose0: /odometry/gps      # GNSS position + yaw from transform
    pose0_config: [ true, true, false, false, false, true, false, false, false, false, false, false, false, false, false ]
    pose0_differential: false

    odom1: /vel               # GNSS velocity (optional)
    odom1_config: [ false, false, false, false, false, false, true, true, false, false, false, false, false, false, false ]

    pose1: /heading           # GNSS heading (optional)
    pose1_config: [ false, false, false, false, false, true, false, false, false, false, false, false, false, false, false ]
```

> **_config** masks are 15‑element arrays selecting which state variables to fuse: [x, y, z, roll, pitch, yaw, vx, vy, vz, v_roll, v_pitch, v_yaw, ax, ay, az].

---

## Launch File

Place the following in `launch/state_estimator.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('revolt_state_estimator')
    gnss_cfg = PathJoinSubstitution([pkg, 'config', 'navsat.yaml'])
    ekf_cfg  = PathJoinSubstitution([pkg, 'config', 'ekf.yaml'])

    return LaunchDescription([
        Node(
            package='robot_localization', executable='navsat_transform_node',
            name='gnss_transform_node', output='screen', parameters=[gnss_cfg]
        ),
        Node(
            package='robot_localization', executable='ekf_node',
            name='ekf_filter_node', output='screen', parameters=[ekf_cfg]
        ),
    ])
```

Launch with:
```bash
ros2 launch revolt_state_estimator state_estimator.launch.py
```

---

## Running and Outputs

- **`/odometry/gps`**: intermediate odometry from GNSS fix
- **`/odometry/filtered`**: fused pose+twist output by EKF
- **TF**: `odom → base_link` broadcast at filter rate

View in RViz or use `ros2 topic echo` and `ros2 run tf2_ros tf2_echo odom base_link`.

---

## Summary

- **No custom filter coding**—all EKF logic is in robot_localization’s prebuilt nodes.
- **YAML-only configuration**—topics, frames, and data masks define behavior.
- **Two-node pipeline**: navsat_transform_node → ekf_node.
- **Outputs**: standard ROS Odometry and TF for seamless integration.

By following this setup, you get a robust state estimator in minutes, with full flexibility via YAML tuning and no filter code to maintain.

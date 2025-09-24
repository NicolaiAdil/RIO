# ReVolt State Estimator
A ROS 2 package providing an Error State Extended Kalman Filter (ES-EKF) for estimating a ship’s state (Pose and velocity) by fusing GNSS and IMU data.

**NOTE**: The current estimate is from NED -> IMU frame since tf2 is not working properly. In theory it is a simple static transformation, but tf2 is struggling to find these transforms so its very slowly. However with the current configuration of the IMU, the imu frame and body frame are the same so no transformation is needed. 

tldr; if the imu changes position or orientation, the tf2 transform will need to be implemented, or a hardcoded transform.

---

## Overview
The **ReVolt State Estimator** fuses:
- **GNSS** (`/fix`, `/heading`, `/vel`)
- **IMU** (`/imu/data`)

It outputs:
- **TF**: `NED → Body`
- **Odometry**: `nav_msgs/Odometry` on topic `/state_estimate/revolt`

## Usage
Nodes will declare parameters from `config/revolt_ekf.yaml` on startup.

## Nodes & Launch
### Node: `revolt_ekf_node`
- **Executable**: `revolt_ekf_node`
- **Package**: `revolt_state_estimator`
- **Launch**: `revolt_state_estimator.launch.py` loads ES-EKF config and starts the node.

## Topics & Messages
### Subscribed
| Topic        | Type                     | Description                         |
|--------------|--------------------------|-------------------------------------|
| `/fix`       | `sensor_msgs/NavSatFix`  | GNSS position (lat, lon + covariance)|
| `/heading`   | `geometry_msgs/QuaternionStamped` | GNSS heading (as quaternion) |
| `/vel`       | `geometry_msgs/TwistStamped`      | GNSS linear velocity (world frame) |
| `/imu/data`  | `sensor_msgs/Imu`        | IMU orientation, accel & gyro       |

### Published
| Topic                         | Type                             | Description                  |
|-------------------------------|----------------------------------|------------------------------|
| `/state_estimate/revolt`      | `nav_msgs/Odometry`              | Estimated [p, v, Theta]      |
| `NED → body` (TF)             | `geometry_msgs/TransformStamped` | Pose transform               |

## Parameters & Configuration
Configurations live in the `config/` folder:
- **revolt_ekf.yaml**: Noise covariances and topic names

### EKF Parameters (`revolt_ekf`)
| Name            | Type   | Units                                            | Description                                         |
|-----------------|--------|--------------------------------------------------|-----------------------------------------------------|
| `Q`             | [float]| process noise diag ([p, v, b_acc, Theta, b_ars]) | Covariance of ins model                             |
| `R_head`        | [float]| rad²                                             | GNSS heading noise                                  |
| `R_vel`         | [float]| [(m/s)², (m/s)², (m/s)²]                                           | GNSS velocity noise                                 |

The covariance of the fix is automatically taken from the /fix topic.

## State & Measurement Vectors
- **State** vector `x = [p, v, b_acc, Theta, b_ars]`:
  - `p = [x, y, z]` – position in NED (m)
  - `v = [v_x, v_y, v_z]` – velocity in NED (m/s)
  - `b_acc` – bias in accelerometer (m/s²)
  - `Theta = [phi, theta, psi]` – attitude in (rad)
  - `b_ars` – bias in angular rate sensor (rad/s)

- **Measurement**:
  - **GNSS fix**: `[x, y, z]` position in longitude, latitude, altitude (m)
  - **GNSS heading**: `[yaw]` according to north (rad)
  - **GNSS vel**: `[v_x, v_y, v_z]` velocity in ENU (m/s)

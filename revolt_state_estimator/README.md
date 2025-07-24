# ReVolt State Estimator
A ROS 2 package providing an Error State Extended Kalman Filter (ES-EKF) for estimating a ship’s state (Pose and velocity) by fusing GNSS and IMU data.

---

## Overview
The **ReVolt State Estimator** fuses:
- **GNSS** (`/fix`, `/heading`, `/vel`)
- **IMU** (`/imu/data`)

It outputs:
- **TF**: `NED → Body`
- **Odometry**: `nav_msgs/Odometry` on topic `/state_estimate/revolt`

## Features
- Dynamic measurement noise from GNSS covariance
- Integrated IMU dead‐reckoning with GNSS reset
- Configurable via YAML (model & EKF)
- Launch file for easy startup

## Usage
Launch the estimator:
```bash
ros2 launch revolt_state_estimator revolt_state_estimator.launch.py
```
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
| `R_fix`         | [float]| [m², m²]                                         | GNSS fix measurement noise (covariance matrix diag) |
| `R_head`        | [float]| rad²                                             | GNSS heading noise                                  |
| `R_vel`         | [float]| (m/s)²                                           | GNSS velocity noise                                 |

## State & Measurement Vectors
- **State** vector `x = [x, y, yaw, v, w_yaw]`:
  - `x, y` – position in East/North (m)
  - `yaw` – heading angle (rad)
  - `v` – forward speed (m/s)
  - `w_yaw` – yaw rate (rad/s)

- **Measurement** vector `z` varies per sensor:
  - **GNSS fix**: `[x, y]` (m)
  - **GNSS heading**: `[yaw]` (rad)
  - **GNSS vel**: `[v]` (m/s)
  - **IMU**: `[yaw, v, w_yaw]` (rad, m/s, rad/s)

Function-to-measurement mappings in `revolt_sensor_transforms.py` fileciteturn0file4.

## Kalman Filter Theory
An Extended Kalman Filter operates in two steps:
1. **Predict**
   - Propagate state via continuous ship model `f(x,u)` using Euler integration.
   - Linearize dynamics: Jacobian `A = ∂f/∂x`, `B = ∂f/∂u` (numerical).  
   - Discretize via Zero-Order Hold: `(A_d, B_d) = expm([[A,B];[0,0]]·dt)`.
   - Update covariance: `P₋ = A_d P₊ A_dᵀ + Q`.

2. **Update**
   - Choose measurement model `h(x)` (position, heading, vel, IMU).  
   - Linearize: `H = ∂h/∂x` (numerical).  
   - Compute innovation `y = z - h(x₋)`, covariance `S = H P₋ Hᵀ + R`.  
   - Kalman gain: `K = P₋ Hᵀ S⁻¹`.  
   - State update: `x₊ = x₋ + K y`, `P₊ = (I − K H) P₋`.

Tuning **Q**, **R** matrices balances trust in model vs. measurements. High **Q** → trust measurements. High **R** → trust prediction.

## Tuning & Analysis Tools
Under `state_estimate_tuning/`, non‑ROS scripts help select `Q,R`:
- **data/**: recorded logs of GNSS & IMU
- **plot_fix.py**: plot position residuals vs. truth
- **plot_head.py**: compare heading estimates
- **plot_vel.py**: velocity error analysis
- **plot_imu.py**: IMU compensation diagnostics
- **plot_estimates.py**: overlay all state estimates
- **plots/**: generated figures

Use these to visualize filter performance, adjust covariance entries, and re‑run estimator until residuals are white and zero‑mean.

## Launch File
`launch/revolt_state_estimator.launch.py`:
- Finds package share, loads `revolt_ekf.yaml` & `revolt_model.yaml`.
- Spawns `revolt_ekf_node` with `output=screen`.

Minimal snippet:
```python
Node(
  package='revolt_state_estimator',
  executable='revolt_ekf_node',
  parameters=[config_ekf, config_model],
)
```
More in `launch/revolt_state_estimator.launch.py`

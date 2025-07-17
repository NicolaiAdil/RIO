# ReVolt State Estimator
A ROS 2 package providing an Extended Kalman Filter (EKF) for estimating a ship’s state (position, heading, velocity) by fusing GNSS and IMU data.

---

## Overview
The **ReVolt State Estimator** fuses:
- **GNSS** (`/fix`, `/heading`, `/vel`)
- **IMU** (`/imu/data`)
- **Thruster commands** (`/tau_m`, `/tau_delta`)

It outputs:
- **TF**: `world → body`
- **Odometry**: `custom_msgs/StateEstimate` on `/state_estimate/revolt`

## Features
- Continuous‐time ship model (surge & yaw dynamics)
- EKF with:
  - Euler forward prediction
  - Zero-order hold discretization of linearized dynamics
  - Numerical Jacobians
- Dynamic measurement noise from GNSS covariance
- Integrated IMU dead‐reckoning with GNSS reset
- Configurable via YAML (model & EKF)
- Launch file for easy startup

## Usage
Launch the estimator:
```bash
ros2 launch revolt_state_estimator revolt_state_estimator.launch.py
```
Nodes will declare parameters from `config/revolt_ekf.yaml` and `config/revolt_model.yaml` on startup.

## Nodes & Launch
### Node: `revolt_ekf_node`
- **Executable**: `revolt_ekf_node`
- **Package**: `revolt_state_estimator`
- **Launch**: `revolt_state_estimator.launch.py` loads model & EKF configs and starts the node fileciteturn0file0.

## Topics & Messages
### Subscribed
| Topic        | Type                     | Description                         |
|--------------|--------------------------|-------------------------------------|
| `/tau_m`     | `std_msgs/Float64`       | Thruster effort command             |
| `/tau_delta` | `std_msgs/Float64`       | Thruster angle command (rad)        |
| `/fix`       | `sensor_msgs/NavSatFix`  | GNSS position (lat, lon + covariance)|
| `/heading`   | `geometry_msgs/QuaternionStamped` | GNSS heading (as quaternion) |
| `/vel`       | `geometry_msgs/TwistStamped`      | GNSS linear velocity (world frame) |
| `/imu/data`  | `sensor_msgs/Imu`        | IMU orientation, accel & gyro       |

### Published
| Topic                         | Type                         | Description                  |
|-------------------------------|------------------------------|------------------------------|
| `/state_estimate/revolt`      | `custom_msgs/StateEstimate`  | Estimated [x, y, yaw, v, w]  |
| `world → body` (TF)           | `geometry_msgs/TransformStamped` | Pose transform         |

## Parameters & Configuration
Configurations live in the `config/` folder:
- **revolt_model.yaml**: ship physical & hydrodynamic parameters
- **revolt_ekf.yaml**: EKF noise covariances & prediction rate

### Model Parameters (`revolt_model`)
| Name                          | Type     | Units    | Description                            |
|-------------------------------|----------|----------|----------------------------------------|
| `m`                           | float    | kg       | Ship mass                              |
| `dimensions`                  | [float]  | [m,m]    | [half‑width (r), length (l)]           |
| `thruster_placement`         | [float]  | [m,m]    | Thruster position in body frame (x,y)  |
| `velocity_linear_max`         | float    | m/s      | Max surge speed (for clipping)         |
| `velocity_angular_max`        | float    | rad/s    | Max yaw rate (for clipping)            |

### EKF Parameters (`revolt_ekf`)
| Name            | Type   | Units                   | Description                                              |
|-----------------|--------|-------------------------|----------------------------------------------------------|
| `Q`             | [float]| process noise diag ([x, y, yaw, v, w]) | Covariance of process noise         |
| `R_fix`         | [float]| [m², m²]               | GNSS fix measurement noise (covariance matrix diag)      |
| `R_head`        | [float]| rad²                   | GNSS heading noise                                        |
| `R_vel`         | [float]| (m/s)²                | GNSS velocity noise                                       |
| `R_imu`         | [float]| [rad², (m/s)², (rad/s)²] | IMU measurement noise (yaw, v, w)                       |
| `pred_freq`     | float  | Hz                     | Prediction loop frequency                                  |

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

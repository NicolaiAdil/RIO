# ReVolt SensorFusion Stack

A unified ROS 2 workspace for advanced perception and estimation on the ReVolt vessel. Initially focused on fusing GNSS, IMU, and thruster data into a robust state estimate, this stack will grow to include modules for target tracking, SLAM, computer vision, and more.

---

## Overview
The SensorFusion workspace integrates custom messages, launch scripts, and algorithm implementations to build a modular pipeline: from low‑level sensor inputs through mid‑level state estimation to high‑level autonomy. It currently provides an EKF‑based state estimator producing vessel pose (`x, y, yaw`) and motion (`v, w`) along with the corresponding TF transform. Future packages will slot in seamlessly for mapping, object detection, and navigation.

---

## Workspace Setup

To prepare your ROS 2 environment:

```bash
# 1. Create workspace
mkdir -p ~/my_ws/src
cd ~/my_ws/src

# 2. Clone dependencies and SensorFusion
git clone <hardware-repo-url> Hardware     # low-level drivers
git clone <revolt-messages-repo-url> RevoltMsgs
git clone <sensorfusion-repo-url> SensorFusion

# 3. Build selected packages
cd ~/my_ws
colcon build --packages-select hardware_launch revolt_msgs revolt_state_estimator sensor_fusion_launch

# 4. Source setup
source install/setup.bash
```

After this, all nodes, messages, and launches are available.

---

## Launching SensorFusion

First bring up hardware drivers, then start the estimation stack:

```bash
ros2 launch hardware_launch hardware.launch.py # thrusters, IMU, GNSS, etc.
ros2 launch sensor_fusion_launch sensor_fusion.launch.py  # EKF state estimator
```

Individual packages can also be launched on demand by specifying `<package> <launch_file>`.

---

## Packages

### sensor_fusion_launch
Main launch package that loads configuration files (`revolt_model.yaml`, `revolt_ekf.yaml`) and spawns the EKF node with required transforms.

```bash
ros2 launch sensor_fusion_launch sensor_fusion.launch.py
```

### revolt_state_estimator

This package hosts the EKF node that fuses incoming sensor and control signals into a unified state estimate:

- **Subscriptions**:

  - `/fix` (`NavSatFix`) for GNSS position
  - `/heading` (`QuaternionStamped`) for GNSS heading
  - `/vel` (`TwistStamped`) for GNSS velocity
  - `/imu/data` (`Imu`) for orientation, acceleration, and angular rate
  - `/tau_m` & `/tau_delta` (`Float64`) for thruster commands (these topics become available when you launch the Control Systems repository alongside SensorFusion)

- **Publications**:

  - `/state_estimate/revolt` (`custom_msgs/StateEstimate`) carrying `[x, y, yaw, v, w]`
  - TF transform `world → body`

The thruster command topics allow the controller to send real‑time control signals to both the hardware drivers and the state estimator. This dual feed means the estimator knows what commands were applied, improving prediction accuracy. In turn, the hardware drivers move the vessel toward the desired reference, and the estimator reports the actual vessel state back to the controller—resulting in a more accurate, robust, stable, and smooth control loop.

Within this package, a `state_estimate_tuning/` directory holds Python scripts for analyzing filter residuals and adjusting noise covariances—a convenient way to refine EKF performance against logged data.

---

Refer to individual package READMEs for detailed parameter descriptions and theoretical background. Happy fusing!

# ReVolt SensorFusion Stack

A unified ROS 2 workspace for advanced perception and estimation on the ReVolt vessel. Initially focused on fusing GNSS, and IMU into a robust state estimate, this stack will grow to include modules for target tracking, SLAM, computer vision, and more.

---

## Overview
The SensorFusion workspace integrates custom messages, launch scripts, and algorithm implementations to build a modular pipeline: from low‑level sensor inputs through mid‑level state estimation to high‑level autonomy. It currently provides an ES-EKF‑based state estimator producing vessel pose and velocity in NED. Future packages will slot in seamlessly for mapping, object detection, and navigation.

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

---

## Packages

### revolt_state_estimator

This package hosts the ES-EKF node that fuses incoming sensor and control signals into a unified state estimate

---

Refer to individual package READMEs for detailed parameter descriptions and theoretical background. Happy fusing!

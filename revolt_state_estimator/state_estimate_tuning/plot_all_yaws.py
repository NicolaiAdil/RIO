import pandas as pd
import matplotlib.pyplot as plt
import tf_transformations
import os
import numpy as np

def extract_yaws(qx, qy, qz, qw):
    return [tf_transformations.euler_from_quaternion([x, y, z, w])[2]
            for x, y, z, w in zip(qx, qy, qz, qw)]

# === Load & parse heading ===
heading = pd.read_csv("data/heading_data_fixed.csv", header=None)
heading.columns = ["sec", "nanosec", "frame_id", "qx", "qy", "qz", "qw"]
heading["time"] = heading["sec"] + heading["nanosec"] * 1e-9
heading["yaw_rad"] = extract_yaws(heading["qx"], heading["qy"], heading["qz"], heading["qw"])
heading["yaw_deg"] = np.degrees(heading["yaw_rad"])

# === Load & parse imu ===
imu = pd.read_csv("data/imu_data_fixed.csv", header=None)
imu = imu.iloc[:, :7]
imu.columns = ["sec", "nanosec", "frame_id", "qx", "qy", "qz", "qw"]
imu["time"] = imu["sec"] + imu["nanosec"] * 1e-9
imu["yaw_rad"] = extract_yaws(imu["qx"], imu["qy"], imu["qz"], imu["qw"])
imu["yaw_deg"] = np.degrees(imu["yaw_rad"])

# === Load & parse EKF ===
ekf = pd.read_csv("data/state_estimate_fixed.csv", header=None)
ekf = ekf.iloc[:, :10]
ekf.columns = [
    "sec", "nanosec", "frame_id",
    "px", "py", "pz", "qx", "qy", "qz", "qw"
]
ekf["time"] = ekf["sec"] + ekf["nanosec"] * 1e-9
ekf["yaw_rad"] = extract_yaws(ekf["qx"], ekf["qy"], ekf["qz"], ekf["qw"])
ekf["yaw_deg"] = np.degrees(ekf["yaw_rad"])

# === Plot ===
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(10, 5))


# EKF: regular line
plt.plot(ekf["time"], ekf["yaw_deg"], label="EKF Estimate", linewidth=1.2)

# IMU: thicker line
# plt.plot(imu["time"], imu["yaw_deg"], label="IMU Yaw", linewidth=2.5)

# GNSS: transparent dots
# plt.scatter(heading["time"], heading["yaw_deg"], label="GNSS Heading", color="black", s=10, alpha=0.1, zorder=3)

plt.xlabel("Time [s]")
plt.ylabel("Yaw [°]")
plt.title("Yaw over Time (Degrees) from IMU, GNSS, and EKF")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("plots/yaw_plot_deg.png", dpi=300)
# plt.savefig("plots/yaw_plot_deg.jpg", dpi=300)

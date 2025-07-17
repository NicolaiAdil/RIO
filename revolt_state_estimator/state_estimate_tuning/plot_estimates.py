import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV (no headers)
df = pd.read_csv('data/ekf_state_estimate_revolt.csv', header=None)

# Columns according to StateEstimate.msg:
# 0: sec, 1: nsec, 2: frame_id, 3: child_frame_id,
# 4: x, 5: y, 6: yaw, 7: linear_velocity, 8: angular_velocity

# 1) Time (s), normalized
time = df[0] + df[1]*1e-9
time -= time.iloc[0]

# 2) Pose & orientation
x   = df[4]
y   = df[5]
yaw = df[6]  # already in radians

# 3) Velocities
speed     = df[7] # forward speed
ang_vel_z = df[8] # yaw rate

# 4) Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 2D Trajectory
axes[0,0].plot(x, y)
axes[0,0].set_aspect('equal', 'box')
axes[0,0].set_xlabel('X (m)')
axes[0,0].set_ylabel('Y (m)')
axes[0,0].set_title('2D Trajectory')

# Yaw vs Time
axes[0,1].plot(time, yaw)
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Yaw (rad)')
axes[0,1].set_title('Yaw vs Time')

# Speed vs Time
axes[1,0].plot(time, speed)
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Speed (m/s)')
axes[1,0].set_title('Speed vs Time')

# Angular Velocity vs Time
axes[1,1].plot(time, ang_vel_z)
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Yaw Rate (rad/s)')
axes[1,1].set_title('Angular Velocity vs Time')

plt.tight_layout()
plt.show()

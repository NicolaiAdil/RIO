import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ——— CONFIG ———
csv_path = "data/imu.csv"  # adjust to your imu.csv path
gt_yaw   = -1.2    # [rad] ground-truth yaw offset
gt_wyaw  =  0.0    # [rad/s] ground-truth yaw rate
gt_ay    =  0.0    # [m/s²] ground-truth lateral acceleration

np.set_printoptions(suppress=True, precision=10)

# ——— LOAD & PREPROCESS ———
cols = [
    'sec','nsec','frame_id',
    'qx','qy','qz','qw',        # orientation quaternion
    'cov_o0','cov_o1','cov_o2', # orientation covariance (unused)
    'cov_o3','cov_o4','cov_o5',
    'cov_o6','cov_o7','cov_o8',
    'wx','wy','wz',             # angular velocity
    'cov_w0','cov_w1','cov_w2', # angular covariance (unused)
    'cov_w3','cov_w4','cov_w5',
    'cov_w6','cov_w7','cov_w8',
    'ax','ay','az',             # linear accel
    'cov_a0','cov_a1','cov_a2',
    'cov_a3','cov_a4','cov_a5',
    'cov_a6','cov_a7','cov_a8'
]
df = pd.read_csv(csv_path, header=None, names=cols)
df = df.dropna(subset=['qx','qy','qz','qw','wz','ay'])

# normalize time
df['time'] = df['sec'] + df['nsec'] * 1e-9
df['time'] -= df['time'].iloc[0]

# ——— EXTRACT YAW & YAW RATE ———
yaw = np.arctan2(
    2*(df['qw']*df['qz'] + df['qx']*df['qy']),
    1 - 2*(df['qy']**2 + df['qz']**2)
)
yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # wrap
wyaw = df['wz'].values

# ——— EXTRACT LATERAL ACCELERATION ———
ay = df['ay'].values

# ——— STATS ———
mean_yaw  = np.mean(yaw)
std_yaw   = np.std(yaw)
mean_wyaw = np.mean(wyaw)
std_wyaw  = np.std(wyaw)
mean_ay   = np.mean(ay)
std_ay    = np.std(ay)

print(f"Yaw   mean: {mean_yaw:.6f} rad, std: {std_yaw:.6f} rad")
print(f"YawRate mean: {mean_wyaw:.6f} rad/s, std: {std_wyaw:.6f} rad/s")
print(f"Acc x mean: {mean_ay:.6f} m/s², std: {std_ay:.6f} m/s²")

# ——— KDE & PLOT ———
fig, axes = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

# 1) Yaw KDE
kde_y = gaussian_kde(yaw)
grid_y = np.linspace(-np.pi, np.pi, 400)
axes[0].plot(grid_y, kde_y(grid_y), label='Yaw KDE')
axes[0].axvline(gt_yaw, color='red', linestyle='--', label='GT Yaw')
axes[0].set_title('Yaw Angle Density')
axes[0].set_xlabel('Yaw [rad]')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True)

# 2) Yaw Rate KDE
kde_w = gaussian_kde(wyaw)
grid_w = np.linspace(wyaw.min(), wyaw.max(), 400)
axes[1].plot(grid_w, kde_w(grid_w), label='Yaw Rate KDE')
axes[1].axvline(gt_wyaw, color='red', linestyle='--', label='GT Rate')
axes[1].set_title('Yaw Rate Density')
axes[1].set_xlabel('Yaw Rate [rad/s]')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True)

# 3) Lateral Acceleration KDE
kde_a = gaussian_kde(ay)
grid_a = np.linspace(ay.min(), ay.max(), 400)
axes[2].plot(grid_a, kde_a(grid_a), label='Lat Accel KDE')
axes[2].axvline(gt_ay, color='red', linestyle='--', label='GT Lat Accel')
axes[2].set_title('Lateral Acceleration Density')
axes[2].set_xlabel('Ay [m/s²]')
axes[2].set_ylabel('Density')
axes[2].legend()
axes[2].grid(True)

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ——— CONFIG ———
csv_path = "data/imu.csv"  # adjust to your imu.csv path
gt_yaw   = -1.2 # [rad] ground-truth yaw offset
gt_wyaw  =  0.0 # [rad/s] ground-truth yaw rate

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
    'ax','ay','az',             # linear accel (unused)
    'cov_a0','cov_a1','cov_a2',
    'cov_a3','cov_a4','cov_a5',
    'cov_a6','cov_a7','cov_a8'
]
df = pd.read_csv(csv_path, header=None, names=cols)
df = df.dropna(subset=['qx','qy','qz','qw','wz'])

# normalize time
df['time'] = df['sec'] + df['nsec'] * 1e-9
df['time'] -= df['time'].iloc[0]

# ——— EXTRACT YAW & YAW RATE ———
# yaw = atan2(2*(w*z + x*y), 1 - 2*(y²+z²))
yaw = np.arctan2(2*(df['qw'] * df['qz'] + df['qx'] * df['qy']),
                 1 - 2*(df['qy']**2 + df['qz']**2))
yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # wrap

wyaw = df['wz'].values

# ——— STATS ———
std_yaw   = np.std(yaw)
std_wyaw  = np.std(wyaw)
mean_yaw  = np.mean(yaw)
mean_wyaw = np.mean(wyaw)

print(f"Yaw   mean: {mean_yaw:.6f} rad, std: {std_yaw:.6f} rad")
print(f"YawRate mean: {mean_wyaw:.6f} rad/s, std: {std_wyaw:.6f} rad/s")

# ——— KDE & PLOT ———
fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

# Yaw KDE
kde_y = gaussian_kde(yaw)
grid_y = np.linspace(-np.pi, np.pi, 400)
axes[0].plot(grid_y, kde_y(grid_y), label='Yaw KDE')
axes[0].axvline(gt_yaw, color='red', linestyle='--', label='GT Yaw')
axes[0].set_title('Yaw Angle Density')
axes[0].set_xlabel('Yaw [rad]')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True)

# Yaw Rate KDE
kde_w = gaussian_kde(wyaw)
grid_w = np.linspace(wyaw.min(), wyaw.max(), 400)
axes[1].plot(grid_w, kde_w(grid_w), label='Yaw Rate KDE')
axes[1].axvline(gt_wyaw, color='red', linestyle='--', label='GT Rate')
axes[1].set_title('Yaw Rate Density')
axes[1].set_xlabel('Yaw Rate [rad/s]')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True)

plt.show()

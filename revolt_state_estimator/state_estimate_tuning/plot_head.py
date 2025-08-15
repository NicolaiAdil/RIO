import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ——— CONFIG ———
csv_path = "data/heading.csv"  # adjust path if needed
heading_gt = -2.57  # [rad] set your ground-truth yaw here

np.set_printoptions(suppress=True, precision=10)

# ——— LOAD & PREPROCESS ———
cols = ['sec', 'nsec', 'frame_id', 'qx', 'qy', 'qz', 'qw']
df = pd.read_csv(csv_path, header=None, names=cols)
df = df.dropna(subset=['qx','qy','qz','qw'])

# normalize time (optional)
df['time'] = df['sec'] + df['nsec'] * 1e-9
df['time'] -= df['time'].iloc[0]

# ——— EXTRACT YAW ANGLE ———
qx, qy, qz, qw = df['qx'], df['qy'], df['qz'], df['qw']
yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
# wrap to [-pi, pi]
yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

# ——— STATISTICS ———
mean_yaw = np.mean(yaw)
std_yaw  = np.std(yaw)
print(f"Yaw mean: {mean_yaw:.6f} rad")
print(f"Yaw  std: {std_yaw:.6f} rad")

# ——— KDE & PLOT DISTRIBUTION WITH GT ———
kde = gaussian_kde(yaw)
angles = np.linspace(-np.pi, np.pi, 500)
density = kde(angles)

plt.figure(figsize=(8,4))
plt.plot(angles, density, linewidth=2, label='KDE density')
plt.axvline(heading_gt, color='red', linestyle='--', label='Ground Truth')
plt.xlabel('Yaw (rad)')
plt.ylabel('Density')
plt.title('Yaw Angle Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

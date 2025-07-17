import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ——— CONFIG ———
csv_path = "data/vel.csv"  # adjust to your CSV path

np.set_printoptions(suppress=True, precision=10)

# ——— GROUND TRUTH VELOCITY ———
gt_vx = 0.0   # [m/s]
gt_vy = 0.0   # [m/s]

# ——— LOAD & PREPROCESS ———
cols = ['sec', 'nsec', 'frame_id', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
df = pd.read_csv(csv_path, header=None, names=cols)
df = df.dropna(subset=['vx','vy','wz'])

# normalize time if needed
df['time'] = df['sec'] + df['nsec'] * 1e-9
df['time'] -= df['time'].iloc[0]

# ——— EXTRACT DATA & STATS ———
data_dict = {
    'vx': (df['vx'].values, gt_vx, 'Forward Velocity (m/s)'),
    'vy': (df['vy'].values, gt_vy, 'Lateral Velocity (m/s)'),
}

fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

for ax, key in zip(axes, data_dict):
    data, gt, label = data_dict[key]
    std = np.std(data)
    ax.axvline(gt, color='red', linestyle='--', label='Ground Truth')

    if std > 0:
        kde = gaussian_kde(data)
        grid = np.linspace(data.min(), data.max(), 300)
        density = kde(grid)
        ax.plot(grid, density, label=f'KDE of {key}')
    else:
        # constant data: annotate
        ax.text(0.5, 0.5, 'Constant data', 
                transform=ax.transAxes, 
                ha='center', va='center',
                bbox=dict(boxstyle='round', fc='wheat', ec='gray'))
    
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

# ——— PRINT STATS ———
print("Vehicle velocities std-dev:")
for key, (data, _, _) in data_dict.items():
    print(f"  σ_{key} = {np.std(data):.6f}")

plt.show()

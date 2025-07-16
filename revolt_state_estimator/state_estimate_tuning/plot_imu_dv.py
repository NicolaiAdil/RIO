# plot_imu_dv.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ——— CONFIG ———
csv_path = "data/imu_dv.csv"   # adjust path if needed
gt_dvx = -0.00100 # [m/s] ground-truth delta-V X
gt_dvy =  0.00025 # [m/s] ground-truth delta-V Y

# ——— LOAD & PREPROCESS ———
cols = ['sec', 'nsec', 'frame_id', 'dvx', 'dvy', 'dvz']
df = pd.read_csv(csv_path, header=None, names=cols)
df = df.dropna(subset=['dvx', 'dvy'])

# normalize time (optional)
df['time'] = df['sec'] + df['nsec'] * 1e-9
df['time'] -= df['time'].iloc[0]

# ——— EXTRACT & STATS ———
dvx = df['dvx'].values
dvy = df['dvy'].values
std_dvx = np.std(dvx)
std_dvy = np.std(dvy)

print("IMU Δv std-dev:")
print(f"  σ_dvx = {std_dvx:.6f} m/s")
print(f"  σ_dvy = {std_dvy:.6f} m/s")

# ——— KDE & PLOT ———
fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

for ax, data, gt, label in [
    (axes[0], dvx, gt_dvx, 'Δv X [m/s]'),
    (axes[1], dvy, gt_dvy, 'Δv Y [m/s]')
]:
    if np.std(data) > 0:
        kde = gaussian_kde(data)
        grid = np.linspace(data.min(), data.max(), 300)
        density = kde(grid)
        ax.plot(grid, density, label=f'KDE of {label}')
    else:
        ax.text(0.5, 0.5, 'Constant data',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', fc='wheat', ec='gray'))
    ax.axvline(gt, color='red', linestyle='--', label='Ground Truth')
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

plt.show()

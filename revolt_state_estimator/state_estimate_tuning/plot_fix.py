import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ——— CONFIG ———
csv_path = "data/fix.csv"  # adjust path if needed
np.set_printoptions(suppress=True, precision=10)

# ——— GROUND TRUTH (meters East, North) ———
gt_east  = 0.0028  # [m] set your ground-truth East coordinate here
gt_north = -0.0041  # [m] set your ground-truth North coordinate here

# ——— LOAD & PREPROCESS ———
cols = [
    'sec','nsec','frame_id','status','service',
    'latitude','longitude','altitude',
    'cov_0','cov_1','cov_2',
    'cov_3','cov_4','cov_5',
    'cov_6','cov_7','cov_8',
    'cov_type'
]
df = pd.read_csv(csv_path, header=None, names=cols)
df = df.dropna(subset=['latitude','longitude'])
df['time'] = df['sec'] + df['nsec'] * 1e-9
df['time'] -= df['time'].iloc[0]

# ——— LAT/LON → ENU (approx) ———
lat0, lon0 = df['latitude'].iloc[0], df['longitude'].iloc[0]
R_earth = 6378137.0
dlat = np.deg2rad(df['latitude'] - lat0)
dlon = np.deg2rad(df['longitude'] - lon0)
north = dlat * R_earth
east  = dlon * R_earth * np.cos(np.deg2rad(lat0))

# ——— CALCULATE ERROR COVARIANCE RELATIVE TO GT ———
errors = np.vstack([east - gt_east, north - gt_north])
cov_error = np.cov(errors)
print("Error covariance [East, North] relative to ground truth:")
print(cov_error)

# ——— KDE & PLOT WITH GROUND TRUTH ———
pts = np.vstack([east, north])
kde = gaussian_kde(pts)
xmin, xmax = east.min(), east.max()
ymin, ymax = north.min(), north.max()
xi, yi = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

plt.figure(figsize=(8,6))
plt.contourf(xi, yi, zi, levels=50, cmap='viridis')
plt.scatter(east, north, s=5, alpha=0.3, label='Measurements')
plt.scatter(gt_east, gt_north, 
            c='red', marker='X', s=100, label='Ground Truth')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.title('GNSS /fix Position Density with Ground Truth')
plt.colorbar(label='Density')
plt.legend()
plt.tight_layout()
plt.show()

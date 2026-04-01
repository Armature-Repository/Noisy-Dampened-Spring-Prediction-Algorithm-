import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ProjectileNet
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load held-out test set
X_test = np.load("test_X.npy")   # raw (un-normalized) [v0, angle_deg, t]
Y_test = np.load("test_Y.npy")   # raw [x, y]

X_mean, X_std = np.load("X_mean.npy"), np.load("X_std.npy")
Y_mean, Y_std = np.load("Y_mean.npy"), np.load("Y_std.npy")

# Load best model
model = ProjectileNet()
model.load_state_dict(torch.load("model_best.pth", map_location="cpu"))
model.eval()

# Predict on full test set
X_norm = (X_test - X_mean) / X_std
inp    = torch.tensor(X_norm, dtype=torch.float32)

with torch.no_grad():
    pred_norm = model(inp).numpy()

# Denormalize predictions back to real units
pred = pred_norm * Y_std + Y_mean
true = Y_test

# --- Metrics ---
mae  = np.abs(pred - true).mean(axis=0)
rmse = np.sqrt(((pred - true) ** 2).mean(axis=0))
print(f"Test set size : {len(true):,}")
print(f"MAE  — x: {mae[0]:.3f} m  |  y: {mae[1]:.3f} m")
print(f"RMSE — x: {rmse[0]:.3f} m  |  y: {rmse[1]:.3f} m")

# --- Plot 1: Predicted vs True (scatter, 2000 samples) ---
idx = np.random.choice(len(true), 2000, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (label, unit) in enumerate([("x position", "m"), ("y position", "m")]):
    axes[i].scatter(true[idx, i], pred[idx, i], alpha=0.3, s=8)
    lo = min(true[idx, i].min(), pred[idx, i].min())
    hi = max(true[idx, i].max(), pred[idx, i].max())
    axes[i].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    axes[i].set_xlabel(f"True {label} ({unit})")
    axes[i].set_ylabel(f"Predicted {label} ({unit})")
    axes[i].set_title(f"{label}  |  MAE={mae[i]:.3f}m  RMSE={rmse[i]:.3f}m")
    axes[i].legend(); axes[i].grid(True)
plt.suptitle("Predicted vs True — Test Set (held-out 20%)")
plt.tight_layout()
plt.savefig("eval_scatter.png")
print("Saved eval_scatter.png")

# --- Plot 2: Full trajectory comparison for 4 random shots ---
g = 9.81
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Pick 4 diverse angles from the test set
sample_configs = [
    (20, 30), (35, 45), (45, 60), (50, 75)   # (v0, angle)
]

for ax, (v0, angle_deg) in zip(axes, sample_configs):
    angle_rad  = np.deg2rad(angle_deg)
    flight_t   = 2 * v0 * np.sin(angle_rad) / g
    times      = np.linspace(0, flight_t, 200)

    # True physics
    true_x = v0 * np.cos(angle_rad) * times
    true_y = v0 * np.sin(angle_rad) * times - 0.5 * g * times**2

    # Model predictions point-by-point
    raw = np.column_stack([
        np.full_like(times, v0),
        np.full_like(times, angle_deg),
        times
    ]).astype(np.float32)
    raw_norm = (raw - X_mean) / X_std
    with torch.no_grad():
        preds = model(torch.tensor(raw_norm)).numpy() * Y_std + Y_mean

    ax.plot(true_x,    true_y,    "b-",  lw=2,   label="True physics")
    ax.plot(preds[:,0], preds[:,1], "r--", lw=2,  label="Model prediction")
    ax.set_title(f"v₀={v0} m/s  |  θ={angle_deg}°")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(); ax.grid(True)
    ax.set_ylim(bottom=0)

plt.suptitle("Trajectory Comparison — Model vs True Physics")
plt.tight_layout()
plt.savefig("eval_trajectories.png")
print("Saved eval_trajectories.png")
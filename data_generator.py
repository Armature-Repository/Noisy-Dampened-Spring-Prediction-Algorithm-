import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def generate_projectile_data(n=50000, noise_std=0.05):
    g = 9.81

    v0     = np.random.uniform(5, 50, size=n)
    angles = np.random.uniform(5, 85, size=n)
    angle_rad = np.deg2rad(angles)

    # Random time: sample within each trajectory's flight time
    flight_time = 2 * v0 * np.sin(angle_rad) / g
    t = np.random.uniform(0, 1, size=n) * flight_time  # always within arc

    x = v0 * np.cos(angle_rad) * t
    y = v0 * np.sin(angle_rad) * t - 0.5 * g * t**2

    x += np.random.normal(0, noise_std, size=n)
    y += np.random.normal(0, noise_std, size=n)

    # inputs: [v0, angle_deg, t], outputs: [x, y]
    inputs  = np.stack([v0, angles, t], axis=1).astype(np.float32)
    outputs = np.stack([x, y],         axis=1).astype(np.float32)

    return inputs, outputs

if __name__ == "__main__":
    X, Y = generate_projectile_data()

    # Compute and save normalization stats from the FULL dataset before splitting
    X_mean, X_std = X.mean(0), X.std(0)
    Y_mean, Y_std = Y.mean(0), Y.std(0)

    split = int(0.8 * len(X))  # 80% train, 20% held-out test
    np.save("train_X.npy", X[:split]);  np.save("test_X.npy", X[split:])
    np.save("train_Y.npy", Y[:split]);  np.save("test_Y.npy", Y[split:])
    np.save("X_mean.npy", X_mean);      np.save("X_std.npy", X_std)
    np.save("Y_mean.npy", Y_mean);      np.save("Y_std.npy", Y_std)

    print(f"Train: {split:,}  |  Test: {len(X)-split:,}")
    print("Saved all .npy files.")
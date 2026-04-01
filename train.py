import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from model import ProjectileNet
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

X = np.load("train_X.npy")
Y = np.load("train_Y.npy")

X_mean, X_std = np.load("X_mean.npy"), np.load("X_std.npy")
Y_mean, Y_std = np.load("Y_mean.npy"), np.load("Y_std.npy")

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

inputs  = torch.tensor(X_norm, dtype=torch.float32)
targets = torch.tensor(Y_norm, dtype=torch.float32)

val_split = int(0.9 * len(inputs))
train_ds = TensorDataset(inputs[:val_split], targets[:val_split])
val_ds   = TensorDataset(inputs[val_split:], targets[val_split:])

# num_workers=0 is required on Windows — data is already in RAM so no speed loss
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

model     = ProjectileNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)  # removed verbose=True
loss_fn   = nn.MSELoss()

epochs = 100
train_losses, val_losses = [], []
best_val = float("inf")

for epoch in range(1, epochs + 1):
    model.train()
    t_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(bx), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item()

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            v_loss += loss_fn(model(bx), by).item()

    t_avg = t_loss / len(train_loader)
    v_avg = v_loss / len(val_loader)
    train_losses.append(t_avg)
    val_losses.append(v_avg)
    scheduler.step(v_avg)

    # Print LR the non-deprecated way
    current_lr = optimizer.param_groups[0]['lr']
    if best_val > v_avg:
        best_val = v_avg
        torch.save(model.state_dict(), "model_best.pth")

    print(f"Epoch {epoch:03d}/{epochs} | Train: {t_avg:.6f} | Val: {v_avg:.6f} | LR: {current_lr:.2e}")

torch.save(model.state_dict(), "model.pth")

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Val")
plt.xlabel("Epoch"); plt.ylabel("Normalized MSE")
plt.title("Training Loss"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("loss.png")
print(f"\nDone. Best val loss: {best_val:.6f}")
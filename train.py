import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import MLPModel

# --------- Config ----------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = 30
USE_INPUT_NOISE = False
NOISE_STD = 0.03
NOISE_PROB = 0.5
# --------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)

if not os.path.exists("assets"):
    os.makedirs("assets")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

df = pd.read_csv("dataset.csv")

feature_cols = [
    'P1_Arrival', 'P2_Arrival', 'P3_Arrival', 'P4_Arrival',
    'P1_Burst', 'P2_Burst', 'P3_Burst', 'P4_Burst'
]

if not set(feature_cols).issubset(df.columns):
    # try fallback by position: first 8 columns if names differ
    X_all = df.iloc[:, :8].values
else:
    X_all = df[feature_cols].values

y_series = df["Best_Algorithm"].astype("category")
y_all = y_series.cat.codes.values
class_names = list(y_series.cat.categories)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

scaler_path = f"assets/scaler_{current_time}.pkl"
joblib.dump(scaler, scaler_path)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
y_test = torch.tensor(y_test_np, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

num_features = X_train.shape[1]
num_classes = len(class_names)

model = MLPModel(input_size=num_features, output_size=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR
)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")
best_state = None
no_improve = 0
best_model_path = f"assets/best_model_{current_time}.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if USE_INPUT_NOISE and np.random.rand() < NOISE_PROB:
            xb = xb + torch.randn_like(xb) * NOISE_STD

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    epoch_train_loss = running_loss / max(total, 1)
    epoch_train_acc = correct / max(total, 1)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)

    epoch_val_loss = val_running_loss / max(val_total, 1)
    epoch_val_acc = val_correct / max(val_total, 1)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)

    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(epoch_val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f"[Epoch {epoch}] LR changed: {old_lr:.6f} -> {new_lr:.6f}")

    print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {epoch_train_loss:.4f} Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_state = model.state_dict()
        torch.save(best_state, best_model_path)
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    outputs = model(X_test.to(device))
    _, preds = torch.max(outputs, 1)
    preds_cpu = preds.cpu().numpy()

final_acc = accuracy_score(y_test_np, preds_cpu)
final_val_loss = val_losses[-1] if val_losses else float("nan")

report = classification_report(y_test_np, preds_cpu, target_names=class_names, digits=4)
matrix = confusion_matrix(y_test_np, preds_cpu)

print("\nFinal Test Accuracy:", round(final_acc, 4))
print("Final Test Loss:", round(final_val_loss, 4))
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", matrix)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plot_path = f"assets/performance_plot_{current_time}.png"
plt.savefig(plot_path)

last_model_path = f"assets/last_model_{current_time}.pth"
torch.save(model.state_dict(), last_model_path)

report_path = f"assets/report_{current_time}.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Device: {device}\n")
    f.write(f"Final Test Accuracy: {final_acc:.4f}\n")
    f.write(f"Final Test Loss: {final_val_loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))
    f.write("\n\nArtifacts:\n")
    f.write(f"Best model: {best_model_path}\n")
    f.write(f"Last model: {last_model_path}\n")
    f.write(f"Scaler: {scaler_path}\n")
    f.write(f"Plot: {plot_path}\n")

plt.show()

print(f"\nSaved: \n- Plot: {plot_path}\n- Best model: {best_model_path}\n- Last model: {last_model_path}\n- Report: {report_path}\n- Scaler: {scaler_path}")

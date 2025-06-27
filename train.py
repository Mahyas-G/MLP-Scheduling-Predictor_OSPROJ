import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model import MLPModel
import os
from datetime import datetime

if not os.path.exists("assets"):
    os.makedirs("assets")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

df = pd.read_csv("dataset.csv")
X = df.drop("Best_Algorithm", axis=1).values
y = df["Best_Algorithm"].astype("category").cat.codes.values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = MLPModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    _, predicted = torch.max(outputs.data, 1)
    train_acc = accuracy_score(y_train, predicted)
    train_accuracies.append(train_acc)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

        _, test_pred = torch.max(test_outputs.data, 1)
        test_acc = accuracy_score(y_test, test_pred)
        test_accuracies.append(test_acc)

final_test_acc = round(test_accuracies[-1], 4)
final_test_loss = round(test_losses[-1], 4)
report = classification_report(y_test, test_pred)
matrix = confusion_matrix(y_test, test_pred)

print("\nFinal Test Accuracy:", final_test_acc)
print("Final Test Loss:", final_test_loss)
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", matrix)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'r-', label='Train Loss')
plt.axhline(y=test_losses[-1], color='b', linestyle='--', label=f'Test Loss = {test_losses[-1]:.4f}')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss (Train vs Test)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'g-', label='Train Accuracy')
plt.axhline(y=test_accuracies[-1], color='orange', linestyle='--', label=f'Test Acc = {test_accuracies[-1]:.4f}')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy (Train vs Test)")
plt.legend()

plt.tight_layout()

plot_path = f"assets/performance_plot_{current_time}.png"
model_path = f"assets/model_{current_time}.pth"
report_path = f"assets/report_{current_time}.txt"

plt.savefig(plot_path)
torch.save(model.state_dict(), model_path)

with open(report_path, 'w') as f:
    f.write(f"Final Test Accuracy: {final_test_acc}\n")
    f.write(f"Final Test Loss: {final_test_loss}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))

plt.show()

print(f"\nResults saved in: {plot_path}, {model_path}, {report_path}")

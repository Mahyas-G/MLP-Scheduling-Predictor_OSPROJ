import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model import MLP
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
X = df.iloc[:, :-1].values
y = df['Best_Algorithm'].values

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

epochs = 100
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        preds = torch.argmax(test_output, dim=1)
        acc = (preds == y_test).float().mean().item()
        test_accuracies.append(acc)

print(f"Final Test Accuracy: {test_accuracies[-1]*100:.2f}%")

plt.plot(train_losses, label='Loss')
plt.plot(test_accuracies, label='Accuracy')
plt.legend()
plt.show()

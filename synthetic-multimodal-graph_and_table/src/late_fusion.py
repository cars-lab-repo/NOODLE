import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

tabular_data = pd.read_csv('tabular_dataset.csv')

json_data = []
for i in range(1, 11):
    with open(f'{i}.json', 'r') as json_file:
        json_content = json.load(json_file)
        json_features = np.array(list(json_content['features'].values())).flatten()
        json_data.append(json_features)

y = tabular_data['Label'].values

X_tabular_train, X_tabular_test, y_train, y_test = train_test_split(
    tabular_data.iloc[:, 1:-1].values, y, test_size=0.2, random_state=42)

X_json_train, X_json_test, _, _ = train_test_split(
    json_data, y, test_size=0.2, random_state=42)

X_tabular_train_tensor = torch.tensor(X_tabular_train, dtype=torch.float32)
X_tabular_test_tensor = torch.tensor(X_tabular_test, dtype=torch.float32)

X_json_train_tensor = torch.tensor(X_json_train, dtype=torch.float32)
X_json_test_tensor = torch.tensor(X_json_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_tabular_data = TensorDataset(X_tabular_train_tensor, y_train_tensor)
train_json_data = TensorDataset(X_json_train_tensor, y_train_tensor)

train_tabular_loader = DataLoader(train_tabular_data, batch_size=64, shuffle=True)
train_json_loader = DataLoader(train_json_data, batch_size=64, shuffle=True)

test_tabular_data = TensorDataset(X_tabular_test_tensor, y_test_tensor)
test_json_data = TensorDataset(X_json_test_tensor, y_test_tensor)

test_tabular_loader = DataLoader(test_tabular_data, batch_size=64, shuffle=False)
test_json_loader = DataLoader(test_json_data, batch_size=64, shuffle=False)

class TabularClassifier(nn.Module):
    def __init__(self, input_size):
        super(TabularClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

class JsonClassifier(nn.Module):
    def __init__(self, input_size):
        super(JsonClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

input_tabular_size = X_tabular_train.shape[1]
input_json_size = X_json_train_tensor.shape[1]

tabular_model = TabularClassifier(input_tabular_size)
json_model = JsonClassifier(input_json_size)

tabular_optimizer = optim.Adam(tabular_model.parameters(), lr=0.001)
json_optimizer = optim.Adam(json_model.parameters(), lr=0.001)

criterion = nn.BCELoss()

tabular_epochs = 10
tabular_train_losses = []
tabular_test_losses = []

for epoch in range(tabular_epochs):
    tabular_model.train()
    tabular_train_loss = 0.0
    for data, target in tqdm(train_tabular_loader, desc=f'Tabular Epoch {epoch + 1}'):
        tabular_optimizer.zero_grad()
        tabular_output = tabular_model(data)
        tabular_loss = criterion(tabular_output, target.unsqueeze(1))
        tabular_loss.backward()
        tabular_optimizer.step()
        tabular_train_loss += tabular_loss.item()
    tabular_train_losses.append(tabular_train_loss / len(train_tabular_loader))

    tabular_model.eval()
    tabular_test_loss = 0.0
    tabular_predictions = []
    with torch.no_grad():
        for data, target in test_tabular_loader:
            tabular_output = tabular_model(data)
            tabular_loss = criterion(tabular_output, target.unsqueeze(1))
            tabular_test_loss += tabular_loss.item()
            tabular_predictions.extend(tabular_output.cpu().numpy())
    tabular_test_losses.append(tabular_test_loss / len(test_tabular_loader))

json_epochs = 10
json_train_losses = []
json_test_losses = []

for epoch in range(json_epochs):
    json_model.train()
    json_train_loss = 0.0
    for data, target in tqdm(train_json_loader, desc=f'JSON Epoch {epoch + 1}'):
        json_optimizer.zero_grad()
        json_output = json_model(data)
        json_loss = criterion(json_output, target.unsqueeze(1))
        json_loss.backward()
        json_optimizer.step()
        json_train_loss += json_loss.item()
    json_train_losses.append(json_train_loss / len(train_json_loader))

    json_model.eval()
    json_test_loss = 0.0
    json_predictions = []
    with torch.no_grad():
        for data, target in test_json_loader:
            json_output = json_model(data)
            json_loss = criterion(json_output, target.unsqueeze(1))
            json_test_loss += json_loss.item()
            json_predictions.extend(json_output.cpu().numpy())
    json_test_losses.append(json_test_loss / len(test_json_loader))

combined_predictions = [p1 * p2 for p1, p2 in zip(tabular_predictions, json_predictions)]

y_pred_combined = [1 if pred > 0.5 else 0 for pred in combined_predictions]
accuracy_combined = accuracy_score(y_test, y_pred_combined)
print(f'Combined Model Accuracy: {accuracy_combined}')

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, tabular_epochs + 1), tabular_train_losses, label='Tabular Train Loss')
plt.plot(range(1, tabular_epochs + 1), tabular_test_losses, label='Tabular Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Tabular Model')

plt.subplot(1, 2, 2)
plt.plot(range(1, json_epochs + 1), json_train_losses, label='JSON Train Loss')
plt.plot(range(1, json_epochs + 1), json_test_losses, label='JSON Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('JSON Model')

plt.tight_layout()
plt.show()

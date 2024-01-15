import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss


#data = pd.read_csv('gan_graph_data.csv') 
data = pd.read_csv('actual_graph_22_16features.csv')  

features = data.iloc[:, :-1].values
labels = data['Trojan'].values


X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=42)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3) 
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 7, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model = SimpleCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train.unsqueeze(1)) 
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


with torch.no_grad():
    outputs = model(X_test.unsqueeze(1))
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_test.view(-1, 1)).sum().item() / len(y_test)
    brier_score = brier_score_loss(y_test, outputs.numpy())
    print(f'Brier Score: {brier_score:.4f}')

print(f'Test Accuracy: {accuracy*100:.2f}%')

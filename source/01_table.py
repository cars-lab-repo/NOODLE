import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss  # Add brier_score_loss
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


#data = pd.read_csv('gan_tabular_data_10001.csv') 
data = pd.read_csv('actual_table_22_16features.csv')  
X = data.iloc[:, :-1].values
y = data['Trojan'].values
# Reshape data for 1D Convolutional layer
X = X.reshape(X.shape[0], 1, X.shape[1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 5, 64)  
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 64 * 5)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = CNNModel()


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())


for epoch in range(50):
    epoch_loss = 0.0  
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  

    
    print(f'Epoch [{epoch+1}/10], Loss: {epoch_loss / len(train_loader):.4f}')

from sklearn.metrics import accuracy_score

with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y_test_tensor.unsqueeze(1)).sum().item() / len(y_test_tensor)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

   
    roc_auc = roc_auc_score(y_test, outputs.numpy())

    
    brier_score = brier_score_loss(y_test, outputs.numpy())
    print(f'Brier Score: {brier_score:.4f}')

   
    fpr, tpr, _ = roc_curve(y_test, outputs.numpy())
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

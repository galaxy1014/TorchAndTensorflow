import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# read csv file
df = pd.read_csv('housing.csv')
print(df.head())

df.columns = np.array(range(0, len(df.columns)))
print(df.head())

# Train data

x_train = df.iloc[:, range(0, 3)]
x_train = torch.FloatTensor(x_train.values)
print(x_train)

y_train = df.iloc[:, 3]
y_train = torch.FloatTensor(y_train.values)
y_train = y_train.view(y_train.size(0), 1)
print(y_train)

# Normalization

x_mean = torch.mean(x_train)
x_std = torch.std(x_train)
x_train -= x_mean
x_train /= x_std
print(x_train)

y_mean = torch.mean(y_train)
y_std = torch.std(y_train)
y_train -= y_mean
y_train /= y_std
print(y_train)

# Model

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(3, 128)
        self.layer1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        ) 
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        out = self.input(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output(out)
        return out

model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training

for epoch in range(1000):
    prediction = model(x_train)
    cost = criterion(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch : {}, cost : {:.6f}'.format(epoch, cost.item()))

# Prediction

new_var = torch.FloatTensor([[-0.8156, -1.2938,  0.7935]])
print(model(new_var))
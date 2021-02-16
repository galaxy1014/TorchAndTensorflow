# Classfying Fashion MNIST using single CNN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

# train data
train_data = datasets.FashionMNIST('/F_MNIST_Data', download=True, train=True,
                                  transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, 
                                          drop_last=True)

# test data
test_data = datasets.FashionMNIST('/F_MNIST_Data', download=True, train=False, 
                                 transform=transforms.ToTensor())

# Model
class SingleCNN(nn.Module):
    def __init__(self):
        super(SingleCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear = nn.Linear(13*13*64, 10)
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

model = SingleCNN().to('cuda:0')
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(30):
    avg_cost = 0
    for X, Y in train_loader:
        X = X.to('cuda:0')
        Y = Y.to('cuda:0')
        
        optimizer.zero_grad()
        hypothesis = model(X).to('cuda:0')
        cost = criterion(hypothesis, Y).to('cuda:0')
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / len(train_loader)
        
    print('Epoch : {}, cost : {:.6f}, avg_cost : {:.6f}'.format(
        epoch, cost.item(), avg_cost.item()))

# Model Evaluation
with torch.no_grad():
    X_test = test_data.data.view(len(test_data), 1, 28, 28).float().to('cuda:0')
    Y_test = test_data.targets.to('cuda:0')
    
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: ', accuracy.item())


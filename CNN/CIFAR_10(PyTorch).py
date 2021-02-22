#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms


# In[2]:


transform = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = datasets.CIFAR10(root='CIFAR10', train=True, 
                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=32)

test_set = datasets.CIFAR10(root='CIFAR10', train=False, download=True,
                           transform=transform)

test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=32)


# In[3]:


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),           
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2*2*128, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 10),
            nn.Softmax()
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# In[4]:


model = CNN().to('cuda:0')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.9)


# In[5]:


from torchsummary import summary
summary(model, (3, 32, 32))


# In[6]:


for epoch in range(1000):
    
    for X, Y in train_loader:
        X = X.to('cuda:0')
        Y = Y.to('cuda:0')
        
        optimizer.zero_grad()
        prediction = model(X).to('cuda:0')
        cost = criterion(prediction, Y)
        cost.backward()
        optimizer.step()
    
    print('Epoch: {}, cost: {:.6f}'.format(epoch+1, cost.item()))


# In[7]:


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to('cuda:0'), data[1].to('cuda:0')
        outputs = model(images).to('cuda:0')
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# In[ ]:





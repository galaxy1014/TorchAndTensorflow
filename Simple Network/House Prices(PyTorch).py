#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 30k per bedroom
# house costs is 50k
# ex. 2 bedroom = 50 + (30 * 2) = 110
# create nueral network that learns this relationship 
# predict 10 bedroom house costs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# In[2]:


# train data
x_train = torch.FloatTensor(np.arange(1, 11))
x_train = x_train.reshape(-1, 1)
print(x_train)


# In[3]:


y_train = torch.FloatTensor(np.arange(0.3, 3.0, 0.3))
y_train = y_train.reshape(-1, 1)
print(y_train)


# In[4]:


# simple model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)


# In[10]:


model = NeuralNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[11]:


for epoch in range(101):
    predict = model(x_train)
    cost = torch.nn.functional.mse_loss(predict, y_train).to('cuda:0')
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print('Epoch : {}, cost: {}'.format(epoch + 1, cost.item()))


# In[12]:


# 7 bedroom -> 210k
model(torch.FloatTensor([[7.0]]))


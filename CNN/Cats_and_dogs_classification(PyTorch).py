import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Train and Test Data
compose = transforms.Compose([transforms.Resize((150, 150)),
                               transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])])
train_data = ImageFolder(root = 'Cats_and_dogs/train',
                                             transform = compose)
test_data = ImageFolder(root = 'Cats_and_dogs/validation',
                                            transform = compose)

der = DataLoader(train_data, batch_size = 64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle=True)

# Model

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Sequential(
            nn.Linear(36*36*64, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.output = nn.Linear(64, 1)
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = out.view(-1, 36*36*64)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.output(out)
        out = torch.sigmoid(out)
        return out

model = DeepCNN().to('cuda:0')
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

from torchsummary import summary
summary(model, (3, 150, 150))

# Training

for epoch in range(200):
    for X, Y in train_loader:
        X = X.to('cuda:0')
        Y = Y.to('cuda:0').float()
        
        optimizer.zero_grad()
        prediction = model(X).to('cuda:0')
        Y = Y.unsqueeze(1)
        cost = criterion(prediction, Y).to('cuda:0')
        cost.backward()
        optimizer.step()
        
    print('Epoch: {}, cost : {:.6f}'.format(epoch + 1,cost.item()))

# Testing

with torch.no_grad():
    for data in test_loader:
        X, Y = data[0].to('cuda:0'), data[1].to('cuda:0').float()
        Y = Y.unsqueeze(1)
        prediction = model(X).to('cuda:0')
        correct_prediction = torch.argmax(prediction, 1) == Y
        accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())


from PIL import Image 
from matplotlib.pyplot import imshow 
import numpy as np
img = Image.open('sample_image.jpg') 
imshow(np.asarray(img))


def custom_image(img):
    image = compose(img).to('cuda:0')
    image = image.unsqueeze(0)
    print(image.size())
    return image

def run():
    img = Image.open('sample_image.jpg') 
    img = custom_image(img)
    result = model(img).to('cuda:0')
    _, predicted = torch.max(result, 1)
    return predicted

print(run())





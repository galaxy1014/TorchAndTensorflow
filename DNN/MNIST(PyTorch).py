import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


# 데이터 셋 생성

directory = 'MNIST_data/'
mnist_train = MNIST(directory, train=True, transform=transforms.ToTensor(),
                   download=True)
mnist_test = MNIST(directory, train=False, transform=transforms.ToTensor(),
                  download=True)

data_loader = DataLoader(dataset=mnist_train, batch_size=64, shuffle=True,
                        drop_last=True)


# input layer와 output layer를 제외한 2개의 hidden layer로 구성
# input_layer : 28 * 28
# hidden_layer1 : 128
# hidden_layer2 : 64
# output_layer : 10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(28 * 28, 128).to('cuda:0'),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64).to('cuda:0'),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 10).to('cuda:0'),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

# optimizer 및 loss 선언

model = NeuralNetwork()
criterion = nn.CrossEntropyLoss().to('cuda:0')
optimizer = optim.Adam(model.parameters(), lr=0.00001)


for epoch in range(31):
    avg_cost = 0
    
    for X, Y in data_loader:
        # 입력 데이터의 shape를 28 * 28 = 784 길이로 변환
        X = X.view(-1, 28 * 28).to('cuda:0')
        Y = Y.to('cuda:0')
        
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / len(data_loader)
        
    print('Epoch : {}, cost : {:.6f}'.format(epoch + 1, avg_cost))


# 성능 테스트

from matplotlib import pyplot as plt
import random

with torch.no_grad():
    X_test = mnist_test.data.view(-1, 28 * 28).float().to('cuda:0')
    Y_test = mnist_test.targets.to('cuda:0')
    
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
    
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.data[r: r + 1].view(-1, 28 * 28).float().to('cuda:0')
    Y_single_data = mnist_test.targets[r: r+1].to('cuda:0')
    
    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('prediction: ', torch.argmax(single_prediction, 1).item())
    
    plt.imshow(mnist_test.data[r: r+1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()


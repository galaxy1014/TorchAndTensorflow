import torch
import torch.nn as nn
import torch.optim as optim

# train data
x_train = torch.FloatTensor([[73, 80, 75],
                            [90, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Model
# input_dim = 3, output_dim = 1 
model = nn.Linear(3, 1)

optimizer = optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(2001):
    hypothesis = model(x_train)
    cost = F.mse_loss(hypothesis, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch : {}, cost : {}'.format(epoch, cost.item()))

new_var = torch.FloatTensor([[73, 80, 75]])
test_result = model(new_var)


print(test_result)


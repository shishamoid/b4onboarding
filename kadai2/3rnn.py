#https://exture-ri.com/2021/01/12/pytorch-rnn/
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

'''GPUチェック'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#半径
r = 10

x = np.linspace(-10, 7)
#len(x)
#x
print(x)
#y = x-1
y = (r**2-x**2)**(1/2)
#y
print(y)
print(len(x))
print(len(y))
#x
#y#sin_x = np.sin(x) + np.random.normal(0, 0.3, len(x))

n_time = 10
n_sample = len(x) - n_time

input_data = np.zeros((n_sample, n_time, 1))
correct_data = np.zeros((n_sample, 1))

for i in range(n_sample):
    input_data[i] = sin_x[i:i+n_time].reshape(-1, 1)
    correct_data[i] = [sin_x[i+n_time]]
input_data = torch.FloatTensor(input_data)
correct_data = torch.FloatTensor(correct_data)

print(input_data)
print(len(input_data))

#print(correct_data)

'''バッチデータの準備'''
dataset = TensorDataset(input_data, correct_data)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

'''モデルの定義'''
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(1, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(device)
        x_rnn, hidden = self.rnn(x, None)
        x = self.fc(x_rnn[:, -1, :])
        return x
model = RNN().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

record_loss_train = []

for i in range(201):
  model.train()
  loss_train = 0
  for j, (x, t) in enumerate(train_loader):
    loss = criterion(model(x), t.to(device))
    loss_train += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  loss_train /= j+1
  record_loss_train.append(loss_train)
  if i%10 == 0:
    print("Epoch:", i, "Loss_Train:", loss_train)
    predicted = list(input_data[0].reshape(-1))
    model.eval()
    with torch.no_grad():
      for i in range(n_sample):
        x = torch.tensor(predicted[-n_time:])
        x = x.reshape(1, n_time, 1)
        predicted.append(model(x)[0].item())

plt.plot(range(len(record_loss_train)), record_loss_train, label='Train')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

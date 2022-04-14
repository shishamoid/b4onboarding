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
r = 2

x_up = np.linspace(-1*r, r,num=25)
y_up = (r**2-x_up**2)**(1/2)

x_down = np.linspace(-1*r,r,num=25)
y_down = -(r**2-x_down**2)**(1/2)

correct_x = np.concatenate([x_up,x_down])
correct_y = np.concatenate([y_up,y_down])

input_data = []
correct_data = []

for i in range(len(correct_x)):
    if i==49:
        input_data.append([correct_x[i],correct_y[i]])
        correct_data.append([correct_x[0],correct_y[0]])
    else:
        input_data.append([correct_x[i],correct_y[i]])
        correct_data.append([correct_x[i+1],correct_y[i+1]])


input_data = torch.FloatTensor(input_data)
correct_data = torch.FloatTensor(correct_data)

'''バッチデータの準備'''

dataset = TensorDataset(input_data,correct_data)
train_loader = DataLoader(dataset, batch_size=8, shuffle=False)

'''モデルの定義'''
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(2, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        #print(x)
        x = x.unsqueeze(1)

        x = x.to(device)
        x_rnn, hidden = self.rnn(x, None)
        x = self.fc(x_rnn[:, -1, :])
        return x
model = RNN().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

record_loss_train = []
predict_list=[]
#pre = input_data[0:10]
pre=input_data

for i in range(600):
  model.train()
  loss_train = 0
  for j,xy in enumerate(train_loader):

    xy[0] = xy[0].to(device)
    xy[1] = xy[1].to(device)

    loss = criterion(model(xy[0]),xy[1])
    #print(xy[1])
    loss_train += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  loss_train /= j+1

  if i%10 == 0:

    print("Epoch:", i, "Loss_Train:", loss_train)
    model.eval()


##円をかく
pre = pre.cpu()

result_x = []
result_y = []

#print(pre)
for i in range(50):
    result = model(pre.unsqueeze(1)[i])
    result = result.cpu()
    #print(result)
    result_x.append(result[0][0])
    result_y.append(result[0][1])

plt.plot(result_x,result_y,linestyle="None",linewidth=0,marker='o')
plt.show()

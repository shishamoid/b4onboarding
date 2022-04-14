#https://www.hellocybernetics.tech/entry/2017/10/20/025702
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

'''GPUチェック'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#データ
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


class feedforward(nn.Module):
    def __init__(self):
        super(feedforward,self).__init__()
        self.l1 = nn.Linear(2,64)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(64,64)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(64,2)
        self.a4 = nn.ReLU()

    def forward(self,x):
        #x = x.unsqueeze(1)
        #print(x)
        #print(x.shape)
        x = self.l1(x)
        h = self.a1(x)
        h = self.a1(self.l2(h))
        y = self.l3(h)
        return y

model = feedforward().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

record_loss_train = []
predict_list=[]

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
model = model.cpu()

result_x = []
result_y = []

#print(pre)
for i in range(50):
    result = model(pre.unsqueeze(1)[i])
    result = result.cpu()
    #print(result)
    #print(result)
    result_x.append(result[0][0])
    result_y.append(result[0][1])

plt.plot(result_x,result_y,linestyle="None",linewidth=0,marker='o')
plt.show()

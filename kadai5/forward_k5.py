import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import copy

def dataload():

    t = np.linspace(0, 2*(math.pi))

    x_list = []
    y_list = []

    for i in range(len(t)):
        x_list.append(math.sin(t[i]+(1/2)*(math.pi)))
        y_list.append(math.sin(2*t[i]))

    correct_x = x_list
    correct_y = y_list

    input_onecircle_data = []
    correct_onecircle_data = []

    for i in range(len(correct_x)):
        if i == 49:
            input_onecircle_data.append([correct_x[i], correct_y[i]])
            correct_onecircle_data.append([correct_x[0], correct_y[0]])
        elif i == 48:
            input_onecircle_data.append([correct_x[i], correct_y[i]])
            correct_onecircle_data.append([correct_x[i+1], correct_y[i+1]])
        elif i == 47:
            input_onecircle_data.append([correct_x[i], correct_y[i]])
            correct_onecircle_data.append([correct_x[i+2], correct_y[i+2]])
        else:
            input_onecircle_data.append([correct_x[i], correct_y[i]])
            correct_onecircle_data.append([correct_x[i+3], correct_y[i+3]])

    input_data = []
    correct_data = []

    for i in range(200):
        input_data.append(copy.deepcopy(input_onecircle_data))
        correct_data.append(copy.deepcopy(correct_onecircle_data))

    input_data = torch.FloatTensor(input_data)
    correct_data = torch.FloatTensor(correct_data)

    dataset = TensorDataset(input_data, correct_data)
    train_loader = DataLoader(dataset, batch_size=12, shuffle=False)
    return input_data, train_loader


class feedforward(nn.Module):
    def __init__(self):
        super(feedforward, self).__init__()
        self.l1 = nn.Linear(2, 64)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(64, 64)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(64, 2)
        self.a4 = nn.ReLU()

    def forward(self, x):

        x = self.l1(x)
        h = self.a1(x)
        h = self.a1(self.l2(h))
        y = self.l3(h)
        return y


def train(train_loader, device, num_epoc):
    model = feedforward().to(device)
    loss_list = []
    epoch_list = []

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03)

    for i in range(num_epoc):
        model.train()
        loss_train = 0
        for j, xy in enumerate(train_loader):

            xy_coo_input = xy[0].to(device)
            xy_coo_correct = xy[1].to(device)

            loss = criterion(model(xy_coo_input), xy_coo_correct)
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train /= j+1
        loss_list.append(loss_train)
        epoch_list.append(i)
        if i % 10 == 0:
            print("Epoch:", i, "Loss_Train:", loss_train)

        if i % 100 == 0:
            torch.save(model.to('cpu').state_dict(),
                       './forward_models/epoch_{}_model.pth'.format(i))
            model.to(device)

    torch.save(model.to('cpu').state_dict(),
               './forward_models/last_epoch_{}_model.pth'.format(num_epoc))

    return model, loss_list, epoch_list


def draw_picture(model, pre):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device",device)
    pre = pre.to(device)
    result_x = []
    result_y = []

    model = model.to(device)

    #print(pre)
    result = model(pre[0][0].unsqueeze(0).unsqueeze(0))#最初だけモデルに入力
    #result = result.cpu()
    result = result.to(device)
    #result = model.cpu()
    #model = model.cpu()
    model = model.to(device)
    #print(pre[0][0].unsqueeze(0)[0])
    result_x.append(pre[0][0].unsqueeze(0)[0][0])
    result_y.append(pre[0][0].unsqueeze(0)[0][1])


    result_x.append(result[0][0][0])
    result_y.append(result[0][0][1])
    #print(result_x,result_y)
    result = result.to(device)
    model = model.to(device)
    pre = pre.to(device)

    count = 0
    while True:
        result = model(result)#クローズドループに変更

        #result = result.cpu()
        result_x.append(result[0][0][0])
        result_y.append(result[0][0][1])
        count +=1
        if count==50:
            break

    #円プロット用
    circle_x=[]
    circle_y=[]
    for i in range(50):
        circle_x.append(pre[0][i][0])
        circle_y.append(pre[0][i][1])

    fig = plt.figure()
    plt.plot(result_x, result_y, linestyle="None", linewidth=0, marker='o')
    plt.plot(circle_x, circle_y)
    fig.savefig("./forward_pictures/forward_Lissajous.png")


def main():
    os.makedirs("./forward_models", exist_ok=True)
    os.makedirs("./forward_pictures", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data, traindata = dataload()
    model, loss_list, epoch_list = train(traindata, device, 930)
    draw_picture(model, input_data)

    fig = plt.figure()
    plt.plot(epoch_list, loss_list)
    fig.savefig("./forward_pictures/loss_forward.png")


if __name__ == '__main__':
    main()

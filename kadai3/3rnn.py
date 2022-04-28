import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import copy

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(2, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        x_rnn, hidden = self.rnn(x, None)
        x = self.fc(x_rnn)
        return x

def dataload():
    r = 2  # 半径

    θ = np.linspace(0, 2*math.pi, num=50)

    correct_x = []
    correct_y = []

    for i in range(len(θ)):
        correct_x.append(r*math.cos(θ[i]))
        correct_y.append(r*math.sin(θ[i]))

    input_onecircle_data = []
    correct_onecircle_data = []

    for i in range(len(correct_x)):
        if i == 49:
            input_onecircle_data.append([correct_x[i], correct_y[i]])
            correct_onecircle_data.append([correct_x[0], correct_y[0]])
        else:

            input_onecircle_data.append([correct_x[i], correct_y[i]])
            correct_onecircle_data.append([correct_x[i+1], correct_y[i+1]])

    input_data = []
    correct_data = []

    for i in range(50):
        input_data.append(copy.deepcopy(input_onecircle_data))
        correct_data.append(copy.deepcopy(correct_onecircle_data))

    input_data = torch.FloatTensor(input_data)
    correct_data = torch.FloatTensor(correct_data)

    dataset = TensorDataset(input_data, correct_data)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=False)

    return input_data, train_loader


def train(train_loader, device, num_epoch):
    epoch_list = []
    loss_list = []
    #train_loader.to(device)
    model = RNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(num_epoch):
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

        epoch_list.append(i)
        loss_list.append(loss_train)

        if i % 10 == 0:
            print("Epoch:", i, "Loss_Train:", loss_train)

        if i % 50 == 0:
            torch.save(model.to('cpu').state_dict(),
                       './rnn_models/epoch_{}_model.pth'.format(i))
            model.to(device)

    torch.save(model.to('cpu').state_dict(),
               './rnn_models/last_epoch_{}_model.pth'.format(num_epoch))

    return model, epoch_list, loss_list


def predict(model, pre):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre.to(device)
    result_x = []
    result_y = []

    model = model.to(device)
    #print(pre[0][0].unsqueeze(0).unsqueeze(0))
    result = model(pre[0][0].unsqueeze(0).unsqueeze(0))#最初だけモデルに入力
    #print(result[0][0])
    result_x.append(result[0][0][0])
    result_y.append(result[0][0][1])
    result = result.to(device)

    count = 0
    while True:
        model = model.to(cpu)
        result = result.to(device)

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
    plt.axes().set_aspect('equal', 'datalim')
    fig.savefig("./rnn_pictures/circle_RNN.png")

def main():
    os.makedirs("./rnn_pictures", exist_ok=True)
    os.makedirs("./rnn_models", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data, train_data = dataload()
    model, epoch_list, loss_list = train(train_data, device, 30)
    #print(input_data[0][0][0])
    predict(model, input_data)

    fig = plt.figure()
    plt.plot(epoch_list, loss_list)
    #plt.axes().set_aspect('equal', 'datalim')
    fig.savefig("./rnn_pictures/loss_RNN.png")


if __name__ == '__main__':
    main()

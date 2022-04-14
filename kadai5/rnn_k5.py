import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def dataload():
    t = np.linspace(0, 2*(math.pi))
    x_list = []
    y_list = []

    for i in range(len(t)):
        x_list.append(math.sin(t[i]+(1/2)*(math.pi)))
        y_list.append(math.sin(2*t[i]))

    correct_x = x_list
    correct_y = y_list

    input_data = []
    correct_data = []

    for i in range(len(correct_x)):
        if i == 49:
            input_data.append([correct_x[i], correct_y[i]])
            correct_data.append([correct_x[0], correct_y[0]])
        else:
            input_data.append([correct_x[i], correct_y[i]])
            correct_data.append([correct_x[i+1], correct_y[i+1]])

    input_data = torch.FloatTensor(input_data)
    correct_data = torch.FloatTensor(correct_data)

    dataset = TensorDataset(input_data, correct_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return input_data, train_loader


class RNN(nn.Module):
    def __init__(self, device):
        super(RNN, self).__init__()
        self.device = device
        self.rnn = nn.RNN(2, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.to(self.device)
        x_rnn, hidden = self.rnn(x, None)
        x = self.fc(x_rnn[:, -1, :])
        return x


def train(train_loader, device, num_epoch):
    loss_list = []
    epoch_list = []

    model = RNN(device).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for i in range(num_epoch):
        model.train()
        loss_train = 0
        for j, xy in enumerate(train_loader):

            xy[0] = xy[0].to(device)
            xy[1] = xy[1].to(device)

            loss = criterion(model(xy[0]), xy[1])
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train /= j+1

        loss_list.append(loss_train)
        epoch_list.append(i)

        if i % 10 == 0 and i != 0:
            print("Epoch:", i, "Loss_Train:", loss_train)

        if i % 100 == 0:
            torch.save(model.to('cpu').state_dict(),
                       './rnn_models/epoch_{}_model.pth'.format(i))

    torch.save(model.to('cpu').state_dict(),
               './rnn_models/last_epoch_{}_model.pth'.format(num_epoch))

    return model, loss_list, epoch_list


def predict(model, pre):

    pre = pre.cpu()
    result_x = []
    result_y = []

    for i in range(50):
        result = model(pre.unsqueeze(1)[i])
        result = result.cpu()
        result_x.append(result[0][0])
        result_y.append(result[0][1])

    fig = plt.figure()
    plt.plot(result_x, result_y, linestyle="None", linewidth=0, marker='o')
    fig.savefig("./rnn_pictures/rnn_Lissajous.png")


def main():
    os.makedirs("./rnn_models", exist_ok=True)
    os.makedirs("./rnn_pictures", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epoch = 600

    input_data, train_data = dataload()
    model, loss_list, epoch_list = train(train_data, device, num_epoch)
    predict(model, input_data)

    fig = plt.figure()
    plt.plot(epoch_list, loss_list)
    fig.savefig("./rnn_pictures/loss_RNN.png")


if __name__ == '__main__':
    main()

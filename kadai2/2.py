import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch
import numpy as np
import os


class Encoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 64)
        self.fc3 = torch.nn.Linear(64, 512)
        self.fc4 = torch.nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(self, org_size):
        super().__init__()
        self.enc = Encoder(org_size)
        self.dec = Decoder(org_size)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    return npimg


def train(net, criterion, optimizer, epochs, trainloader, input_size):
    losses = []
    output_and_label = []

    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch}, ', end='')
        running_loss = 0.0
        for counter, (img, _) in enumerate(trainloader, 1):
            optimizer.zero_grad()
            img = img.reshape(-1, input_size)
            output = net(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / counter
        losses.append(avg_loss)
        print('loss:', avg_loss)
        output_and_label.append((output, img))
    print('finished')

    return output_and_label, losses


def resultshow(output_and_label, losses):

    output, org = output_and_label[-1]
    npimg = imshow(org.reshape(-1, 1, 28, 28))
    plt.imsave("./autoencoder_pictures/original_picture.png",
               np.transpose(npimg, (1, 2, 0)))

    npimg = imshow(output.reshape(-1, 1, 28, 28))
    plt.imsave("./autoencoder_pictures/output_picture.png",
               np.transpose(npimg, (1, 2, 0)))

    plt.imsave("./autoencoder_pictures/loss.png",
               losses)


def main():
    os.makedirs("./autoencoder_pictures", exist_ok=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST('./data', train=True, transform=transform, download=True)

    batch_size = 50
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    input_size = 28 * 28
    net = AutoEncoder(input_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    EPOCHS = 100

    output_and_label, losses = train(
        net, criterion, optimizer, EPOCHS, trainloader, input_size)

    resultshow(output_and_label, losses)


if __name__ == '__main__':
    main()

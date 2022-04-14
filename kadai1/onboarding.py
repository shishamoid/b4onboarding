import torch.nn as nn
from dataloader import Mydatasets
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(
            in_features=4 * 4 * 128, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def dataload():
    train_dataset, val_dataset = \
        torch.utils.data.random_split(Mydatasets(
            ["cifar-10-batches-py/data_batch_1",
             "cifar-10-batches-py/data_batch_2",
             "cifar-10-batches-py/data_batch_3",
             "cifar-10-batches-py/data_batch_4",
             "cifar-10-batches-py/data_batch_5"]), [45000, 5000])

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    validation_dataloader = DataLoader(
        val_dataset, batch_size=256, shuffle=False)
    print("dataload finished")
    return train_dataloader, validation_dataloader


def train(num_epoch, train_dataloader, validation_dataloader):
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    model = CNN(10)  # 10クラス分類
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(num_epoch):
        running_loss = 0.0
        running_acc = 0.0
        for imgs, labels in train_dataloader:

            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)

            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            running_acc += torch.mean(pred.eq(labels).float())
            optimizer.step()
        running_loss /= len(train_dataloader)
        running_acc /= len(train_dataloader)
        losses.append(running_loss)
        accs.append(running_acc)

        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_imgs, val_labels in validation_dataloader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)
            val_output = model(val_imgs)
            val_loss = criterion(val_output, val_labels)
            val_running_loss += val_loss.item()
            val_pred = torch.argmax(val_output, dim=1)
            val_running_acc += torch.mean(val_pred.eq(val_labels).float())
        val_running_loss /= len(validation_dataloader)
        val_running_acc /= len(validation_dataloader)
        val_losses.append(val_running_loss)
        val_accs.append(val_running_acc)

        print("epoch: {}, loss: {}, acc: {}    "
              "val_epoch: {}, val_loss: {}, val_acc: {}"
              .format(epoch, running_loss, running_acc, epoch, val_running_loss, val_running_acc))

        if epoch % 20 == 0:
            torch.save(model.to('cpu').state_dict(),
                       './cifar_models/epoch_{}_model.pth'.format(epoch))

    torch.save(model.to('cpu').state_dict(),
               './cifar_models/last_epoch_{}_model.pth'.format(num_epoch))

    return val_accs


def main():
    os.makedirs("cifar_models", exist_ok=True)
    os.makedirs("cifar_accuracy", exist_ok=True)

    train_dataloader, validation_dataloader = dataload()
    accuracy = train(100, train_dataloader, validation_dataloader)
    fig = plt.figure()
    print(accuracy)
    plt.plot(accuracy)
    fig.savefig("./cifar_accuracy/cifar_accuracy.png")


if __name__ == '__main__':
    main()

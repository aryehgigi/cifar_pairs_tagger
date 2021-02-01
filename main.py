from dataclasses import dataclass
import argparse
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler


exploration = False
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


@dataclass(frozen=True)
class Hyperparams:
    optimizer: str
    batch_size: int
    learning_rate: float
    num_of_epochs: int


optimizers = {
    "SGD": optim.SGD, "Adadelta": optim.Adadelta, "Adagrad": optim.Adagrad, "Adam": optim.Adam, "RMSProp": optim.RMSprop
}


class Net(nn.Module):
    def __init__(self, convs_channel_dims, linear_dims, drop):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, convs_channel_dims[0], 5)
        self.bn1 = nn.BatchNorm2d(convs_channel_dims[0])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(convs_channel_dims[0], convs_channel_dims[1], 3)
        self.bn2 = nn.BatchNorm2d(convs_channel_dims[1])
        self.conv3 = nn.Conv2d(convs_channel_dims[1], convs_channel_dims[2], 3)
        self.bn3 = nn.BatchNorm2d(convs_channel_dims[2])
        self.fc_input_dim = convs_channel_dims[2] * 4 * 12
        self.fc1 = nn.Linear(self.fc_input_dim, linear_dims[0])
        self.fc1_bn = nn.BatchNorm1d(linear_dims[0])
        self.fc2 = nn.Linear(linear_dims[0], linear_dims[1])
        self.fc2_bn = nn.BatchNorm1d(linear_dims[1])
        self.fc3 = nn.Linear(linear_dims[1], 10)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.fc_input_dim)
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)
        return x


def make_pairs(dataloader, seed: int = 0):
    random.seed(seed)  # for train set
    pairs = list()
    labels = list()
    label_to_image = list()
    dataiter = iter(dataloader)
    for data, label in dataiter:
        data = torch.squeeze(data)
        label_to_image.append((label[0], data))

    while len(pairs) < 10000:
        if len(pairs) == 5000:
            # moving to validation set, so lets have it recoverable
            random.seed(0)
        sample1 = random.randint(0, len(label_to_image) - 1)
        sample2 = random.randint(0, len(label_to_image) - 1)
        if label_to_image[sample1][0] != label_to_image[sample2][0]:
            pairs.append(torch.unsqueeze(torch.cat((label_to_image[sample1][1], label_to_image[sample2][1]), 2), 0))
            labels.append(torch.unsqueeze(torch.FloatTensor([1 if i in [label_to_image[sample1][0], label_to_image[sample2][0]] else 0 for i in range(10)]), 0))
            # popping will change the size so fix the index of the second sample
            label_to_image.pop(sample1)
            label_to_image.pop(sample2 if sample2 < sample1 else sample2 - 1)

    return pairs[:5000], labels[:5000], pairs[5000:], labels[5000:]


def prepare_dataset(seed: int, root_path: str):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=root_path if root_path else '.', train=True, transform=transform, download=False if root_path else True)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=False)
    return make_pairs(trainloader, seed=seed)


def load_model(p):
    return torch.load(p)


def save_model(m, i):
    torch.save(m, f'models/model_{i}.pt')


def train_classifier(model, x, y, hypers: Hyperparams):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizers[hypers.optimizer](model.parameters(), lr=hypers.learning_rate)

    for epoch in range(hypers.num_of_epochs):
        total = 0
        for batch_id in range(int(5000 / hypers.batch_size) + (1 if (5000 % hypers.batch_size != 0) else 0)):
            optimizer.zero_grad()
            output = model(torch.cat(x[batch_id * hypers.batch_size:(batch_id + 1) * hypers.batch_size]))
            loss = criterion(output, torch.cat(y[batch_id * hypers.batch_size:(batch_id + 1) * hypers.batch_size]))
            loss.backward()
            optimizer.step()
            total += loss.item()
        print('Loss: {:.3f}'.format(total / (batch_id + 1)))


def evaluate_classifier(model, x, y):
    total = 0.0
    exact = 0.0
    model.eval()
    with torch.no_grad():
        for cur_x, cur_y in zip(x, y):
            output = model(cur_x)
            _, predicted = torch.topk(output, k=2)
            total += torch.squeeze(cur_y)[torch.unsqueeze(predicted, 0)].sum().item()
            exact += 1 if (torch.squeeze(cur_y)[torch.unsqueeze(predicted, 0)].sum().item() == 2) else 0
    return 100 * total / 10000, 100 * exact / 5000


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Deep Learning Challenge")
    argparser.add_argument(
        "-m",
        "--model_to_load",
        default="",
        type=str,
        help="path to a model to load (instead of training one)",
    )
    argparser.add_argument(
        "-c",
        "--convs_channel_dims",
        default=[6, 16, 32],
        nargs=3,
        type=int,
        help="list of dimensions for the channels of the 3 convolutional layers",
    )
    argparser.add_argument(
        "-l",
        "--linear_dims",
        default=[120, 80],
        nargs=2,
        type=int,
        help="list of out dimensions of the 2 fully connected layers",
    )
    argparser.add_argument(
        "-d",
        "--dropout",
        default=0.25,
        type=float,
        help="the value for the dropout layers all over the model",
    )
    argparser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="a seed to randomize the training set differently",
    )
    argparser.add_argument(
        "-o",
        "--optimizer",
        choices=["SGD", "Adadelta", "Adagrad", "Adam", "RMSProp"],
        default="Adam",
        type=str,
        help="which optimizer to use",
    )
    argparser.add_argument(
        "-b",
        "--batch_size",
        default=10,
        type=int,
        help="the batch size to use in training",
    )
    argparser.add_argument(
        "-r",
        "--learning_rate",
        default=0.001,
        type=float,
        help="the learning rate to use in training",
    )
    argparser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="the amount of epochs to use in training",
    )
    argparser.add_argument(
        "-p",
        "--root_path",
        default='',
        type=str,
        help="path to root dir where cifar-10-batches-py folder is found with the data_batch files, if not given will be downloaded",
    )

    args = argparser.parse_args()

    train, train_labels, validation, validation_labels = prepare_dataset(args.seed, args.root_path)
    if args.model_to_load:
        m = load_model(args.model_to_load)
        t, e = evaluate_classifier(m, validation, validation_labels)
        print(f'Multi label Precision: {t:.3f}, exact accuracy: {e:.3f}')
    elif not exploration:
        m = Net(args.convs_channel_dims, args.linear_dims, args.dropout)
        train_classifier(m, train, train_labels, Hyperparams(args.optimizer, args.batch_size, args.learning_rate, args.epochs))
        t, e = evaluate_classifier(m, validation, validation_labels)
        print(f'Multi label Precision: {t:.3f}, exact accuracy: {e:.3f}, params:{args.optimizer, args.batch_size, args.learning_rate, args.epochs}')
    else:
        prev_t = 0
        count = 0
        for opt in ["RMSProp", "Adam", "SGD"]:
            for lr in [0.1, 0.05, 0.01]:
                for bs in [5, 10, 25, 100]:
                    for ep in [10, 25, 50, 100]:
                        start = time.time()
                        m = Net([6, 16, 32], [120, 80], 0.25)
                        train_classifier(m, train, train_labels, Hyperparams(opt, bs, lr, ep))
                        t, e = evaluate_classifier(m, validation, validation_labels)
                        print(f'Multi label Precision: {t:.3f}, exact accuracy: {e:.3f}, params:{opt, bs, lr, ep}')
                        if t > prev_t:
                            save_model(m, count)
                            prev_t = t
                        print(f"time: {time.time() - start}")
                        count += 1

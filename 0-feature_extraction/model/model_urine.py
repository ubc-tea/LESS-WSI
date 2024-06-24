# model for CIFAR 10

import torch.nn as nn
import torch.nn.functional as F


class NetworkPhi(nn.Module):
    def __init__(self):
        super(NetworkPhi, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=2)
        self.conv3 = nn.Conv2d(96, 192, 1)
        self.conv4 = nn.Conv2d(192, 10, 1)
        self.fc1 = nn.Linear(1960, 1000)
        self.fc2 = nn.Linear(1000, 128)
        self.fc3 = nn.Linear(128, 2)
        self.LogSoftMax = nn.Softmax(dim=1)
        self.af = F.relu

    def forward(self, x):
        print('x size',x.size())
        h = self.conv1(x)
        print('1 layer',h.size())
        h = self.af(h)
        h = self.conv2(h)
        print('2 layer', h.size())
        h = self.af(h)
        h = self.conv3(h)
        print('3 layer', h.size())
        h = self.af(h)
        h = self.conv4(h)
        print('4 layer', h.size())
        h = self.af(h)
        h = h.view(-1, 1960)
        h = self.fc1(h)
        print('5 layer', h.size())
        h = self.af(h)
        h = self.fc2(h)
        # print('6 layer', h.size())
        h = self.af(h)
        last_h = self.fc3(h)
        # print(last_h.size())
        return self.LogSoftMax(last_h), h

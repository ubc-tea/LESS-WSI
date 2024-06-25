# model for CIFAR 10

import torch.nn as nn
import torch.nn.functional as F

# 256
class NetworkPhi_scale256(nn.Module):
    def __init__(self):
        super(NetworkPhi_scale256, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3) # 64,64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32,32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 16，16
        self.conv4 = nn.Conv2d(128, 10, kernel_size=1) #
        self.fc1 = nn.Linear(2560, 1000)
        self.fc2 = nn.Linear(1000, 768)
        self.fc3 = nn.Linear(768, 2)
        self.LogSoftMax = nn.LogSoftmax(dim=1)
        self.af = F.relu

    def forward(self, x):
        h = self.conv1(x)
        # print('1 layer',h.size())
        h = self.af(h)
        h = self.conv2(h)
        h = self.af(h)
        # print('2 layer', h.size())
        # h = self.pool(h)
        h = self.conv3(h)
        # print('3 layer', h.size())
        h = self.af(h)
        h = self.conv4(h)
        # print('4 layer', h.size())
        h = self.af(h)
        h = h.view(-1, 2560)
        # h = h.view(-1,32)
        h = self.fc1(h)

        # print('5 layer', h.size())
        h = self.af(h)
        h = self.fc2(h)
        embedding = h
        last_h = self.fc3(h)
        # print(last_h.size())
        return self.LogSoftMax(last_h), embedding
#128
class NetworkPhi_scale128(nn.Module):
    def __init__(self):
        super(NetworkPhi_scale128, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3) # 64,64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32,32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 16，16
        self.conv4 = nn.Conv2d(128, 10, kernel_size=1) #
        self.fc1 = nn.Linear(640, 1000)
        self.fc2 = nn.Linear(1000, 384)
        self.fc3 = nn.Linear(384, 2)
        self.LogSoftMax = nn.LogSoftmax(dim=1)
        self.af = F.relu

    def forward(self, x):
        print(x.size())
        h = self.conv1(x)
        print('1 layer',h.size())
        h = self.af(h)
        h = self.conv2(h)
        h = self.af(h)
        print('2 layer', h.size())
        # h = self.pool(h)
        h = self.conv3(h)
        print('3 layer', h.size())
        h = self.af(h)
        h = self.conv4(h)
        print('4 layer', h.size())
        h = self.af(h)
        print(h.size())
        h = h.view(-1, 640)
        # h = h.view(-1,32)
        h = self.fc1(h)
        print('5 layer', h.size())
        h = self.af(h)
        h = self.fc2(h)
        embedding = h
        print('embedding',embedding.size())
        last_h = self.fc3(h)
        print(last_h.size())
        return self.LogSoftMax(last_h), embedding


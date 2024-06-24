import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hidden1 = nn.Linear(in_features=19200, out_features=2, bias=True)
        # self.hidden2 = nn.Linear(9600, 2400 )
        # self.hidden3 = nn.Linear(2400, 600)
        # self.predict = nn.Linear(600,2)

    def forward(self, x):
        # x = x.transpose(1,2)
        x = self.pooling(x)
        print(x.size())

        x = x.view(x.size(0), -1)

        print(x.size())
        output = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # output = self.predict(x)

        # out = output.view(-1)

        return output

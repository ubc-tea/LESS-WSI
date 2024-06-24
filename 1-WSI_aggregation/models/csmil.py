import torch.nn as nn
import torch

class DeepAttnMIL_Surv(nn.Module):
    """
    Deep AttnMISL Model definition
    """
    def __init__(self, cluster_num=1):
        super(DeepAttnMIL_Surv, self).__init__()
        # self.embedding_net1 = nn.Linear(768,s64)
        # self.embedding_net2 = nn.Linear(384,64)
        self.embedding_net1 = nn.Sequential(nn.Conv2d(384,64, 1),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1,1))
                                     )
        self.embedding_net2 = nn.Sequential(nn.Conv2d(768, 64,1),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1,1))
                                     )
        # self.embedding_net = nn.Sequential(nn.Conv2d(2048, 64, 1),
        #                              nn.ReLU(),
        #                              nn.AdaptiveAvgPool2d((1,1))
        #                              )

        self.res_attention = nn.Sequential(
            nn.Conv2d(64, 32, 1),  # V
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

        self.attention = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )

        self.fc6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.cluster_num = cluster_num

    def masked_softmax(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / (mask+1e-5))
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)

    # def forward(self, x, mask):
    def forward(self, x1,x2):
        x1 = x1.permute(2, 0, 1)
        x2 = x2.permute(2, 0, 1)
        # x2 = x2.unsqueeze(0)
        print(x1.size())
        print(x2.size())

        " x is a tensor list"
        res = []
        for i in range(self.cluster_num):
            x1 = x1.type(torch.FloatTensor).to("cuda")
            x2 = x2.type(torch.FloatTensor).to("cuda")
            # hh = x[i].type(torch.FloatTensor).to("cuda")
            output1 = self.embedding_net1(x1)
            output2 = self.embedding_net2(x2)
            # output1 = self.embedding_net(hh[:,:,0:1,:])
            # output2 = self.embedding_net(hh[:,:,1:2,:])
            # output3 = self.embedding_net(hh[:,:,2:3,:])
            # output = torch.cat([output1, output2, output3],2)
            print(output1.size())
            output = torch.cat([output1, output2],2)
            res_attention = self.res_attention(output).squeeze(-1)

            final_output = torch.matmul(output.squeeze(-1), torch.transpose(res_attention,2,1)).squeeze(-1)
            res.append(final_output)

        h = torch.cat(res)

        b = h.size(0)
        c = h.size(1)

        h = h.view(b, c)

        A = self.attention(h)
        A = torch.transpose(A, 1, 0)  # KxN

        A = self.masked_softmax(A)

        M = torch.mm(A, h)  # KxL

        Y_pred = self.fc6(M)

        return Y_pred


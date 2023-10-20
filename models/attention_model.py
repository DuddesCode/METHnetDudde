"""
This part of the code was originally published by mahmoodlab and adapted for this project.
https://github.com/mahmoodlab/CLAM

Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images.
Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm1d

import sys

def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout = 0.25, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()

        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]

        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]

        if dropout != 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)

        A = a.mul(b)
        A = self.attention_c(A)

        return A,x

class Attention_SB(nn.Module):
    def __init__(self, settings):
        super(Attention_SB, self).__init__()

        self.F = settings.get_network_setting().get_F()
        self.L = settings.get_network_setting().get_L()
        self.D = settings.get_network_setting().get_D()
        self.dropout = settings.get_network_setting().get_dropout()
        self.n_classes = settings.get_class_setting().get_n_classes()

        fc = [nn.Linear(self.F, self.L), nn.ReLU()]

        if self.dropout != 0:
            fc.append(nn.Dropout(self.dropout))

        attention_net = Attn_Net_Gated(L=self.L, D=self.D, dropout=self.dropout, n_classes=1)
        fc.append(attention_net)

        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(self.L, self.n_classes)

        init_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, label=None):
        device = h.device

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        
        A_raw = A
        A = F.softmax(A, dim=1)

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat, A_raw


class Attention_MB(Attention_SB):
    def __init__(self, settings):
        nn.Module.__init__(self)

        self.F = settings.get_network_setting().get_F()
        self.L = settings.get_network_setting().get_L()
        self.D = settings.get_network_setting().get_D()
        self.dropout = settings.get_network_setting().get_dropout()
        self.n_classes = settings.get_class_setting().get_n_classes()

        fc = [nn.Linear(self.F, self.L), nn.ReLU()]

        if self.dropout != 0:
            fc.append(nn.Dropout(self.dropout))

        attention_net = Attn_Net_Gated(L=self.L, D=self.D, dropout=self.dropout, n_classes=self.n_classes)
        fc.append(attention_net)

        self.attention_net = nn.Sequential(*fc)

        bag_classifiers = [nn.Linear(self.L, 1) for i in range(self.n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        init_weights(self)

    def forward(self, h, label=None):
        device = h.device

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)

        A_raw = A
        A = F.softmax(A, dim=1)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)


        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        print(logits)
        

        return logits, Y_prob, Y_hat, A_raw
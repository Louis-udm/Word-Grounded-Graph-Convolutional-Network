#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Title: MLP models

Description: 
The Multi-layer Perceptron model with 1-hidden-layer and 2-hidden-layer.

"""

# =======================================
# @author Zhibin.Lu
# @email zhibin.lu@umontreal.ca
# =======================================


import math

import torch
import torch.nn as nn
import torch.nn.init as init


class MLP_1h(nn.Module):
    """ 1-hidden-layer MLP

    The same with WGCN (A=I) when w/o bias for first fc layer.
    """

    def __init__(self, input_dim, hid_dim, dropout_rate=0.0, num_classes=10):

        super(MLP_1h, self).__init__()

        self.w1 = nn.Parameter(torch.randn(input_dim, hid_dim))
        # w/o first layer bias
        # self.fc1 = nn.Linear(input_dim, hid_dim)
        # w/ first layer bias
        self.fc2 = nn.Linear(hid_dim, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def forward(self, x):
        # w/o first layer bias
        out = x.mm(self.dropout(self.w1))
        # w/ first layer bias
        # out = self.fc1(self.dropout(x))

        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith("w"):
                # init.xavier_uniform_(p, gain=1.414)
                init.kaiming_uniform_(p, a=math.sqrt(5))


class MLP_2h(nn.Module):
    """2-hidden-layer MLP
    
    """

    def __init__(
        self, input_dim, hid_dim1, hid_dim2, dropout_rate=0.0, num_classes=10
    ):
        super(MLP_2h, self).__init__()

        self.fc1 = nn.Linear(input_dim, hid_dim1)
        self.fc2 = nn.Linear(hid_dim1, hid_dim2)
        self.fc3 = nn.Linear(hid_dim2, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.relu(self.fc1(self.dropout(x)))
        out = self.relu(self.fc2(self.dropout(out)))
        out = self.fc3(self.dropout(out))
        return out

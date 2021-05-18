#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Title: GCN models

Description: 
The original Graph convolutional network model and GCN layer.

Refer to: https://arxiv.org/abs/1609.02907

"""

# =======================================
# @author Zhibin.Lu
# @email zhibin.lu@umontreal.ca
# =======================================


import collections
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GraphConvolutionLayer(nn.Module):
    """Original Graph Convolutional Layer

    Reference GCN equation:
    F = A(relu(AW))W

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        support,
        act_func=None,
        featureless=False,
        dropout_rate=0.0,
        bias=False,
    ):
        super().__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):
            setattr(
                self,
                "W{}".format(i),
                nn.Parameter(torch.randn(input_dim, output_dim)),
            )

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if not self.featureless:
            x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, "W{}".format(i))
            else:
                pre_sup = x.mm(getattr(self, "W{}".format(i)))

            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out


class GraphConvolutionLayer_NoActBtwLayer(nn.Module):
    """ GraphConvolution Layer without the activation
    function between 2 graph convolution layers.

    No-activation-func GCN equation:
    F = (relu(A(AW)))W

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        support,
        act_func=None,
        featureless=False,
        dropout_rate=0.0,
        bias=False,
    ):
        super().__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):
            setattr(
                self,
                "W{}".format(i),
                nn.Parameter(torch.randn(input_dim, output_dim)),
            )

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if not self.featureless:
            x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = self.support[i]
            else:
                pre_sup = self.support[i].mm(x)

            if self.act_func is not None:
                pre_sup = self.act_func(pre_sup)

            if i == 0:
                out = pre_sup.mm(getattr(self, "W{}".format(i)))
            else:
                out += pre_sup.mm(getattr(self, "W{}".format(i)))

        self.embedding = out
        return out


class GCN_2Layers(nn.Module):
    """ The 2-layer GCN

    1. Original GCN model when mode is "only_gcn_act",
    equation is A(relu(AW))W

    2. No act func btw graph layer when mode is "only_fc_act",
    equation is (relu(A(AW)))W

    """

    def __init__(
        self,
        input_dim,
        support,
        hid_dim=200,
        dropout_rate=0.0,
        num_classes=10,
        act_func=None,
        mode="only_gcn_act",
    ):
        super().__init__()

        # GraphConvolution
        if mode == "only_gcn_act":  # original Text_GCN
            # A(relu(AW))W
            self.layer1 = GraphConvolutionLayer(
                input_dim,
                hid_dim,
                support,
                act_func=act_func,
                featureless=True,
                dropout_rate=dropout_rate,
            )
            self.layer2 = GraphConvolutionLayer(
                hid_dim, num_classes, support, dropout_rate=dropout_rate
            )
        elif mode == "only_fc_act":
            # (relu(A(AW)))W
            self.layer1 = GraphConvolutionLayer_NoActBtwLayer(
                input_dim,
                hid_dim,
                support,
                featureless=True,
                dropout_rate=dropout_rate,
            )
            self.layer2 = GraphConvolutionLayer_NoActBtwLayer(
                hid_dim,
                num_classes,
                support,
                act_func=act_func,
                dropout_rate=dropout_rate,
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

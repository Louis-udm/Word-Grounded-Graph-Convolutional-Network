#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Title: WGCN models

Description: 
Word-Grounded Graph Convolutional Network models.
1. WGCN = (ReLu(XAW0))W
2. WGCN_VocabEmbedding = (ReLu(X_dv.A.Emb_Vocab.W0)W
3. WGCN_embedding_classifier = 
     ( ReLu( X_{mev} A_{vv} W_{vh}) W_{hg} + b_g) W_{gc} + b_c

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


class WGCN(nn.Module):
    """Word-Grounded Graph Convolutional Network Model

    Equations:
    1. (relu(XAW0))W   (the same as our paper)
    2. (relu(XW0+XAW0))W
    3. A can be A^{n} or A^{n}X_{train_set}, in general n=1

    parameters
    ----------
    voc_dim : int
        The vocabulary size
    num_adj : int
        The number of the WGraph, there is only 1 WGraph in general.
    hid_dim : int
        The dimension of the hidden graph convolutinal layer.
    out_dim : int
        The output dimension, can be the number of category (number of categories)
    act_func : function
        The non-linear activation function.
    dropout_rate : float
        dropout rate, in [0,1)

    """

    def __init__(
        self,
        voc_dim,
        num_adj,
        hid_dim,
        out_dim,
        act_func=nn.ReLU(),
        dropout_rate=0.0,
    ):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        # Wx_vh
        for i in range(self.num_adj):
            setattr(
                self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
            )

        # self.layerNorm=nn.LayerNorm(voc_dim, eps=1e-12)
        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith("W"):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        """forward function

        parameters
        ----------
        vocab_adj_list : list
            Every item is a WGraph adjacent matrix (or called vocabulary graph)
            In general there is only one WGraph.
        X_dv : Tensor
            The input X support word embedding.
            The dimension shape is (batch, emb_dim, vocab) or (batch, vocab)
        add_linear_mapping_term : bool
            True to add the linear mapping term. (refer to WGCN paper)

        returns
        -------
        out : Float vector
            the class size output vector.
        
        """

        # X_dv=self.dropout(X_dv)
        for i in range(self.num_adj):
            # H_vh=getattr(self, 'W%d_vh'%i) # MLP mode, A=I
            H_vh = vocab_adj_list[i].mm(getattr(self, "W%d_vh" % i))

            # Three options
            # H_vh=self.dropout(F.elu(H_vh))
            # H_vh=self.dropout(self.act_func(H_vh))
            H_vh = self.dropout(H_vh)

            H_dh = X_dv.mm(H_vh)

            if add_linear_mapping_term:
                H_linear = X_dv.mm(getattr(self, "W%d_vh" % i))
                H_linear = self.dropout(H_linear)
                # H_linear=self.dropout(self.act_func(H_linear))
                H_dh += H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        fused_H = self.act_func(fused_H)
        fused_H = self.dropout(fused_H)

        out = self.fc_hc(fused_H)
        return out


class WGCN_VocabEmbedding(nn.Module):
    """WGCN with a word embedding vocabulary

    Equation:
    (relu(X_dv.A.Emb_Vocab.W0)W

    parameters
    ----------
    voc_dim : int
        The vocabulary size
    num_adj : int
        The number of the WGraph, there is only 1 WGraph in general.
    voc_X_dim : int
        The word embedding dimension in vocabulary.
    hid_dim : int
        The dimension of the hidden graph convolutinal layer.
    out_dim : int
        The output dimension, can be the number of category (number of categories)
    act_func : function
        The non-linear activation function.
    dropout_rate : float
        dropout rate, in [0,1)

    """

    def __init__(
        self,
        voc_dim,
        num_adj,
        voc_X_dim,
        hid_dim,
        out_dim,
        act_func=nn.ReLU(),
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_adj = num_adj
        self.voc_X_dim = voc_X_dim

        # Wx_vh
        for i in range(self.num_adj):
            setattr(
                self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
            )

        # self.Wx_vh=nn.Parameter(torch.randn(voc_dim, hid_dim))
        for i in range(self.num_adj):
            setattr(
                self,
                "W%d_eh" % i,
                nn.Parameter(torch.randn(voc_X_dim, hid_dim)),
            )

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith("W"):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(
        self, vocab_adj_list, X_dv, vocab_Xve, add_linear_mapping_term=False
    ):
        """forward function

        parameters
        ----------
        vocab_adj_list : list
            Every item is a WGraph adjacent matrix (or called vocabulary graph)
            In general there is only one WGraph.
        X_dv : Tensor
            The input X support word embedding.
            The dimension shape is (batch, emb_dim, vocab) or (batch, vocab)
        vocab_Xve : Matrix
            The vocabulary (WGraph) based on word embedding, 
            all words are word embedding vector.
        add_linear_mapping_term : bool
            True to add the linear mapping term. (refer to WGCN paper)

        returns
        -------
        out : Float vector
            the class size output vector.
        
        """

        vocab_Xve = self.dropout(vocab_Xve)
        for i in range(self.num_adj):
            HX_vh = vocab_Xve.mm(getattr(self, "W%d_eh" % i))
            # HX_vh=self.dropout(F.elu(HX_vh))
            H_vh = vocab_adj_list[i].mm(HX_vh)
            # H_vh=self.dropout(self.act_func(H_vh))
            H_vh = self.dropout(H_vh)
            H_dh = X_dv.mm(H_vh)
            # H_dh=torch.addmm(getattr(self, 'b%d_vh'%i), X_dv,H_vh)
            # H_dh=self.dropout(self.act_func(H_dh))

            if add_linear_mapping_term:
                H_linear = X_dv.mm(getattr(self, "W%d_vh" % i))
                # H_linear=self.dropout(F.elu(H_linear))
                H_linear = self.dropout(H_linear)
                # H_linear=self.dropout(self.act_func(H_linear))
                H_dh += H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        fused_H = self.dropout(self.act_func(fused_H))

        out = self.fc_hc(fused_H)
        return out


class WGCN_embedding_classifier(nn.Module):
    """WGCN classifier based on word embedding
    Equation: 
    ( ReLu( X_{mev} A_{vv} W_{vh}) W_{hg} + b_g) W_{gc} + b_c
    x: word embedding for one sample, extend to vocabulary size, 
    m: batch_size, e: word embedding dimension, v: vocabulary size
    A: WGraph adjacent matrix; g: WGCN's output, graph embedding number.
    c: categories number, h: hidden WGraph layer dimension.

    parameters
    ----------
    word_emb : torch.nn.Embedding
        The pre-trained word embedding model
    word_emb_dim : int
        The word embedding dimension
    gcn_adj_dim : int
        The WGCN graph adjacent matrix dimension
    gcn_adj_num : int
        The number of the WGraph, there is only 1 WGraph in general.
    gcn_embedding_dim : int
        The number of graph embeddings. It equals to WGCN's out_dim.
    num_labels : int
        The number of category (number of categories)
    act_func : function
        The non-linear activation function.
    dropout_rate : float
        dropout rate, in [0,1)

    """

    def __init__(
        self,
        word_emb,
        word_emb_dim,
        gcn_adj_dim,
        gcn_adj_num,
        gcn_embedding_dim,
        num_labels,
        act_func=nn.ReLU(),
        dropout_rate=0.2,
    ):
        super().__init__()
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)
        self.word_emb = word_emb
        self.wgcn = WGCN(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim)
        self.classifier = nn.Linear(
            gcn_embedding_dim * word_emb_dim, num_labels
        )

    def forward(self, vocab_adj_list, gcn_swap_eye, input_ids):
        """forward function

        parameters
        ----------
        vocab_adj_list : list
            Every item is a WGraph adjacent matrix (or called vocabulary graph)
            In general there is only one WGraph.
        gcn_swap_eye : Matrix
            The transform matrix for transform the token sequence (sentence) 
            to the Vocabulary order (BoW order)
        input_ids : Matrix
            A torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)

        returns
        -------
        out : Float vector
            the class size output vector.
        
        """
        words_embeddings = self.word_emb(input_ids)
        vocab_input = gcn_swap_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.wgcn(vocab_adj_list, vocab_input).transpose(1, 2)
        gcn_vocab_out = self.dropout(self.act_func(gcn_vocab_out))
        out = self.classifier(gcn_vocab_out.flatten(start_dim=1))
        return out

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Title: train MLP models

Description: 

"""

# =======================================
# @author Zhibin.Lu
# @email zhibin.lu@umontreal.ca
# =======================================


from __future__ import division, print_function

import argparse
import os
import pickle as pkl

# Set random seed
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

from models import MLP_1h, MLP_2h
from utils.utils import *

cuda_yes = torch.cuda.is_available()
# cuda_yes = False
# print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="mr")
parser.add_argument("--model", type=str, default="MLP_1h")  # MLP_1h, MLP_2h
args = parser.parse_args()
cfg_ds = args.ds
cfg_model = args.model

if cfg_model not in ["MLP_1h", "MLP_2h"]:
    sys.exit("wrong model name")

datasets = ["20ng", "R8", "R52", "ohsumed", "mr"]
if cfg_ds not in datasets:
    sys.exit("wrong dataset name")

cfg_data_dir = "data"
cfg_learning_rate = 0.002
cfg_early_stopping = 10
cfg_hidden_dim = 250
cfg_act_func = nn.ReLU()
cfg_dropout = 0.6
cfg_weight_decay = 0.0
cfg_epochs = 400
cfg_add_linear_mapping_term = False


random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
if cuda_yes:
    torch.cuda.manual_seed_all(44)

print("\nStart at:", time.asctime(), "Machine:", os.uname()[1])


def sparse_scipy2torch(coo_sparse):
    # coo_sparse=coo_sparse.tocoo()
    i = torch.LongTensor(np.vstack((coo_sparse.row, coo_sparse.col)))
    # make sure the original type is np.float32
    v = torch.from_numpy(coo_sparse.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


# Load data
(
    _,
    _,
    y_train,
    y_val,
    y_test,
    train_mask,
    val_mask,
    test_mask,
    train_size,
    test_size,
    vocab,
    _,
    _,
    _,
    tfidf_list,
) = load_corpus(cfg_ds, cfg_data_dir)
y_train = y_train[train_mask, :]
y_val = y_val[val_mask, :]
y_test = y_test[test_mask, :]

real_train_x, valid_x, test_x = tuple(tfidf_list)

num_classes = y_train.shape[1]
vocab_size = len(vocab)

print(
    "Dataset:",
    cfg_ds,
    "train size:",
    real_train_x.shape[0],
    "vocab_size:",
    vocab_size,
)

# features = sp.identity(features.shape[0])  # featureless
# use sparse:
t_X_train = sparse_scipy2torch(real_train_x.tocoo()).to(device)
t_X_valid = sparse_scipy2torch(valid_x.tocoo()).to(device)
t_X_test = sparse_scipy2torch(test_x.tocoo()).to(device)
# use dense:
# t_X_train=torch.from_numpy(real_train_x.A).to(device)
# t_X_valid=torch.from_numpy(valid_x.A).to(device)
# t_X_test=torch.from_numpy(test_x.A).to(device)

# Define placeholders
# t_features = torch.from_numpy(features)
t_train_y = torch.from_numpy(y_train).to(device)
t_valid_y = torch.from_numpy(y_val).to(device)
t_test_y = torch.from_numpy(y_test).to(device)

if cfg_model == "MLP_1h":
    model = MLP_1h(vocab_size, cfg_hidden_dim, cfg_dropout, num_classes)
elif cfg_model == "MLP_2h":
    model = MLP_2h(vocab_size, 512, 100, cfg_dropout, num_classes)
else:
    raise ValueError("Invalid argument for model: " + str(cfg_model))

model = model.to(device)

# Loss and optimizer
# criterion = nn.CrossEntropyLoss(weight=loss_weight)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=cfg_learning_rate, weight_decay=cfg_weight_decay
)

# Define model evaluation function
# def evaluate(features, labels, mask):
def evaluate(X, y, segment="valid"):
    t_test = time.time()
    y = torch.max(y, 1)[1]
    model.eval()
    with torch.no_grad():
        # logits = model(features)
        if cfg_model.startswith("MLP"):
            logits = model(X)
        else:
            logits = model(
                t_vocab_adj_list,
                X,
                add_linear_mapping_term=cfg_add_linear_mapping_term,
            )
        loss = criterion(logits, y)
        pred = torch.max(logits, 1)[1]
        acc = pred.eq(y).sum().item() / len(y)

    return loss.cpu().numpy(), acc, pred.cpu().numpy(), (time.time() - t_test)


val_losses = []

# Train model
for epoch in range(cfg_epochs):

    t = time.time()
    model.train()

    # Forward pass
    # logits = model(t_features)
    if cfg_model in ("MLP_1h", "MLP_2h"):
        logits = model(t_X_train.to_dense())
    else:
        logits = model(
            t_vocab_adj_list,
            t_X_train,
            add_linear_mapping_term=cfg_add_linear_mapping_term,
        )
    loss = criterion(logits, torch.max(t_train_y, 1)[1])
    pred = torch.max(logits, 1)[1]
    acc = pred.eq(torch.max(t_train_y, 1)[1]).sum().item() / len(t_train_y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    # val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
    val_loss, val_acc, val_pred, duration = evaluate(
        t_X_valid, t_valid_y, segment="valid"
    )
    val_losses.append(val_loss)

    # Testing
    # test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
    test_loss, test_acc, _, _ = evaluate(t_X_test, t_test_y, segment="test")

    print_log(
        "Epoch:{:.0f}, train_loss={:.5f}, train_acc={:.5f}, v_loss={:.5f}, v_acc={:.5f}, t_loss={:.5f}, t_acc={:.5f}, time= {:.5f}".format(
            epoch + 1,
            loss,
            acc,
            val_loss,
            val_acc,
            test_loss,
            test_acc,
            time.time() - t,
        )
    )

    if epoch > cfg_early_stopping and val_losses[-1] > np.mean(
        val_losses[-(cfg_early_stopping + 1) : -1]
    ):
        print_log("Early stopping...")
        break


print_log("Optimization Finished!")

# Testing
# test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
test_loss, test_acc, test_pred, test_duration = evaluate(
    t_X_test, t_test_y, segment="test"
)

print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(y_test.argmax(1), test_pred, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(
    metrics.precision_recall_fscore_support(
        y_test.argmax(1), test_pred, average="macro"
    )
)
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(
    metrics.precision_recall_fscore_support(
        y_test.argmax(1), test_pred, average="micro"
    )
)

print_log("Weighted F1-Score...")
f1_score = metrics.f1_score(y_test.argmax(1), test_pred, average="weighted")
print_log(
    "Test set results: \n\t epoch= {:d}, accuracy= {:.5f}, loss= {:.5f}, f1= {:.5f}, time= {:.5f}".format(
        epoch + 1, test_acc, test_loss, f1_score, test_duration
    )
)

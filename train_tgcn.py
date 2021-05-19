#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Title: Train Traditional GCN / Text-GCN models

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

from models import GCN_2Layers
from utils.utils import *

cuda_yes = torch.cuda.is_available()
cuda_yes = False
# print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="mr")
parser.add_argument("--model", type=str, default="GCN")  # MLP_1h, MLP_2h
parser.add_argument(
    "--mode", type=str, default="only_gcn_act"
)  # MLP_1h, MLP_2h
parser.add_argument("--adj", type=str, default="pmi")  # all,pmi,npmi,tf
# parser.add_argument('--threshold_k', type=int, default='-1') #
args = parser.parse_args()
cfg_ds = args.ds
cfg_vocab_adj = args.adj
cfg_model = "WGCN"

datasets = ["20ng", "R8", "R52", "ohsumed", "mr"]
if cfg_ds not in datasets:
    sys.exit("wrong dataset name")

cfg_data_dir = "data"

cfg_model = args.model
# cfg_mode = "only_gcn_act"  # original Text GCN: A(relu(AW))W
# cfg_mode='only_fc_act' #(relu(A(AW)))W
cfg_mode = args.mode

cfg_learning_rate = 0.02  #
# if cfg_dataset=='mr':
#     cfg_learning_rate = 0.006 # 2 adj with xw, X is tfidf no norm


cfg_hidden_dim = 200  # Number of units in hidden layer 1.
cfg_dropout = 0.5  # Dropout rate (1 - keep probability).
# cfg_dropout = 0.  # Dropout rate (1 - keep probability).
cfg_weight_decay = 0.0  # Weight for L2 loss on embedding matrix.
cfg_epochs = 400  # Number of epochs to train.
cfg_early_stopping = 10  # Tolerance for early stopping (# of epochs).
cfg_max_degree = 3  # Maximum Chebyshev polynomial degree.
cfg_act_func = nn.ReLU()

print("\nStart at:", time.asctime())
print("---Config---\nModel:", cfg_model, "Data set:", cfg_ds)
print(
    "Total epochs:",
    cfg_epochs,
    "LR:",
    cfg_learning_rate,
    "hidden_dim:",
    cfg_hidden_dim,
)
print(
    "Dropout:", cfg_dropout, "L2:", cfg_weight_decay, "act_func:", cfg_act_func
)


random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
if cuda_yes:
    torch.cuda.manual_seed_all(44)

# Load data
(
    adj,
    features,
    y_train,
    y_val,
    y_test,
    train_mask,
    val_mask,
    test_mask,
    train_size,
    test_size,
    _,
    _,
    _,
    _,
    _,
) = load_corpus(cfg_ds, cfg_data_dir)

features = sp.identity(features.shape[0], dtype=np.float32)  # featureless


def get_class_count_and_weight(y, n_classes):
    classes_count = []
    weight = []
    for i in range(n_classes):
        count = np.sum(y == i)
        classes_count.append(count)
        weight.append(len(y) / (n_classes * count))
    return classes_count, weight


num_classes = y_train.shape[1]
train_classes_num, train_classes_weight = get_class_count_and_weight(
    np.argmax(y_train, 1), num_classes
)
loss_weight = torch.tensor(train_classes_weight).to(device)


# Some preprocessing
features = preprocess_features(features)
# original gcn: mode=only_gcn_act; No act func: mode=only_fc_act
if cfg_model == "GCN":
    support = [normalize_adj(adj)]
    num_supports = 1
    model_func = GCN_2Layers
elif cfg_model == "GCN_cheby":
    support = chebyshev_polynomials(adj, cfg_max_degree)
    num_supports = 1 + cfg_max_degree
    model_func = GCN_2Layers
else:
    raise ValueError("Invalid argument for model: " + str(cfg_model))

# Define placeholders
t_features = torch.from_numpy(features).to(device)
t_y_train = torch.from_numpy(y_train).to(device)
t_y_val = torch.from_numpy(y_val).to(device)
t_y_test = torch.from_numpy(y_test).to(device)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32)).to(device)
# t_val_mask = torch.from_numpy(val_mask.astype(np.float32)).to(device)
# t_test_mask = torch.from_numpy(test_mask.astype(np.float32)).to(device)
tm_train_mask = (
    torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0)
    .repeat(1, y_train.shape[1])
    .to(device)
)


def sparse_scipy2torch(coo_sparse):
    # coo_sparse=coo_sparse.tocoo()
    i = torch.LongTensor(np.vstack((coo_sparse.row, coo_sparse.col)))
    # make sure the original type is np.float32
    v = torch.from_numpy(coo_sparse.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


t_support = []
for i in range(len(support)):
    # t_support.append(torch.Tensor(support[i].A).to(device))
    t_support.append(sparse_scipy2torch(support[i].tocoo()).to(device))


# support is adj
model = model_func(
    input_dim=features.shape[0],
    support=t_support,
    hid_dim=cfg_hidden_dim,
    dropout_rate=cfg_dropout,
    num_classes=num_classes,
    act_func=cfg_act_func,
    mode=cfg_mode,
)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=loss_weight)
optimizer = torch.optim.Adam(
    model.parameters(), lr=cfg_learning_rate, weight_decay=cfg_weight_decay
)


# Define model evaluation function
def evaluate(features, labels, mask):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        t_mask = torch.from_numpy(np.array(mask * 1.0, dtype=np.float32)).to(
            device
        )
        tm_mask = (
            torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0)
            .repeat(1, labels.shape[1])
            .to(device)
        )
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = (
            (pred == torch.max(labels, 1)[1]).float() * t_mask
        ).sum().item() / t_mask.sum().item()

    return (
        loss.cpu().numpy(),
        acc,
        pred.cpu().numpy(),
        labels.cpu().numpy(),
        (time.time() - t_test),
    )


val_losses = []
# Train model
for epoch in range(cfg_epochs):

    t = time.time()
    model.train()

    # Forward pass
    logits = model(t_features)
    loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
    acc = (
        (torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float()
        * t_train_mask
    ).sum().item() / t_train_mask.sum().item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(
        t_features, t_y_val, val_mask
    )
    val_losses.append(val_loss)

    print_log(
        "Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}".format(
            epoch + 1, loss, acc, val_loss, val_acc, time.time() - t
        )
    )

    if epoch > cfg_early_stopping and val_losses[-1] > np.mean(
        val_losses[-(cfg_early_stopping + 1) : -1]
    ):
        print_log("Early stopping...")
        break


print_log("Optimization Finished!")


# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(
    t_features, t_y_test, test_mask
)

test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(np.argmax(labels[i]))

print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(test_labels, test_pred, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(
    metrics.precision_recall_fscore_support(
        test_labels, test_pred, average="macro"
    )
)
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(
    metrics.precision_recall_fscore_support(
        test_labels, test_pred, average="micro"
    )
)
print_log("Weighted F1-Score...")
f1_score = metrics.f1_score(test_labels, test_pred, average="weighted")
print_log(
    "Test set results: \n\t epoch= {:d}, accuracy= {:.5f}, loss= {:.5f}, f1= {:.5f}, time= {:.5f}".format(
        epoch + 1, test_acc, test_loss, f1_score, test_duration
    )
)

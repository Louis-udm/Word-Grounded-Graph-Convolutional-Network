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

from models import WGCN
from utils.utils import *

cuda_yes = torch.cuda.is_available()
# cuda_yes = False
# print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="mr")
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

cfg_adj_tf_threshold = 0.0
cfg_adj_pmi_threshold = 0.0
cfg_adj_npmi_threshold = 0.2
# cfg_adj_npmi_threshold=0.98999

# cfg_learning_rate = 0.02
# 1 adj pmi no xw
cfg_learning_rate = 0.018  #
if cfg_model.startswith("MLP"):
    cfg_learning_rate = 0.002  # 2hidden layer mlp
cfg_early_stopping = 10
# if cfg_dataset=='mr':
# cfg_learning_rate = 0.006 # 2 adj with xw, X is tfidf no norm

cfg_hidden_dim = 250

cfg_act_func = nn.ReLU()
cfg_dropout = 0.6
cfg_weight_decay = 0.0
# cfg_weight_decay = 5e-5
cfg_epochs = 400
cfg_add_linear_mapping_term = False

cfg_normlize_x_mode = "normalize_features"

print("\n Start at:", time.asctime(), "Machine:", os.uname()[1])
print(
    "---Config---\nModel:",
    cfg_model,
    "vocab_adj:",
    cfg_vocab_adj,
    "Data set:",
    cfg_ds,
    "data_dir:",
    cfg_data_dir,
)
print(
    "Total epochs:",
    cfg_epochs,
    "LR:",
    cfg_learning_rate,
    "hidden_dim:",
    cfg_hidden_dim,
)
print(
    "adj_tf_threshold:",
    cfg_adj_tf_threshold,
    "adj_pmi_threshold:",
    cfg_adj_pmi_threshold,
    "cfg_adj_npmi_threshold:",
    cfg_adj_npmi_threshold,
)
print(
    "Dropout:",
    cfg_dropout,
    "L2:",
    cfg_weight_decay,
    "act_func:",
    cfg_act_func,
    "normlize_x_mode:",
    cfg_normlize_x_mode,
)
print(
    "add_linear_mapping_term:",
    cfg_add_linear_mapping_term,
    "early_stopping:",
    cfg_early_stopping,
)


random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
if cuda_yes:
    torch.cuda.manual_seed_all(44)


def normalize_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_sp_features(sp_features):
    sp = sp_features.tolil()
    for i in range(sp.shape[0]):
        sp.data[i] /= np.linalg.norm(sp.data[i])
    return sp.tocsr()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D-degree matrix
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def t_normalize_adj(t_adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = t_adj.sum(1)  # D-degree matrix
    d_inv_sqrt = 1 / torch.sqrt(rowsum)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # D^0.5 .A. D^0.5=A对称，D对角=(A. D^0.5).T.(D^0.5)
    return t_adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)


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
    vocab_adj_tf,
    vocab_adj_pmi,
    vocab_adj_npmi,
    tfidf_list,
) = load_corpus(cfg_ds, cfg_data_dir)
y_train = y_train[train_mask, :]
y_val = y_val[val_mask, :]
y_test = y_test[test_mask, :]


real_train_x, valid_x, test_x = tuple(tfidf_list)
if cfg_normlize_x_mode == "normalize_sp_features":
    real_train_x = normalize_sp_features(real_train_x)
    valid_x = normalize_sp_features(valid_x)
    test_x = normalize_sp_features(test_x)
else:
    real_train_x = normalize_features(real_train_x)
    valid_x = normalize_features(valid_x)
    test_x = normalize_features(test_x)

num_classes = y_train.shape[1]
vocab_size = vocab_adj_pmi.shape[0]

if cfg_adj_tf_threshold > 0:
    vocab_adj_tf.data *= vocab_adj_tf.data > cfg_adj_tf_threshold
    vocab_adj_tf.eliminate_zeros()
if cfg_adj_pmi_threshold > -100:
    vocab_adj_pmi.data *= vocab_adj_pmi.data > cfg_adj_pmi_threshold
    vocab_adj_pmi.eliminate_zeros()
if cfg_adj_npmi_threshold > -1:
    vocab_adj_npmi.data *= vocab_adj_npmi.data > cfg_adj_npmi_threshold
    vocab_adj_npmi.eliminate_zeros()

if cfg_vocab_adj == "pmi":
    vocab_adj_list = [vocab_adj_pmi]
elif cfg_vocab_adj == "npmi":
    vocab_adj_list = [vocab_adj_npmi]
elif cfg_vocab_adj == "tf":
    vocab_adj_list = [vocab_adj_tf]
elif cfg_vocab_adj == "all":
    vocab_adj_list = [vocab_adj_tf, vocab_adj_pmi]

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
# t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
# tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

t_vocab_adj_list = []
for i in range(len(vocab_adj_list)):
    adj = vocab_adj_list[i]  # .tocsr() #(lr是用非norm时的1/10)
    print(
        "Zero ratio(?>66%%) for vocab adj %dth: %.8f"
        % (i, 100 * (1 - adj.count_nonzero() / (adj.shape[0] * adj.shape[1])))
    )
    adj = normalize_adj(adj)
    t_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()).to(device))

model = WGCN(
    vocab_size,
    len(t_vocab_adj_list),
    cfg_hidden_dim,
    num_classes,
    cfg_act_func,
    cfg_dropout,
)
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
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
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

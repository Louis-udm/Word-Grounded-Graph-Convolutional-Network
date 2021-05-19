#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Title: Build Graphs

Description: Input the dataset, 
build graphs and train/dev/test set, and dump to files.

"""

# =======================================
# @author Zhibin.Lu
# @email zhibin.lu@umontreal.ca
# =======================================


import argparse
import os
import pickle as pkl
import random
import sys
import time
from math import log

import nltk
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.utils import clean_str, loadWord2Vec

################################
#     Configuration
################################

random.seed(44)
np.random.seed(44)

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="mr")
parser.add_argument("--window", type=int, default=20)
args = parser.parse_args()
cfg_ds = args.ds

if cfg_ds == "mr":
    cfg_stop_word = False
    cfg_del_infreqent_word = 0
else:
    cfg_stop_word = True
    cfg_del_infreqent_word = 5  # this parameter must >=0

# word co-occurence with context windows
# window_size = 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100
cfg_window_size = args.window

# build corpus
datasets = ["20ng", "R8", "R52", "ohsumed", "mr"]
if cfg_ds not in datasets:
    sys.exit("wrong cfg_ds name")

data_dir = "data"

print("\nStart at:", time.asctime(), "Machine:", os.uname()[1])
print("Datset selected:", cfg_ds)

word_embeddings_dim = 300
word_vector_map = {}

tfidf_mode = "all_tfidf"
# tfidf_mode='only_tf'

################################
#     pre-process the corpus
################################
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
# vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

doc_content_list = []
with open(os.path.join(data_dir, cfg_ds + ".txt"), "rb") as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode("latin1"))

word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        # word not in stop_words and word_freq[word] >= 5
        if word_freq[word] > cfg_del_infreqent_word:
            if cfg_stop_word:
                if word not in stop_words:
                    doc_words.append(word)
            else:
                doc_words.append(word)

    doc_str = " ".join(doc_words).strip()
    # if doc_str == '':
    #     doc_str = temp
    clean_docs.append(doc_str)

clean_corpus_str = "\n".join(clean_docs)

with open(os.path.join(data_dir, cfg_ds + ".clean.txt.dump"), "w") as f:
    f.write(clean_corpus_str)

##############################

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

with open(os.path.join(data_dir, cfg_ds + "_label.txt"), "r") as f:
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find("test") != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find("train") != -1:
            doc_train_list.append(line.strip())

doc_content_list = []
doc_len_list = []
with open(os.path.join(data_dir, cfg_ds + ".clean.txt.dump"), "r") as f:
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
        doc_len_list.append(len(line.split()))
print(
    "Max doc length:",
    np.array(doc_len_list).max(),
    " doc length mean:",
    np.array(doc_len_list).mean(),
)

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print("original train_ids len:", len(train_ids))
random.shuffle(train_ids)

# before seprate real_train and valid
train_ids_str = "\n".join(str(index) for index in train_ids)
with open(os.path.join(data_dir, cfg_ds + ".train.index.dump"), "w") as f:
    f.write(train_ids_str)

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print("test_ids len:", len(test_ids))
random.shuffle(test_ids)

test_ids_str = "\n".join(str(index) for index in test_ids)
with open(os.path.join(data_dir, cfg_ds + ".test.index.dump"), "w") as f:
    f.write(test_ids_str)

ids = train_ids + test_ids
print("all(real train+valid+test) ids len:", len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = "\n".join(shuffle_doc_name_list)
shuffle_doc_words_str = "\n".join(shuffle_doc_words_list)

with open(
    os.path.join(data_dir, cfg_ds + "_names.shuffle.txt.dump"), "w"
) as f:
    f.write(shuffle_doc_name_str)

with open(
    os.path.join(data_dir, cfg_ds + "_texts.shuffle.txt.dump"), "w"
) as f:
    f.write(shuffle_doc_words_str)

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = "\n".join(vocab)

with open(os.path.join(data_dir, cfg_ds + "_vocab.txt.dump"), "w") as f:
    f.write(vocab_str)

print("Vocab size", args.ds, len(vocab))

label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split("\t")
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = "\n".join(label_list)
with open(
    os.path.join(data_dir, cfg_ds + "_labels.shuffle.txt.dump"), "w"
) as f:
    f.write(label_list_str)


# x: feature vectors of training docs, no initial features
# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = "\n".join(real_train_doc_names)

with open(os.path.join(data_dir, cfg_ds + ".real_train.name.dump"), "w") as f:
    f.write(real_train_doc_names_str)


row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

x = sp.csr_matrix(
    (data_x, (row_x, col_x)),
    shape=(real_train_size, word_embeddings_dim),
    dtype=np.float32,
)

y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split("\t")
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix(
    (data_tx, (row_tx, col_tx)),
    shape=(test_size, word_embeddings_dim),
    dtype=np.float32,
)

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split("\t")
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words
word_vectors = np.random.uniform(
    -0.01, 0.01, (vocab_size, word_embeddings_dim)
)

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)),
    shape=(train_size + vocab_size, word_embeddings_dim),
    dtype=np.float32,
)

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split("\t")
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

# x is real_train_size, tx is test_size, allx is real_train+valid+vocab size
print(
    "shapes: real_train x",
    x.shape,
    y.shape,
    "test x",
    tx.shape,
    ty.shape,
    "real_train+valid+vocab x",
    allx.shape,
    ally.shape,
)


################################
#  Doc word heterogeneous graph
################################

windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= cfg_window_size:
        windows.append(words)
    else:
        for j in range(length - cfg_window_size + 1):
            window = words[j : j + cfg_window_size]
            windows.append(window)


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

# PMI: #W(i) is the number of sliding windows in a corpus that contain word i,
# and #W(i, j) is the number of sliding windows that contain both word i and j,
# and #W is the total number of sliding windows in the corpus.
word_pair_count = {}
for window in windows:
    appeared = set()
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + "," + str(word_j_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)
            # two orders
            word_pair_str = str(word_j_id) + "," + str(word_i_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)

row = []
col = []
weight = []
tfidf_row = []
tfidf_col = []
tfidf_weight = []
vocab_adj_npmi_row = []
vocab_adj_npmi_col = []
vocab_adj_npmi_weight = []
vocab_adj_pmi_row = []
vocab_adj_pmi_col = []
vocab_adj_pmi_weight = []

# pmi as weights
num_window = len(windows)
tmp_max_npmi = 0
tmp_min_npmi = 0
tmp_max_pmi = 0
tmp_min_pmi = 0

for key in word_pair_count:
    temp = key.split(",")
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log(
        (1.0 * count / num_window)
        / (1.0 * word_freq_i * word_freq_j / (num_window * num_window))
    )
    npmi = (
        log(1.0 * word_freq_i * word_freq_j / (num_window * num_window))
        / log(1.0 * count / num_window)
        - 1
    )
    if npmi > tmp_max_npmi:
        tmp_max_npmi = npmi
    if npmi < tmp_min_npmi:
        tmp_min_npmi = npmi
    if pmi > tmp_max_pmi:
        tmp_max_pmi = pmi
    if pmi < tmp_min_pmi:
        tmp_min_pmi = pmi
    if pmi > 0:
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)
        vocab_adj_pmi_row.append(i)
        vocab_adj_pmi_col.append(j)
        vocab_adj_pmi_weight.append(pmi)
    if npmi > 0:
        vocab_adj_npmi_row.append(i)
        vocab_adj_npmi_col.append(j)
        vocab_adj_npmi_weight.append(npmi)
print("max_pmi:", tmp_max_pmi, "min_pmi:", tmp_min_pmi)
print("max_npmi:", tmp_max_npmi, "min_npmi:", tmp_min_npmi)

# add pmi(self), i.e. the diagonale of adjacency matrix A of pmi
for i in range(len(vocab)):
    pmi_self = log((1.0 * num_window) / (1.0 * word_window_freq[vocab[i]]))
    if pmi_self > tmp_max_pmi:
        tmp_max_pmi = pmi_self
    vocab_adj_pmi_row.append(i)
    vocab_adj_pmi_col.append(i)
    vocab_adj_pmi_weight.append(pmi_self)
print("max_pmi after pmi_self:", tmp_max_pmi)

# word vector cosine similarity as weights

"""
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
"""
# word frequency for every doc
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + "," + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1


for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    tfidf_vec = []
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + "," + str(j)
        tf = doc_word_freq[key]
        tfidf_row.append(i)
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        tfidf_col.append(j)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) / word_doc_freq[vocab[j]])
        # idf = log((1.0 + len(shuffle_doc_words_list)) / (1.0+word_doc_freq[vocab[j]])) +1.0

        if tfidf_mode == "only_tf":
            tfidf_vec.append(tf)
        else:
            tfidf_vec.append(tf * idf)
        doc_word_set.add(word)
    if len(tfidf_vec) > 0:
        weight.extend(tfidf_vec)
        tfidf_weight.extend(tfidf_vec)
        # tfidf_weight.extend(tfidf_vec/np.linalg.norm(tfidf_vec))

vocab_adj_npmi = sp.csr_matrix(
    (vocab_adj_npmi_weight, (vocab_adj_npmi_row, vocab_adj_npmi_col)),
    shape=(vocab_size, vocab_size),
    dtype=np.float32,
)
vocab_adj_npmi.setdiag(1.0)
vocab_adj_pmi = sp.csr_matrix(
    (vocab_adj_pmi_weight, (vocab_adj_pmi_row, vocab_adj_pmi_col)),
    shape=(vocab_size, vocab_size),
    dtype=np.float32,
)
# adj order is real_train, valid, vocab, test
node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size), dtype=np.float32
)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj.setdiag(1.0)

# tfidf_all=np.vstack((adj[:train_size, train_size:train_size+vocab_size],tfidf_test))
tfidf_all = sp.csr_matrix(
    (tfidf_weight, (tfidf_row, tfidf_col)),
    shape=(train_size + test_size, vocab_size),
    dtype=np.float32,
)
tfidf_real_train = tfidf_all[:real_train_size]
tfidf_valid = tfidf_all[real_train_size:train_size]
tfidf_test = tfidf_all[train_size:]
tfidf_X_list = [tfidf_real_train, tfidf_valid, tfidf_test]
# tfidf_train_valid=adj[:train_size, train_size:train_size+vocab_size]
# vocab_adj_tfidf=tfidf_train_valid.T.dot(tfidf_train_valid)
vocab_tfidf = tfidf_all.T.tolil()
# vocab_tfidf/=np.linalg.norm(vocab_tfidf,axis=1)[:,None]
for i in range(vocab_size):
    norm = np.linalg.norm(vocab_tfidf.data[i])
    if norm > 0:
        vocab_tfidf.data[i] /= norm
# after dot, idf has not effect
vocab_adj_tf = vocab_tfidf.dot(vocab_tfidf.T)


################################
#      Dump Objects
################################

with open(os.path.join(data_dir, cfg_ds + ".tfidf_list.dump"), "wb") as f:
    pkl.dump(tfidf_X_list, f)
with open(os.path.join(data_dir, cfg_ds + ".vocab_adj_npmi.dump"), "wb") as f:
    pkl.dump(vocab_adj_npmi, f)
with open(os.path.join(data_dir, cfg_ds + ".vocab_adj_pmi.dump"), "wb") as f:
    pkl.dump(vocab_adj_pmi, f)
with open(os.path.join(data_dir, cfg_ds + ".vocab_adj_tf.dump"), "wb") as f:
    pkl.dump(vocab_adj_tf, f)
with open(os.path.join(data_dir, cfg_ds + ".vocab.dump"), "wb") as f:
    pkl.dump(vocab, f)

with open(os.path.join(data_dir, cfg_ds + ".x.dump"), "wb") as f:
    pkl.dump(x, f)

with open(os.path.join(data_dir, cfg_ds + ".y.dump"), "wb") as f:
    pkl.dump(y, f)

with open(os.path.join(data_dir, cfg_ds + ".tx.dump"), "wb") as f:
    pkl.dump(tx, f)

with open(os.path.join(data_dir, cfg_ds + ".ty.dump"), "wb") as f:
    pkl.dump(ty, f)

with open(os.path.join(data_dir, cfg_ds + ".allx.dump"), "wb") as f:
    pkl.dump(allx, f)

with open(os.path.join(data_dir, cfg_ds + ".ally.dump"), "wb") as f:
    pkl.dump(ally, f)

with open(os.path.join(data_dir, cfg_ds + ".adj.dump"), "wb") as f:
    pkl.dump(adj, f)

# WGCN
The implementation of **Word-Grounded-Graph-Convolutional-Network**.

#### Authors: Zhibin Lu, Qianqian Xie, Jian-yun Nie, Benyou Wang

## Requirements

* Python 3.7.2
* PyTorch 1.0
* scikit-learn 0.20.1
* scipy 1.1.0
* numpy 1.15.4
* glove.6B.300d.txt (copy to data/ dir)

## Datasets

1. Demo dataset is `mr`, in `data/` dir.

## Pre-processing

1. Run `python build_graph.py mr`

## Trainning
1. For Original GCN, Text GCN, run `python train_tgcn.py`
2. For MLP, run `python train_mlp.py`
3. For WGCN, run `python train_wgcn.py`
4. For WGCN using a Vocabuary embedding, run `python train_wgcn_vocab_embedding.py` (download glove.6B.300d.txt and copy to data/ dir)
5. For WGCN using a word embedding X, run `python train_wgcn_word_embedding.py` (download glove.6B.300d.txt and copy to data/ dir)

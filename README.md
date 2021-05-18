# WGCN
The implementation of **Word-Grounded-Graph-Convolutional-Network**.

#### Authors: Zhibin Lu, Qianqian Xie, Jian-yun Nie, Benyou Wang

## Requirements

* Python 3.7.2
* PyTorch 1.0
* scikit-learn 0.20.1
* scipy 1.1.0
* numpy 1.15.4

## Datasets

1. Demo dataset is `mr`, in `data/` dir.

## Pre-processing

1. Run `python build_graph.py mr`

## Trainning
1. For Original GCN, Text GCN, Run `python train_tgcn.py`
2. For MLP, Run `python train_mlp.py`
3. For WGCN, Run `python train_wgcn.py`

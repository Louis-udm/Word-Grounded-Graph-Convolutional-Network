# WGCN
The implementation of **[Word Grounded Graph Convolutional Network](https://arxiv.org/abs/2305.06434)**.

#### Authors: [Zhibin Lu](https://louis-udm.github.io) (zhibin.lu@umontreal.ca), Qianqian Xie (qianqian.xie@manchester.ac.uk), Benyou Wang (wang@dei.unipd.it), [Jian-Yun Nie](http://rali.iro.umontreal.ca/nie/jian-yun-nie/) (nie@iro.umontreal.ca)

## Overview
This is the implementation of [Word Grounded Graph Convolutional Network](https://arxiv.org/abs/2305.06434). If you make use of this code or the WGCN or WGraph approach in your work, please cite the following paper:

     @inproceedings{ZhibinluWGCN,
	     author    = {Zhibin Lu and Qianqian Xie and Benyou Wang and Jian-Yun Nie},
	     title     = {Word Grounded Graph Convolutional Network},
	     publisher = {arXiv},
	     year      = {2023},
	  }

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

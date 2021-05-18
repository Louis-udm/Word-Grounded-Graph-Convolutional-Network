"""
The model package
"""

from models.gcn import GCN_2Layers
from models.mlp import MLP_1h, MLP_2h
from models.wgcn import WGCN, WGCN_embedding_classifier, WGCN_VocabEmbedding

__all__ = [
    "MLP_1h",
    "MLP_2h",
    "GCN_2Layers",
    "WGCN",
    "WGCN_embedding_classifier",
    "WGCN_VocabEmbedding",
]

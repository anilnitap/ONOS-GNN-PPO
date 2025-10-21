"""
gnn_model.py
Simple Graph Neural Network (GNN) implementation using PyTorch.

Purpose:
- Provide a small, reproducible GNN that computes node embeddings from an adjacency matrix
  and node features. Used by PPO agent as state embeddings.

Usage (example):
    from gnn_model import SimpleGNN
    gnn = SimpleGNN(in_feat=8, hidden=64, out_feat=64)
    embeddings = gnn(adj, features)   # adj: torch.FloatTensor [N,N], features: [N, in_feat]

Dependencies:
    torch, torch.nn, torch.nn.functional, numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)

    def forward(self, adj, h):
        # adj: [N, N], h: [N, in_feats]
        # simple symmetric normalization
        deg = adj.sum(dim=1, keepdim=True) + 1e-6
        h_agg = torch.matmul(adj, h) / deg
        return self.linear(h_agg)


class SimpleGNN(nn.Module):
    def __init__(self, in_feat=8, hidden=64, out_feat=64, num_layers=2, activation=F.relu):
        super(SimpleGNN, self).__init__()
        self.activation = activation
        layers = []
        if num_layers == 1:
            layers.append(GraphConvLayer(in_feat, out_feat))
        else:
            layers.append(GraphConvLayer(in_feat, hidden))
            for _ in range(num_layers - 2):
                layers.append(GraphConvLayer(hidden, hidden))
            layers.append(GraphConvLayer(hidden, out_feat))
        self.layers = nn.ModuleList(layers)

    def forward(self, adj, features):
        """
        adj: torch.FloatTensor [N, N]
        features: torch.FloatTensor [N, in_feat]
        returns: embeddings [N, out_feat]
        """
        h = features
        for layer in self.layers:
            h = layer(adj, h)
            h = self.activation(h)
        return h


if __name__ == "__main__":
    # quick demo with synthetic graph
    torch.manual_seed(0)
    N = 10
    in_feat = 8
    adj = torch.eye(N)
    # connect some edges
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    features = torch.randn(N, in_feat)
    gnn = SimpleGNN(in_feat=in_feat, hidden=32, out_feat=16, num_layers=2)
    emb = gnn(adj, features)
    print("GNN embeddings shape:", emb.shape)

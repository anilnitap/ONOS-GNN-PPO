"""
utils.py
Helper utilities for synthetic dataset generation, seeding, and metrics.

Functions:
- set_seed(seed)
- synthetic_topology_example(num_nodes, feat_dim)
- synthetic_hciot_samples(n_samples, feat_dim, imbalance=True, seed=None)
- compute_auc(logits, labels)  # simplified AUC estimation for demo

Save this file as src/utils.py
"""

import random
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def synthetic_topology_example(num_nodes=16, feat_dim=8):
    """
    Generate a simple chain+random edges adjacency and random node features.
    Returns: adj (torch.FloatTensor [N,N]), features (np.ndarray [N, feat_dim])
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=float)
    # chain
    for i in range(num_nodes - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    # add some random extra edges
    rng = np.random.RandomState(42)
    extra = rng.choice(range(num_nodes), size=(num_nodes // 2, 2))
    for a, b in extra:
        if a != b:
            adj[a, b] = adj[b, a] = 1.0
    # features
    features = rng.randn(num_nodes, feat_dim)
    return torch.tensor(adj, dtype=torch.float32), features


def synthetic_hciot_samples(n_samples=1000, feat_dim=16, imbalance=True, seed=None):
    """
    Generate synthetic H-CIoT-like features and binary labels (0=normal,1=attack).
    If imbalance=True, attack samples are minority.
    Returns: X (np.array), y (np.array)
    """
    rng = np.random.RandomState(seed if seed is not None else 0)
    if imbalance:
        # 80% normal, 20% attack
        n_attack = int(0.2 * n_samples)
        n_normal = n_samples - n_attack
    else:
        n_attack = n_samples // 2
        n_normal = n_samples - n_attack

    X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, feat_dim))
    X_attack = rng.normal(loc=1.5, scale=1.0, size=(n_attack, feat_dim))  # shifted mean
    X = np.vstack([X_normal, X_attack])
    y = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_attack, dtype=int)])
    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def compute_auc(logits, labels):
    """
    Simplified AUC: expects logits as numpy array [N,2]; uses softmax predicted probabilities for class 1.
    """
    try:
        probs = np.exp(logits[:, 1]) / (np.exp(logits).sum(axis=1) + 1e-8)
        return roc_auc_score(labels, probs)
    except Exception:
        # fallback: random
        return 0.5

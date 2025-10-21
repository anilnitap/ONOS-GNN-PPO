"""
federated_ids.py
Minimal federated training simulation for anomaly detection (privacy-preserving).
- Each client trains a small MLP on local synthetic data.
- Server aggregates models using weighted averaging (FederatedAveraging).
- Demonstrates defense hook for Krum/median detection (simple outlier removal).

Usage:
    python federated_ids.py
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import set_seed, synthetic_hciot_samples, compute_auc

set_seed(123)


class MLPDetector(nn.Module):
    def __init__(self, in_dim=16, hidden=64):
        super(MLPDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2)  # binary: normal vs attack
        )

    def forward(self, x):
        return self.net(x)


def local_train(model, data_x, data_y, epochs=3, lr=1e-3):
    model = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        inputs = torch.tensor(data_x, dtype=torch.float32)
        labels = torch.tensor(data_y, dtype=torch.long)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.state_dict()


def aggregate_models(state_dicts, weights):
    # simple weighted average
    agg = {}
    for k in state_dicts[0].keys():
        agg[k] = sum([state_dicts[i][k] * weights[i] for i in range(len(state_dicts))])
    return agg


def detect_outliers(state_dicts):
    # naive: remove the model whose weight norms are far from median
    norms = []
    for sd in state_dicts:
        norm = 0.0
        for k, v in sd.items():
            norm += np.linalg.norm(v.cpu().numpy().ravel())
        norms.append(norm)
    med = np.median(norms)
    filtered = []
    for i, n in enumerate(norms):
        if abs(n - med) < 1.5 * np.std(norms):
            filtered.append(state_dicts[i])
    if len(filtered) == 0:
        # fallback: return all
        return state_dicts
    return filtered


def federated_simulation(num_clients=5, rounds=5):
    in_dim = 16
    global_model = MLPDetector(in_dim=in_dim)
    client_sizes = []
    client_states = []
    print("Starting federated simulation with {} clients".format(num_clients))

    # generate per-client synthetic data
    for i in range(num_clients):
        X, y = synthetic_hciot_samples(n_samples=500, feat_dim=in_dim, imbalance=True, seed=100 + i)
        client_sizes.append(len(y))
        state = local_train(global_model, X, y, epochs=2)
        client_states.append(state)

    for r in range(rounds):
        # optional: detect outliers and filter
        filtered = detect_outliers(client_states)
        weights = [client_sizes[i] / sum(client_sizes) for i in range(len(client_states))]
        agg_state = aggregate_models(filtered, weights[:len(filtered)])
        global_model.load_state_dict(agg_state)

        # evaluate global model on a held-out synthetic testset
        Xtest, ytest = synthetic_hciot_samples(n_samples=1000, feat_dim=in_dim, imbalance=True, seed=999)
        global_model.eval()
        with torch.no_grad():
            logits = global_model(torch.tensor(Xtest, dtype=torch.float32)).numpy()
            auc = compute_auc(logits, ytest)
        print(f"Round {r+1}/{rounds}: Global model AUC (est): {auc:.3f}")

        # simulate clients performing another local update
        client_states = []
        for i in range(num_clients):
            X, y = synthetic_hciot_samples(n_samples=500, feat_dim=in_dim, imbalance=True, seed=200 + r + i)
            state = local_train(global_model, X, y, epochs=1)
            client_states.append(state)

    print("Federated simulation complete.")


if __name__ == "__main__":
    federated_simulation(num_clients=5, rounds=4)

"""
ppo_agent.py
Lightweight PPO agent skeleton that consumes GNN embeddings as state and outputs
routing decisions in a simulated environment.

This is a simplified, educational implementation:
- Actor: MLP producing action logits over 'num_links' possible actions per node.
- Critic: MLP value estimator.
- Uses a simple on-policy loop with synthetic transitions for demonstration.

Usage:
    python ppo_agent.py
This will run a tiny training loop with synthetic data.

Dependencies:
    torch, numpy
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import set_seed, synthetic_topology_example

set_seed(42)


class Actor(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=4):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class Critic(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc2(x)
        return v.squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, clip_epsilon=0.2, gamma=0.99):
        self.actor = Actor(state_dim, hidden=128, out_dim=action_dim)
        self.critic = Critic(state_dim, hidden=128)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

    def select_action(self, state):
        # state: [N_nodes, state_dim]
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        logp = m.log_prob(action)
        return action, logp, probs

    def compute_returns(self, rewards, dones, last_value=0.0):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def ppo_update(self, states, actions, old_logps, returns, advantages, epochs=4):
        for _ in range(epochs):
            logits = self.actor(states)
            probs = F.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            logps = m.log_prob(actions)
            ratio = torch.exp(logps - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def synthetic_training_loop():
    # generate synthetic topology features and adjacency
    adj, features = synthetic_topology_example(num_nodes=16, feat_dim=16)
    device = torch.device("cpu")
    features = torch.tensor(features, dtype=torch.float32).to(device)
    # For simplicity, use mean of node features as global state vector per node
    state = features  # [N, feat_dim]
    state_dim = state.shape[1]
    action_dim = 4  # assume up to 4 outgoing links choices per node
    ppo = PPOAgent(state_dim, action_dim)

    # synthetic rollout buffers
    all_states = []
    all_actions = []
    all_logps = []
    rewards = []
    dones = []

    # tiny training loop
    for step in range(50):
        action, logp, _ = ppo.select_action(state)
        # synthetic reward: prefer lower 'simulated latency' derived from features
        reward = (state.mean(dim=1) * 0.1).detach().numpy()
        # convert reward to tensor per node
        rew_tensor = torch.tensor(reward, dtype=torch.float32)
        done = np.array([False] * state.shape[0])
        all_states.append(state)
        all_actions.append(action)
        all_logps.append(logp)
        rewards.append(rew_tensor)
        dones.append(done)

        # small random perturbation to state to simulate environment change
        state = state + 0.01 * torch.randn_like(state)

    # Prepare tensors for update (concatenate along nodes*steps)
    states_cat = torch.cat([s for s in all_states], dim=0)
    actions_cat = torch.cat([a for a in all_actions], dim=0)
    old_logps_cat = torch.cat([lp for lp in all_logps], dim=0)
    rewards_cat = torch.cat(rewards, dim=0)
    dones_cat = torch.tensor(np.concatenate(dones), dtype=torch.bool)

    # compute returns and advantages (simple)
    returns = ppo.compute_returns(rewards_cat.tolist(), dones_cat.tolist())
    values = ppo.critic(states_cat).detach()
    advantages = returns - values

    ppo.ppo_update(states_cat, actions_cat, old_logps_cat, returns, advantages)
    print("PPO training: single synthetic update complete.")


if __name__ == "__main__":
    synthetic_training_loop()

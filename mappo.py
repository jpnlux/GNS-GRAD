# mappo.py

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        """
        obs: (B, obs_dim)
        戻り値: Categorical distribution
        """
        logits = self.net(obs)
        return Categorical(logits=logits)


class CentralCritic(nn.Module):
    def __init__(self, obs_dim: int, num_agents: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs_all):
        """
        obs_all: (T, obs_dim * num_agents) or (B, obs_dim * num_agents)
        戻り値: (T,) or (B,) の V(s)
        """
        v = self.net(obs_all)
        return v.squeeze(-1)

import torch
import torch.nn as nn

class LinearActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.layer = nn.Linear(obs_dim, action_dim)

    def forward(self, obs):
        # logits を返す
        return self.layer(obs)

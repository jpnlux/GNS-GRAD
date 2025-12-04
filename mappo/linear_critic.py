import torch
import torch.nn as nn

class LinearCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.value = nn.Linear(state_dim, 1)

    def forward(self, state):
        return self.value(state)

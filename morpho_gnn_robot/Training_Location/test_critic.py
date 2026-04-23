import torch
import torch.nn as nn
from gnn_actor_critic import _layer_init

# Create critic head exactly as in code
hidden_dim = 48
critic_head = nn.Sequential(_layer_init(nn.Linear(hidden_dim, 32)), nn.Tanh(), _layer_init(nn.Linear(32, 1), std=1.0))

# Pass random inputs simulating pooled GNN features
inputs = torch.randn(100, 48) * 0.27
outputs = critic_head(inputs)

print("Critic head output variance:", outputs.var().item())

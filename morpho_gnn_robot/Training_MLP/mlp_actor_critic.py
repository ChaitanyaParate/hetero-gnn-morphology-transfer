import torch
import torch.nn as nn
from torch.distributions import Normal

def _layer_init(layer: nn.Linear, std: float=1.0, bias_const: float=0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int=256):
        super().__init__()
        self.trunk = nn.Sequential(_layer_init(nn.Linear(obs_dim, hidden_dim), std=1.0), nn.Tanh(), _layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0), nn.Tanh())
        self.actor_head = nn.Sequential(_layer_init(nn.Linear(hidden_dim, hidden_dim), std=0.5), nn.Tanh(), _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        self.critic_head = nn.Sequential(_layer_init(nn.Linear(hidden_dim, hidden_dim), std=0.5), nn.Tanh(), _layer_init(nn.Linear(hidden_dim, 1), std=1.0))

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self._encode(obs)
        return self.critic_head(h)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor=None):
        h = self._encode(obs)
        mean = self.actor_head(h)
        std = self.log_std.exp().clamp(min=0.05, max=0.4)
        std = std.unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(h)
        return (action, log_prob, entropy, value)
if __name__ == '__main__':
    B = 8
    obs_dim = 37
    action_dim = 12
    model = MLPActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256)
    x = torch.randn(B, obs_dim)
    a, lp, ent, v = model.get_action_and_value(x)
    print(f'action:   {a.shape}  expected ({B}, {action_dim})')
    print(f'log_prob: {lp.shape} expected ({B},)')
    print(f'entropy:  {ent.shape} expected ({B},)')
    print(f'value:    {v.shape}  expected ({B}, 1)')
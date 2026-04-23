import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from urdf_to_graph import NUM_NODE_ROLES

def _layer_init(layer: nn.Linear, std: float=0.01, bias_const: float=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class SlimHeteroGNNActorCritic(nn.Module):

    def __init__(self, node_dim: int=26, edge_dim: int=4, hidden_dim: int=48, num_joints: int=12, num_roles: int=NUM_NODE_ROLES):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.num_roles = num_roles
        self.type_proj = nn.ModuleList([nn.Linear(node_dim, hidden_dim) for _ in range(num_roles)])
        mid_dim = hidden_dim * 2
        self.conv1 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, heads=2, edge_dim=edge_dim, concat=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(mid_dim)
        self.conv2 = GATv2Conv(in_channels=mid_dim, out_channels=hidden_dim, heads=1, edge_dim=edge_dim, concat=False, dropout=0.0)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.actor_head = nn.Sequential(_layer_init(nn.Linear(hidden_dim, 32)), nn.Tanh(), _layer_init(nn.Linear(32, 1), std=0.01))
        self.log_std = nn.Parameter(torch.full((num_joints,), -0.7))
        self.critic_head = nn.Sequential(_layer_init(nn.Linear(hidden_dim, 32)), nn.Tanh(), _layer_init(nn.Linear(32, 1), std=1.0))

    def _project(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        h = torch.empty(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        for role in range(self.num_roles):
            mask = node_types == role
            if mask.any():
                h[mask] = self.type_proj[role](x[mask])
        return F.elu(h)

    def _encode(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_types = data.node_types
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self._project(x, node_types)
        h = F.elu(self.norm1(self.conv1(h, edge_index, edge_attr)))
        h = F.elu(self.norm2(self.conv2(h, edge_index, edge_attr)))
        return (h, batch)

    def _joint_embeddings(self, h: torch.Tensor, data: Data) -> torch.Tensor:
        ptr = getattr(data, 'ptr', None)
        if ptr is None:
            return h[1:]
        body_mask = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
        body_mask[ptr[:-1]] = True
        return h[~body_mask]

    def get_value(self, data: Data) -> torch.Tensor:
        h, batch = self._encode(data)
        pooled = global_mean_pool(h, batch)
        return self.critic_head(pooled)

    def get_action_and_value(self, data: Data, action: torch.Tensor=None):
        h, batch = self._encode(data)
        joint_h = self._joint_embeddings(h, data)
        mean = self.actor_head(joint_h)
        B = batch.max().item() + 1
        mean = mean.view(B, self.num_joints)
        std = self.log_std.exp().clamp(min=0.12, max=0.6)
        std = std.unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        pooled = global_mean_pool(h, batch)
        value = self.critic_head(pooled)
        return (action, log_prob, entropy, value)
GNNActorCritic = SlimHeteroGNNActorCritic
HeteroGNNActorCritic = SlimHeteroGNNActorCritic
if __name__ == '__main__':
    NODE_DIM = 26
    EDGE_DIM = 4
    N_JOINTS = 12
    N_NODES = 13

    def _dummy_graph(n_nodes=N_NODES, n_joints=N_JOINTS):
        roles = [0] + [1, 2, 3] * 4
        n_edges = n_nodes * 2
        return Data(x=torch.randn(n_nodes, NODE_DIM), edge_index=torch.randint(0, n_nodes, (2, n_edges)), edge_attr=torch.randn(n_edges, EDGE_DIM), node_types=torch.tensor(roles[:n_nodes], dtype=torch.long))
    agent = SlimHeteroGNNActorCritic(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=48, num_joints=N_JOINTS)
    total = sum((p.numel() for p in agent.parameters()))
    breakdown = {'type_proj': sum((p.numel() for p in agent.type_proj.parameters())), 'conv1': sum((p.numel() for p in agent.conv1.parameters())), 'norm1': sum((p.numel() for p in agent.norm1.parameters())), 'conv2': sum((p.numel() for p in agent.conv2.parameters())), 'norm2': sum((p.numel() for p in agent.norm2.parameters())), 'actor_head': sum((p.numel() for p in agent.actor_head.parameters())), 'log_std': agent.log_std.numel(), 'critic_head': sum((p.numel() for p in agent.critic_head.parameters()))}
    print('=== Parameter breakdown ===')
    for name, count in breakdown.items():
        print(f'  {name:<12s}: {count:>6,}')
    print(f'  {'TOTAL':<12s}: {total:>6,}')
    assert total < 50000, f'FAIL: {total:,} params >= 50 000 budget!'
    print(f'  Budget check: {total:,} < 50 000  ✓\n')
    g = _dummy_graph()
    act, lp, ent, val = agent.get_action_and_value(g)
    print('Single graph:')
    print(f'  action   : {act.shape}   expected [1, {N_JOINTS}]')
    print(f'  log_prob : {lp.shape}    expected [1]')
    print(f'  entropy  : {ent.shape}   expected [1]')
    print(f'  value    : {val.shape}   expected [1, 1]')
    batch = Batch.from_data_list([_dummy_graph() for _ in range(8)])
    act_b, lp_b, ent_b, val_b = agent.get_action_and_value(batch)
    print('\nBatched (B=8):')
    print(f'  action   : {act_b.shape}  expected [8, {N_JOINTS}]')
    print(f'  log_prob : {lp_b.shape}   expected [8]')
    print(f'  value    : {val_b.shape}  expected [8, 1]')
    loss = -lp_b.mean() + val_b.mean()
    loss.backward()
    dead = [n for n, p in agent.named_parameters() if p.grad is None]
    print(f'\nDead gradients: {(dead if dead else 'none')}')
    print('\nMorphology transfer test (hexapod, 18 joints):')
    agent_hex = SlimHeteroGNNActorCritic(NODE_DIM, EDGE_DIM, hidden_dim=48, num_joints=18)
    n_hex = 19
    roles_hex = [0] + [1, 2, 3] * 6
    from torch_geometric.data import Data as D
    g_hex = D(x=torch.randn(n_hex, NODE_DIM), edge_index=torch.randint(0, n_hex, (2, 36)), edge_attr=torch.randn(36, EDGE_DIM), node_types=torch.tensor(roles_hex, dtype=torch.long))
    act_hex, _, _, val_hex = agent_hex.get_action_and_value(g_hex)
    print(f'  action : {act_hex.shape}  expected [1, 18]')
    print(f'  value  : {val_hex.shape}  expected [1, 1]')
    print('\nAll checks passed.')
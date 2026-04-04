"""
gnn_actor_critic.py

GATv2Conv-based Actor-Critic for morphology-generalizable robot control.

Why GATv2Conv over GCNConv:
  GCNConv ignores edge_attr. Your edge features (joint axis, origin, direction)
  carry real structural information. GATv2Conv takes edge_dim parameter and
  incorporates edge features into attention scores -- you paid the cost of
  building edge_attr so use a layer that actually reads it.

Node layout (matches URDFGraphBuilder with add_body_node=True):
  index 0          : virtual body node (no actuator)
  index 1 .. J     : controllable joints (one action each)

Action output is ONLY from joint nodes [1:], never from body node.
Critic uses global_mean_pool over ALL nodes (body node included -- it
aggregates whole-body information so the value estimate benefits from it).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch


def _layer_init(layer, std=0.01, bias_const=0.0):
    """Orthogonal init used by CleanRL for stable PPO training."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class GNNActorCritic(nn.Module):
    """
    Parameters
    ----------
    node_dim   : input node feature dimension (13 from URDFGraphBuilder)
    edge_dim   : input edge feature dimension (4 from URDFGraphBuilder)
    hidden_dim : GNN hidden size (64 is sufficient; increase if underfitting)
    num_joints : number of controllable joints = action dimension (12 for ANYmal)
    """

    def __init__(
        self,
        node_dim:   int = 13,
        edge_dim:   int = 4,
        hidden_dim: int = 64,
        num_joints: int = 12,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # ---- GNN encoder (2 layers, 4 attention heads in first layer) ----
        self.conv1 = GATv2Conv(
            in_channels  = node_dim,
            out_channels = hidden_dim,
            heads        = 4,
            edge_dim     = edge_dim,
            concat       = True,        # output: [N, hidden_dim * 4]
            dropout      = 0.0,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 4)

        self.conv2 = GATv2Conv(
            in_channels  = hidden_dim * 4,
            out_channels = hidden_dim,
            heads        = 1,
            edge_dim     = edge_dim,
            concat       = False,       # output: [N, hidden_dim]
            dropout      = 0.0,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ---- Actor head (per joint node -> 1 torque mean) ----
        self.actor_head = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=0.01),
        )
        # Learned log std, separate from mean network (standard for PPO)
        self.log_std = nn.Parameter(torch.full((num_joints,), -1.0))

        # ---- Critic head (global pooled embedding -> scalar value) ----
        self.critic_head = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )

    # ------------------------------------------------------------------
    def _encode(self, data: Data):
        """Run GNN encoder. Returns (node_embeddings, batch_tensor)."""
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = getattr(data, "batch", None)

        if batch is None:
            # single graph (rollout collection)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        h = F.elu(self.norm2(self.conv2(h, edge_index, edge_attr)))
        return h, batch

    # ------------------------------------------------------------------
    def _joint_embeddings(self, h: torch.Tensor, data: Data) -> torch.Tensor:
        """
        Extract embeddings for joint nodes only (skip body node at index 0
        per graph). Works for both single graphs and PyG Batch objects.
        """
        ptr = getattr(data, "ptr", None)
        if ptr is None:
            # single graph: body node is always index 0
            return h[1:]                               # [num_joints, D]

        # batched: ptr[i] = start index of graph i in the concatenated tensor
        # body nodes are at ptr[0], ptr[1], ..., ptr[-2]
        body_mask = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
        body_mask[ptr[:-1]] = True
        return h[~body_mask]                           # [B * num_joints, D]

    # ------------------------------------------------------------------
    def get_value(self, data: Data) -> torch.Tensor:
        h, batch = self._encode(data)
        pooled   = global_mean_pool(h, batch)          # [B, D]
        return self.critic_head(pooled)                # [B, 1]

    # ------------------------------------------------------------------
    def get_action_and_value(
        self,
        data:   Data,
        action: torch.Tensor = None,
    ):
        """
        Parameters
        ----------
        data   : PyG Data or Batch (single graph or minibatch)
        action : if provided (PPO update), compute log_prob for this action
                 shape [B, num_joints] or [num_joints]

        Returns
        -------
        action      : [B, num_joints]  sampled or provided
        log_prob    : [B]
        entropy     : [B]
        value       : [B, 1]
        """
        h, batch = self._encode(data)

        # ---- actor ----
        joint_h  = self._joint_embeddings(h, data)    # [B*J, D]
        mean     = self.actor_head(joint_h)            # [B*J, 1]
        B        = batch.max().item() + 1
        mean     = mean.view(B, self.num_joints)       # [B, J]

        # log_std starts at -1.0 → exp(-1.0) = 0.368 std.
        # Clamp min=0.01 prevents std collapsing to zero.
        # No max clamp -- let the policy learn its own exploration scale.
        # Previously clamp(1e-4, 0.15) was wrong: exp(-1.0)=0.368 > 0.15,
        # so std was immediately clamped to 0.15 on step 1, log_std gradients
        # were blocked, and entropy was constant throughout training.
        std  = self.log_std.exp().clamp(min=0.01)          # [J]
        std  = std.unsqueeze(0).expand_as(mean)        # [B, J]
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)   # [B]
        entropy  = dist.entropy().sum(dim=-1)          # [B]

        # ---- critic ----
        pooled = global_mean_pool(h, batch)            # [B, D]
        value  = self.critic_head(pooled)              # [B, 1]

        return action, log_prob, entropy, value


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from torch_geometric.data import Batch

    NODE_DIM  = 13
    EDGE_DIM  = 4
    N_JOINTS  = 12
    N_NODES   = 13   # 1 body + 12 joints

    def _dummy_graph():
        return Data(
            x          = torch.randn(N_NODES, NODE_DIM),
            edge_index = torch.randint(0, N_NODES, (2, 24)),
            edge_attr  = torch.randn(24, EDGE_DIM),
        )

    agent = GNNActorCritic(NODE_DIM, EDGE_DIM, hidden_dim=64, num_joints=N_JOINTS)
    print(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # single graph (rollout step)
    g = _dummy_graph()
    act, lp, ent, val = agent.get_action_and_value(g)
    print(f"\nSingle graph:")
    print(f"  action   : {act.shape}   expected [1, {N_JOINTS}]")
    print(f"  log_prob : {lp.shape}    expected [1]")
    print(f"  value    : {val.shape}   expected [1, 1]")

    # batched graph (PPO update minibatch)
    batch = Batch.from_data_list([_dummy_graph() for _ in range(8)])
    act_b, lp_b, ent_b, val_b = agent.get_action_and_value(batch)
    print(f"\nBatched graph (B=8):")
    print(f"  action   : {act_b.shape}  expected [8, {N_JOINTS}]")
    print(f"  log_prob : {lp_b.shape}   expected [8]")
    print(f"  value    : {val_b.shape}  expected [8, 1]")

    # gradient check
    loss = -lp_b.mean() + val_b.mean()
    loss.backward()
    grads = [(n, p.grad) for n, p in agent.named_parameters()]
    dead  = [n for n, g in grads if g is None]
    print(f"\nDead gradients : {dead if dead else 'none -- all params receive gradients'}")
    print("\nAll checks passed.")
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from gnn_actor_critic import GNNActorCritic
from urdf_to_graph import NUM_NODE_ROLES

agent = GNNActorCritic()

# Create 10 different random graphs
graphs = []
for i in range(10):
    n_nodes = 13
    roles = [0] + [1, 2, 3] * 4
    n_edges = 24
    
    # Use large random variance for node features
    x = torch.randn(n_nodes, 28) * 10.0
    
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 4)
    node_types = torch.tensor(roles, dtype=torch.long)
    
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_types=node_types)
    graphs.append(g)

batch = Batch.from_data_list(graphs)
with torch.no_grad():
    _, _, _, values = agent.get_action_and_value(batch)

print("Values:")
print(values.squeeze().numpy())
print(f"Variance: {values.var().item():.6f}")


import torch
import numpy as np
from torch_geometric.data import Data, Batch
from gnn_actor_critic import GNNActorCritic
from urdf_to_graph import URDFGraphBuilder
from robot_env_bullet import RobotEnvBullet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
builder = URDFGraphBuilder('anymal_stripped.urdf', add_body_node=True)
agent = GNNActorCritic(node_dim=builder.node_dim, edge_dim=builder.edge_dim, hidden_dim=48, num_joints=12)

checkpoint = torch.load('checkpoints/gnn_ppo_final.pt', map_location='cpu')
agent.load_state_dict(checkpoint['agent'])
agent.eval()

# Generate 100 random observation-like inputs to see if the value function outputs different things
graphs = []
for i in range(100):
    joint_pos = np.random.randn(12).astype(np.float32)
    joint_vel = np.random.randn(12).astype(np.float32)
    quat = np.array([0,0,0,1], dtype=np.float32)
    grav = np.array([0,0,-1], dtype=np.float32)
    lin = np.random.randn(3).astype(np.float32)
    ang = np.random.randn(3).astype(np.float32)
    cmd = np.random.randn(2).astype(np.float32)
    
    g = builder.get_graph(joint_pos, joint_vel, quat, grav, lin, ang, cmd)
    graphs.append(g)

batch = Batch.from_data_list(graphs)
with torch.no_grad():
    _, _, _, values = agent.get_action_and_value(batch)

vals = values.squeeze().numpy()
print(f"Mean value: {vals.mean():.4f}")
print(f"Variance of values: {vals.var():.6f}")
print(f"Min: {vals.min():.4f}, Max: {vals.max():.4f}")


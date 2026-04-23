import torch
import numpy as np
from torch_geometric.data import Batch
from gnn_actor_critic import GNNActorCritic
from urdf_to_graph import URDFGraphBuilder
from robot_env_bullet import RobotEnvBullet
from train_gnn_ppo import RunningNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
builder = URDFGraphBuilder('anymal_stripped.urdf', add_body_node=True)
agent = GNNActorCritic(node_dim=builder.node_dim, edge_dim=builder.edge_dim, hidden_dim=48, num_joints=12).to(device)

checkpoint = torch.load('/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location/multi/gnn_ppo_980992.pt', map_location=device, weights_only=False)
agent.load_state_dict(checkpoint['agent'])
agent.eval()

obs_norm = RunningNorm(shape=(30,))
obs_norm.mean = checkpoint['obs_norm_mean']
obs_norm.var = checkpoint['obs_norm_var']

env = RobotEnvBullet('anymal_stripped.urdf')
obs, _ = env.reset()
obs[37:39] = np.array([0.8, 0.0])

obs_n = obs_norm.normalize(obs[:30])
graph = builder.get_graph(obs_n[:12], obs_n[12:24], obs[30:34], obs[34:37], obs_n[24:27], obs_n[27:30], obs[37:39])
batch = Batch.from_data_list([graph]).to(device)

with torch.no_grad():
    joint_h = agent._joint_embeddings(agent._encode(batch)[0], batch)
    mean_action = agent.actor_head(joint_h).view(1, agent.num_joints)

print("Mean Action:", mean_action.squeeze().cpu().numpy())

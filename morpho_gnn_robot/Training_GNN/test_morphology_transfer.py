import torch
import torch.nn as nn
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core')))
from robot_env_bullet import RobotEnvBullet
from gnn_actor_critic import SlimHeteroGNNActorCritic
from urdf_to_graph import URDFGraphBuilder
import os
import glob
import time
import numpy as np
import sys

class RunningNorm:

    def __init__(self, shape, clip: float=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.clip = clip

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-08), -self.clip, self.clip)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Training_GNN/
HEXAPOD_URDF = os.path.join(BASE_DIR, '..', 'URDFs', 'hexapod_anymal.urdf')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
if len(sys.argv) > 1:
    latest_checkpoint = sys.argv[1]
    if not os.path.exists(latest_checkpoint):
        print(f'Provided checkpoint not found: {latest_checkpoint}')
        exit(1)
else:
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pt'))
    if not checkpoints:
        print(f'No checkpoints found in {CHECKPOINT_DIR}')
        exit(1)
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
print(f'Loading checkpoint: {latest_checkpoint}')
print(f'Initializing Hexapod Environment using {HEXAPOD_URDF}...')
env = RobotEnvBullet(urdf_path=HEXAPOD_URDF, render_mode='human', height_threshold=0.15, max_episode_steps=2000)
graph_builder = URDFGraphBuilder(HEXAPOD_URDF, add_body_node=True)
num_joints = graph_builder.action_dim
num_nodes = num_joints + 1
print(f'Parsed Hexapod Graph: {num_nodes} nodes, {num_joints} controllable joints.')
device = torch.device('cpu')
model = SlimHeteroGNNActorCritic(node_dim=28, edge_dim=4, hidden_dim=48, num_joints=num_joints).to(device)
full_checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
if 'agent' in full_checkpoint:
    state_dict = full_checkpoint['agent']
elif 'model_state_dict' in full_checkpoint:
    state_dict = full_checkpoint['model_state_dict']
else:
    state_dict = full_checkpoint
hex_obs_dim = num_joints * 2 + 6
obs_norm = RunningNorm(shape=(hex_obs_dim,))
if 'obs_norm_mean' in full_checkpoint:
    q_mean = full_checkpoint['obs_norm_mean']
    q_var = full_checkpoint['obs_norm_var']
    h_mean = np.zeros(hex_obs_dim, dtype=np.float64)
    h_var = np.ones(hex_obs_dim, dtype=np.float64)
    h_mean[0:6] = q_mean[0:6]
    h_mean[6:9] = q_mean[0:3]
    h_mean[9:15] = q_mean[6:12]
    h_mean[15:18] = q_mean[6:9]
    h_var[0:6] = q_var[0:6]
    h_var[6:9] = q_var[0:3]
    h_var[9:15] = q_var[6:12]
    h_var[15:18] = q_var[6:9]
    h_mean[18:24] = q_mean[12:18]
    h_mean[24:27] = q_mean[12:15]
    h_mean[27:33] = q_mean[18:24]
    h_mean[33:36] = q_mean[18:21]
    h_var[18:24] = q_var[12:18]
    h_var[24:27] = q_var[12:15]
    h_var[27:33] = q_var[18:24]
    h_var[33:36] = q_var[18:21]
    h_mean[36:42] = q_mean[24:30]
    h_var[36:42] = q_var[24:30]
    obs_norm.mean = h_mean
    obs_norm.var = h_var
    print(f'Intercepted and padded 30-dim Running Normalization layer to {hex_obs_dim}-dim!')
target_size = model.log_std.size(0)
stored_size = state_dict['log_std'].size(0)
print(f'Intercepting checkpoint sizes... Saved log_std size: {stored_size}, Target size: {target_size}')
if stored_size < target_size:
    print(f'Expanding log_std from {stored_size} to {target_size} for zero-shot morphology transfer.')
    avg_log_std = state_dict['log_std'].mean()
    expanded_log_std = torch.full((target_size,), avg_log_std.item(), device=device)
    expanded_log_std[:stored_size] = state_dict['log_std']
    state_dict['log_std'] = expanded_log_std
model.load_state_dict(state_dict)
model.eval()
print('✅ Zero-Shot Transfer successful: Trained quadruped weights loaded perfectly onto 18-joint Hexapod architecture!')
print('\nPress Ctrl+C to stop simulation.')
try:
    obs, info = env.reset()
    while True:
        OBS_NORM_DIM = num_joints * 2 + 6
        obs_n = obs.copy()
        obs_n[:OBS_NORM_DIM] = obs_norm.normalize(obs[:OBS_NORM_DIM])
        joint_pos = obs_n[:num_joints]
        joint_vel = obs_n[num_joints:num_joints * 2]
        body_lin_vel = obs_n[num_joints * 2:num_joints * 2 + 3]
        body_ang_vel = obs_n[num_joints * 2 + 3:num_joints * 2 + 6]
        body_quat = obs[num_joints * 2 + 6:num_joints * 2 + 10].astype(np.float32)
        body_grav = obs[num_joints * 2 + 10:num_joints * 2 + 13].astype(np.float32)
        cmd = np.array([0.7, 0.0], dtype=np.float32)
        pyg_data = graph_builder.get_graph(joint_pos, joint_vel, body_quat=body_quat, body_grav=body_grav, body_lin_vel=body_lin_vel, body_ang_vel=body_ang_vel, command=cmd).to(device)
        with torch.no_grad():
            h, batch = model._encode(pyg_data)
            joint_h = model._joint_embeddings(h, pyg_data)
            action_mean = model.actor_head(joint_h).view(1, num_joints)
        action_np = action_mean[0].cpu().numpy()
        print(f'Step action min: {action_np.min():.3f}, max: {action_np.max():.3f}')
        obs, reward, terminated, truncated, info = env.step(action_np)
        if terminated or truncated:
            print(f'Episode ended. Reason: {info.get('term_reason')}, Height: {info.get('base_height'):.2f}')
            obs, info = env.reset()
            time.sleep(1.0)
        time.sleep(1 / 240.0)
except KeyboardInterrupt:
    print('\nExiting...')
finally:
    env.close()
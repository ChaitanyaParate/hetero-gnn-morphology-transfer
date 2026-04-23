import os
import glob
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
BASE_DIR = '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'Training_Location'))
sys.path.append(os.path.join(BASE_DIR, 'Training_MLP'))

from Training_Location.robot_env_bullet import RobotEnvBullet
from Training_Location.gnn_actor_critic import SlimHeteroGNNActorCritic
from Training_Location.urdf_to_graph import URDFGraphBuilder
from Training_MLP.mlp_actor_critic import MLPActorCritic

# --- Settings ---
NUM_EPISODES = 20
QUADRUPED_URDF = os.path.join(BASE_DIR, 'Training_Location', 'anymal_stripped.urdf')
HEXAPOD_URDF = os.path.join(BASE_DIR, 'Training_Location', 'hexapod_anymal.urdf')

# Set aesthetic styling for paper plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

class RunningNorm:
    def __init__(self, shape, clip: float=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.clip = clip

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-08), -self.clip, self.clip)

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def load_gnn_model(checkpoint_path, num_joints, device='cpu'):
    model = SlimHeteroGNNActorCritic(node_dim=26, edge_dim=4, hidden_dim=48, num_joints=num_joints).to(device)
    full_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = full_checkpoint.get('agent', full_checkpoint.get('model_state_dict', full_checkpoint))
    
    # Handle Observation Normalization Adaptation
    obs_dim = num_joints * 2 + 6
    obs_norm = RunningNorm(shape=(obs_dim,))
    
    if 'obs_norm_mean' in full_checkpoint:
        q_mean = full_checkpoint['obs_norm_mean']
        q_var = full_checkpoint['obs_norm_var']
        h_mean = np.zeros(obs_dim, dtype=np.float64)
        h_var = np.ones(obs_dim, dtype=np.float64)
        
        if num_joints == 18: # Hexapod transfer logic
            h_mean[0:6] = q_mean[0:6]
            h_mean[6:9] = q_mean[0:3]
            h_mean[9:15] = q_mean[6:12]
            h_mean[15:18] = q_mean[6:9]
            h_mean[18:24] = q_mean[12:18]
            h_mean[24:27] = q_mean[12:15]
            h_mean[27:33] = q_mean[18:24]
            h_mean[33:36] = q_mean[18:21]
            h_mean[36:42] = q_mean[24:30]

            h_var[0:6] = q_var[0:6]
            h_var[6:9] = q_var[0:3]
            h_var[9:15] = q_var[6:12]
            h_var[15:18] = q_var[6:9]
            h_var[18:24] = q_var[12:18]
            h_var[24:27] = q_var[12:15]
            h_var[27:33] = q_var[18:24]
            h_var[33:36] = q_var[18:21]
            h_var[36:42] = q_var[24:30]
        else: # Quadruped
            h_mean = q_mean
            h_var = q_var

        obs_norm.mean = h_mean
        obs_norm.var = h_var

    # Handle Log Std Expansion
    target_size = model.log_std.size(0)
    stored_size = state_dict['log_std'].size(0)
    if stored_size < target_size:
        avg_log_std = state_dict['log_std'].mean()
        expanded_log_std = torch.full((target_size,), avg_log_std.item(), device=device)
        expanded_log_std[:stored_size] = state_dict['log_std']
        state_dict['log_std'] = expanded_log_std

    model.load_state_dict(state_dict)
    model.eval()
    return model, obs_norm

def load_mlp_model(checkpoint_path, obs_dim, action_dim, device='cpu'):
    model = MLPActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    full_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = full_checkpoint.get('agent', full_checkpoint.get('model_state_dict', full_checkpoint))
    
    obs_norm = RunningNorm(shape=(30,)) # MLP uses 30 dim normalization based on train script
    if 'obs_norm_mean' in full_checkpoint:
        obs_norm.mean = full_checkpoint['obs_norm_mean']
        obs_norm.var = full_checkpoint['obs_norm_var']

    model.load_state_dict(state_dict)
    model.eval()
    return model, obs_norm

def evaluate_gnn(urdf_path, model, obs_norm, num_joints, num_episodes=10, render=False):
    env = RobotEnvBullet(urdf_path=urdf_path, render_mode='human' if render else None, height_threshold=0.15, max_episode_steps=1000)
    graph_builder = URDFGraphBuilder(urdf_path, add_body_node=True)
    device = next(model.parameters()).device
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        OBS_NORM_DIM = num_joints * 2 + 6

        while not done:
            obs_n = obs.copy()
            obs_n[:OBS_NORM_DIM] = obs_norm.normalize(obs[:OBS_NORM_DIM])
            joint_pos = obs_n[:num_joints]
            joint_vel = obs_n[num_joints:num_joints * 2]
            body_lin_vel = obs_n[num_joints * 2:num_joints * 2 + 3]
            body_ang_vel = obs_n[num_joints * 2 + 3:num_joints * 2 + 6]
            body_quat = obs[num_joints * 2 + 6:num_joints * 2 + 10].astype(np.float32)
            body_grav = obs[num_joints * 2 + 10:num_joints * 2 + 13].astype(np.float32)
            
            pyg_data = graph_builder.get_graph(joint_pos, joint_vel, body_quat=body_quat, body_grav=body_grav, body_lin_vel=body_lin_vel, body_ang_vel=body_ang_vel).to(device)
            
            with torch.no_grad():
                h, batch = model._encode(pyg_data)
                joint_h = model._joint_embeddings(h, pyg_data)
                action_mean = model.actor_head(joint_h).view(1, num_joints)
            
            action_np = action_mean[0].cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += reward
            done = terminated or truncated
            
        episode_rewards.append(ep_reward)
        print(f"GNN Episode {ep+1}/{num_episodes} Reward: {ep_reward:.2f}")
        
    env.close()
    return episode_rewards

def evaluate_mlp(urdf_path, model, obs_norm, num_episodes=10, render=False):
    env = RobotEnvBullet(urdf_path=urdf_path, render_mode='human' if render else None, height_threshold=0.15, max_episode_steps=1000)
    device = next(model.parameters()).device
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            obs_n = obs.copy()
            # MLP normalizes first 30
            obs_n[:30] = obs_norm.normalize(obs[:30])
            obs_tensor = torch.FloatTensor(obs_n).unsqueeze(0).to(device)
            
            with torch.no_grad():
                h = model._encode(obs_tensor)
                action_mean = model.actor_head(h)
            
            action_np = action_mean[0].cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += reward
            done = terminated or truncated
            
        episode_rewards.append(ep_reward)
        print(f"MLP Episode {ep+1}/{num_episodes} Reward: {ep_reward:.2f}")
        
    env.close()
    return episode_rewards

def generate_plots(results):
    print("\nGenerating publication-quality plots...")
    
    # Prepare DataFrame for Seaborn
    data = []
    for condition, rewards in results.items():
        for r in rewards:
            data.append({"Model-Morphology": condition, "Cumulative Reward": r})
    df = pd.DataFrame(data)

    os.makedirs(os.path.join(BASE_DIR, 'plots'), exist_ok=True)

    # 1. Boxplot (Variance and Median)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Model-Morphology", y="Cumulative Reward", data=df, palette="Set2")
    sns.stripplot(x="Model-Morphology", y="Cumulative Reward", data=df, color="black", alpha=0.5, jitter=True)
    plt.title("Zero-Shot Morphology Transfer Performance")
    plt.ylabel("Cumulative Reward (Per Episode)")
    plt.xlabel("Evaluation Scenario")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'plots', 'zero_shot_transfer_boxplot.png'), dpi=300)
    plt.savefig(os.path.join(BASE_DIR, 'plots', 'zero_shot_transfer_boxplot.pdf'))
    plt.close()

    # 2. Bar Plot (Mean with Confidence Intervals)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Model-Morphology", y="Cumulative Reward", data=df, palette="Set2", capsize=.1)
    plt.title("Average Reward in Morphology Transfer")
    plt.ylabel("Average Cumulative Reward")
    plt.xlabel("Evaluation Scenario")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'plots', 'zero_shot_transfer_barplot.png'), dpi=300)
    plt.savefig(os.path.join(BASE_DIR, 'plots', 'zero_shot_transfer_barplot.pdf'))
    plt.close()
    
    print(f"Plots saved to {os.path.join(BASE_DIR, 'plots')}")

def main():
    gnn_ckpt = get_latest_checkpoint(os.path.join(BASE_DIR, 'Training_Location', 'checkpoints'))
    mlp_ckpt = get_latest_checkpoint(os.path.join(BASE_DIR, 'Training_MLP', 'checkpoints'))
    
    if not gnn_ckpt or not mlp_ckpt:
        print("Missing checkpoints. Ensure both MLP and GNN have trained models.")
        return

    print("--- Loading Models ---")
    gnn_quad_model, gnn_quad_norm = load_gnn_model(gnn_ckpt, num_joints=12)
    gnn_hex_model, gnn_hex_norm = load_gnn_model(gnn_ckpt, num_joints=18)
    mlp_quad_model, mlp_quad_norm = load_mlp_model(mlp_ckpt, obs_dim=37, action_dim=12)
    
    results = {}

    print(f"\n--- Evaluating MLP on Quadruped ({NUM_EPISODES} Episodes) ---")
    results["MLP (Quadruped)"] = evaluate_mlp(QUADRUPED_URDF, mlp_quad_model, mlp_quad_norm, num_episodes=NUM_EPISODES)
    
    print(f"\n--- Evaluating GNN on Quadruped ({NUM_EPISODES} Episodes) ---")
    results["GNN (Quadruped)"] = evaluate_gnn(QUADRUPED_URDF, gnn_quad_model, gnn_quad_norm, num_joints=12, num_episodes=NUM_EPISODES)
    
    print(f"\n--- Evaluating GNN on Hexapod ({NUM_EPISODES} Episodes) ---")
    results["GNN (Hexapod)"] = evaluate_gnn(HEXAPOD_URDF, gnn_hex_model, gnn_hex_norm, num_joints=18, num_episodes=NUM_EPISODES)

    # Note: MLP on Hexapod automatically fails, which is the point of the paper.
    # We record 0 (or a very low failure score) to visually represent this structural failure.
    print(f"\n--- Evaluating MLP on Hexapod ({NUM_EPISODES} Episodes) ---")
    print("MLP structure is rigid. Cannot transfer to 18-joint Hexapod.")
    results["MLP (Hexapod)\n[Architecture Failure]"] = [0.0] * NUM_EPISODES

    generate_plots(results)

if __name__ == '__main__':
    main()

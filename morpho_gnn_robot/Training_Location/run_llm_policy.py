import argparse
import time
import torch
import numpy as np
from torch_geometric.data import Batch
from gnn_actor_critic import GNNActorCritic
from urdf_to_graph import URDFGraphBuilder
from robot_env_bullet import RobotEnvBullet
from train_gnn_ppo import RunningNorm
import pybullet as p

def get_command_vector(llm_instruction: str) -> np.ndarray:
    instruction = llm_instruction.lower()
    
    # Simple semantic mapping (in a real system, an LLM would output these values)
    if 'forward' in instruction:
        cmd = [0.8, 0.0]
    elif 'backward' in instruction:
        cmd = [-0.5, 0.0]
    elif 'left' in instruction:
        cmd = [0.0, 0.6]
    elif 'right' in instruction:
        cmd = [0.0, -0.6]
    elif 'stand' in instruction or 'stop' in instruction:
        cmd = [0.0, 0.0]
    else:
        print(f"Instruction '{llm_instruction}' not recognized. Defaulting to standing.")
        cmd = [0.0, 0.0]
        
    return np.array(cmd, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained GNN policy checkpoint')
    parser.add_argument('--urdf', type=str, default='anymal_stripped.urdf')
    parser.add_argument('--instruction', type=str, default='move forward')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions (like during training)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    builder = URDFGraphBuilder(args.urdf, add_body_node=True)
    agent = GNNActorCritic(node_dim=builder.node_dim, edge_dim=builder.edge_dim, hidden_dim=48, num_joints=builder.action_dim).to(device)
    agent.load_state_dict(checkpoint['agent'])
    agent.eval()

    obs_norm = RunningNorm(shape=(30,))
    if 'obs_norm_mean' in checkpoint:
        obs_norm.mean = checkpoint['obs_norm_mean']
        obs_norm.var = checkpoint['obs_norm_var']
        obs_norm.count = checkpoint['obs_norm_count']
        print("Loaded running normalization statistics.")
    else:
        print("Warning: No running normalization statistics found in checkpoint.")

    env = RobotEnvBullet(args.urdf, render_mode="human")
    obs, _ = env.reset()
    
    # Override the environment's random command with our LLM command
    target_cmd = get_command_vector(args.instruction)
    print(f"\n[LLM INSTRUCTION]: '{args.instruction}'")
    print(f"[MAPPED COMMAND]: vx={target_cmd[0]:.2f}, wy={target_cmd[1]:.2f}\n")
    
    env.command = target_cmd
    
    step = 0
    try:
        while True:
            # 1. Extract observation components
            joint_pos = obs[:12]
            joint_vel = obs[12:24]
            body_lin_vel = obs[24:27]
            body_ang_vel = obs[27:30]
            body_quat = obs[30:34].astype(np.float32)
            body_grav = obs[34:37].astype(np.float32)
            
            # The observation vector expects the command at the end
            # We override it to make sure the policy sees the LLM command
            obs[37:39] = target_cmd
            
            # Apply the running normalizer
            obs_n = obs_norm.normalize(obs[:30])
            joint_pos_n = obs_n[:12]
            joint_vel_n = obs_n[12:24]
            body_lin_vel_n = obs_n[24:27]
            body_ang_vel_n = obs_n[27:30]
            
            # 2. Build Graph
            graph = builder.get_graph(joint_pos_n, joint_vel_n, body_quat, body_grav, body_lin_vel_n, body_ang_vel_n, target_cmd)
            batch = Batch.from_data_list([graph]).to(device)
            
            # 3. Get Action
            with torch.no_grad():
                if args.stochastic:
                    action, _, _, _ = agent.get_action_and_value(batch)
                    mean_action = action
                else:
                    # We use the mean action for deterministic evaluation
                    joint_h = agent._joint_embeddings(agent._encode(batch)[0], batch)
                    mean_action = agent.actor_head(joint_h).view(1, agent.num_joints)
            
            action_np = mean_action.squeeze(0).cpu().numpy()
            
            # 4. Step Environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            env.command = target_cmd # Re-override after step resets it if terminated
            
            # Print tracking stats every 100 steps
            if step % 100 == 0:
                current_vx = info.get('forward_vel', body_lin_vel[0])
                current_wy = body_ang_vel[2]
                print(f"Step {step}: Target [vx:{target_cmd[0]:.2f}, wy:{target_cmd[1]:.2f}] | Actual [vx:{current_vx:.2f}, wy:{current_wy:.2f}]")
                
            step += 1
            time.sleep(1/400.0) # Real-time simulation delay
            
            if terminated or truncated:
                print(f"Episode ended at step {step}. Reason: {info.get('term_reason')}")
                obs, _ = env.reset()
                env.command = target_cmd
                step = 0
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()

if __name__ == '__main__':
    main()

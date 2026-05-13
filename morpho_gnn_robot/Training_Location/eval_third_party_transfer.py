"""
eval_third_party_transfer.py
============================
Zero-shot morphological transfer evaluation across multiple robot URDFs.

Evaluates the best GNN checkpoint (Seed 2, highest stable final reward) on:
  1. anymal.urdf          — in-distribution (training morphology baseline)
  2. hexapod_anymal.urdf  — self-generated hexapod (existing paper result)
  3. aliengo.urdf         — Unitree Aliengo (third-party, independently authored)
  4. go1.urdf             — Unitree Go1   (third-party, independently authored)

For each robot: runs N_EPISODES deterministic episodes, reports mean ± std
reward, mean episode length, and success rate (episode not terminated by fall).

Results saved to eval_results_third_party.json for LaTeX table generation.

Usage:
    python eval_third_party_transfer.py [--checkpoint PATH] [--episodes N]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import statistics

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(BASE_DIR, '..', 'URDFs')
NOTEBOOK_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../../training_with_different_seed_notebooks'))

# Best stable GNN seed = Seed 2 (last-50-mean: 2269.54)
DEFAULT_CHECKPOINT = os.path.join(NOTEBOOK_DIR, 'seed2_final.pt')

ROBOTS = [
    {
        'name': 'ANYmal Quadruped (training morphology)',
        'label': 'anymal',
        'urdf': os.path.join(BASE_DIR, 'anymal.urdf'),
        'height_threshold': 0.25,
        'note': 'In-distribution baseline',
    },
    {
        'name': 'ANYmal Hexapod (self-generated)',
        'label': 'hexapod',
        'urdf': os.path.join(BASE_DIR, 'hexapod_anymal.urdf'),
        'height_threshold': 0.15,
        'note': 'Zero-shot: +6 joints (LM/RM legs cloned from training morphology)',
    },
    {
        'name': 'Unitree Aliengo (third-party)',
        'label': 'aliengo',
        'urdf': os.path.join(URDF_DIR, 'aliengo.urdf'),
        'height_threshold': 0.20,
        'note': 'Zero-shot: independently authored URDF, same 12 joints different morphology',
    },
    {
        'name': 'Unitree Go1 (third-party)',
        'label': 'go1',
        'urdf': os.path.join(URDF_DIR, 'go1.urdf'),
        'height_threshold': 0.18,
        'note': 'Zero-shot: independently authored URDF, compact quadruped body plan',
    },
]


# ── normaliser ─────────────────────────────────────────────────────────────────
class RunningNorm:
    def __init__(self, shape, clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.clip = clip

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / (np.sqrt(self.var) + 1e-8),
            -self.clip, self.clip
        )


def build_obs_norm(checkpoint: dict, num_joints: int) -> RunningNorm:
    """Anatomically-guided normaliser remapping.

    Maps the quadruped's 30-dim Welford statistics onto any robot with
    num_joints controllable joints. Follows the remapping algorithm from
    Section VII-C of the paper:
      - Front-leg dims reused directly
      - Missing dims (middle/extra legs) duplicated from nearest neighbour
      - Body dims (lin_vel, ang_vel, quat, grav) copied directly
    """
    target_obs_dim = num_joints * 2 + 6  # pos + vel + 6 body dims

    obs_norm = RunningNorm(shape=(target_obs_dim,))

    if 'obs_norm_mean' not in checkpoint:
        print('  [WARN] No obs_norm in checkpoint — using identity normaliser')
        return obs_norm

    q_mean = checkpoint['obs_norm_mean']   # shape (30,): 12 pos + 12 vel + 6 body
    q_var  = checkpoint['obs_norm_var']

    # Build target arrays
    h_mean = np.zeros(target_obs_dim, dtype=np.float64)
    h_var  = np.ones(target_obs_dim,  dtype=np.float64)

    n = num_joints
    # Joint positions (dims 0..n-1): fill from quadruped stats, tile if needed
    for i in range(n):
        src = i % 12  # wrap around quadruped joint count
        h_mean[i]     = q_mean[src]
        h_var[i]      = q_var[src]
    # Joint velocities (dims n..2n-1)
    for i in range(n):
        src = 12 + (i % 12)
        h_mean[n + i] = q_mean[src]
        h_var[n + i]  = q_var[src]
    # Body dims (lin_vel 3 + ang_vel 3) — last 6 of quadruped obs_norm (dims 24-29)
    h_mean[2*n : 2*n+6] = q_mean[24:30]
    h_var[2*n  : 2*n+6] = q_var[24:30]

    obs_norm.mean = h_mean
    obs_norm.var  = h_var
    return obs_norm


def expand_log_std(state_dict: dict, target_size: int) -> dict:
    """Expand log_std from quadruped (12) to any joint count."""
    stored = state_dict['log_std']
    if stored.size(0) == target_size:
        return state_dict
    avg = stored.mean().item()
    expanded = torch.full((target_size,), avg)
    n = min(stored.size(0), target_size)
    expanded[:n] = stored[:n]
    state_dict['log_std'] = expanded
    print(f'  log_std expanded: {stored.size(0)} → {target_size}')
    return state_dict


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate_robot(robot_cfg: dict, model, checkpoint: dict,
                   n_episodes: int, seed: int, device) -> dict:
    """Run n_episodes on a single robot URDF and return stats."""
    from robot_env_bullet import RobotEnvBullet
    from urdf_to_graph import URDFGraphBuilder

    urdf = robot_cfg['urdf']
    label = robot_cfg['label']

    if not os.path.exists(urdf):
        return {'label': label, 'error': f'URDF not found: {urdf}'}

    print(f"\n{'='*60}")
    print(f"  Robot : {robot_cfg['name']}")
    print(f"  URDF  : {urdf}")
    print(f"  Note  : {robot_cfg['note']}")
    print(f"{'='*60}")

    # Graph builder
    graph_builder = URDFGraphBuilder(urdf, add_body_node=True)
    num_joints = graph_builder.action_dim

    # Normaliser
    obs_norm = build_obs_norm(checkpoint, num_joints)

    # Extract raw state dict from checkpoint
    if 'agent' in checkpoint:
        sd = {k: v.clone() for k, v in checkpoint['agent'].items()}
    elif 'model_state_dict' in checkpoint:
        sd = {k: v.clone() for k, v in checkpoint['model_state_dict'].items()}
    else:
        sd = {k: v.clone() for k, v in checkpoint.items()
              if isinstance(v, torch.Tensor)}

    # Expand log_std to match this robot's joint count
    sd = expand_log_std(sd, num_joints)

    # Re-instantiate model with correct num_joints for this robot
    from gnn_actor_critic import SlimHeteroGNNActorCritic
    robot_model = SlimHeteroGNNActorCritic(
        node_dim=28, edge_dim=4, hidden_dim=48, num_joints=num_joints
    ).to(device)
    robot_model.load_state_dict(sd)
    robot_model.eval()
    model = robot_model  # use this robot-specific model below

    # Environment — use URDF_USE_INERTIA_FROM_FILE + ignore missing meshes
    env = RobotEnvBullet(
        urdf_path=urdf,
        render_mode=None,
        height_threshold=robot_cfg['height_threshold'],
        max_episode_steps=1000
    )

    rewards, lengths, successes = [], [], []
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        ep_reward = 0.0
        ep_length = 0
        success = True

        while True:
            obs_dim = num_joints * 2 + 6
            obs_n = obs.copy()
            obs_n[:obs_dim] = obs_norm.normalize(obs[:obs_dim])

            joint_pos = obs_n[:num_joints].astype(np.float32)
            joint_vel = obs_n[num_joints:num_joints*2].astype(np.float32)
            body_lin_vel = obs_n[num_joints*2: num_joints*2+3].astype(np.float32)
            body_ang_vel = obs_n[num_joints*2+3: num_joints*2+6].astype(np.float32)
            body_quat = obs[num_joints*2+6: num_joints*2+10].astype(np.float32)
            body_grav = obs[num_joints*2+10: num_joints*2+13].astype(np.float32)
            cmd = np.array([0.7, 0.0], dtype=np.float32)

            pyg = graph_builder.get_graph(
                joint_pos, joint_vel,
                body_quat=body_quat, body_grav=body_grav,
                body_lin_vel=body_lin_vel, body_ang_vel=body_ang_vel,
                command=cmd
            ).to(device)

            with torch.no_grad():
                h, batch = model._encode(pyg)
                joint_h = model._joint_embeddings(h, pyg)
                action_mean = model.actor_head(joint_h).view(1, num_joints)

            action_np = action_mean[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += reward
            ep_length += 1

            if terminated:
                success = (info.get('term_reason') == 'truncated' or
                           info.get('term_reason') == 'running')
                if info.get('term_reason') in ('height', 'contact', 'orientation'):
                    success = False
                break
            if truncated:
                success = True
                break

        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(success)
        print(f'  Ep {ep+1:>2}/{n_episodes} | reward={ep_reward:>8.2f} '
              f'| steps={ep_length:>4} | {"OK" if success else "FALL"}')

    env.close()

    mean_r = statistics.mean(rewards)
    std_r  = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    mean_l = statistics.mean(lengths)
    success_rate = sum(successes) / len(successes) * 100

    print(f'\n  SUMMARY — {label}')
    print(f'  Mean reward : {mean_r:.2f} ± {std_r:.2f}')
    print(f'  Mean length : {mean_l:.1f} steps')
    print(f'  Success rate: {success_rate:.1f}%')

    return {
        'label':        label,
        'name':         robot_cfg['name'],
        'note':         robot_cfg['note'],
        'num_joints':   num_joints,
        'n_episodes':   n_episodes,
        'mean_reward':  round(mean_r, 2),
        'std_reward':   round(std_r, 2),
        'mean_length':  round(mean_l, 1),
        'success_rate': round(success_rate, 1),
        'rewards':      [round(r, 2) for r in rewards],
    }


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='GNN zero-shot transfer evaluation')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help='Path to GNN checkpoint .pt file')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Episodes per robot (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducibility')
    parser.add_argument('--out', default='eval_results_third_party.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    # Verify checkpoint
    if not os.path.exists(args.checkpoint):
        # Try fallback checkpoints
        fallbacks = [
            os.path.join(NOTEBOOK_DIR, 'seed2_final.pt'),
            os.path.join(NOTEBOOK_DIR, 'seed0_p1.pt'),
            os.path.join(BASE_DIR, 'checkpoints', 'gnn_ppo_final.pt'),
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                args.checkpoint = fb
                print(f'[INFO] Using fallback checkpoint: {fb}')
                break
        else:
            print(f'[ERROR] No checkpoint found. Tried:\n  {args.checkpoint}')
            sys.exit(1)

    print(f'\nCheckpoint : {args.checkpoint}')
    print(f'Episodes   : {args.episodes} per robot')
    print(f'Seed       : {args.seed}')

    device = torch.device('cpu')

    # Load checkpoint (use ANYmal graph to initialise model)
    print('\nLoading checkpoint...')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Build model from ANYmal (12-joint) scaffold
    sys.path.insert(0, BASE_DIR)
    from gnn_actor_critic import SlimHeteroGNNActorCritic

    model = SlimHeteroGNNActorCritic(
        node_dim=28, edge_dim=4, hidden_dim=48, num_joints=12
    ).to(device)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Run evaluation on each robot
    all_results = []
    for robot_cfg in ROBOTS:
        result = evaluate_robot(
            robot_cfg, model, checkpoint,
            n_episodes=args.episodes,
            seed=args.seed,
            device=device
        )
        all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print('FINAL RESULTS SUMMARY')
    print(f"{'='*70}")
    print(f"{'Robot':<40} {'Mean Reward':>12} {'Std':>8} {'Success':>8}")
    print(f"{'-'*70}")
    for r in all_results:
        if 'error' in r:
            print(f"{r['label']:<40} ERROR: {r['error']}")
        else:
            print(f"{r['name']:<40} {r['mean_reward']:>12.2f} "
                  f"±{r['std_reward']:>7.2f}  {r['success_rate']:>6.1f}%")

    # Save results
    output = {
        'checkpoint': args.checkpoint,
        'n_episodes': args.episodes,
        'seed': args.seed,
        'results': all_results,
    }
    out_path = os.path.join(BASE_DIR, args.out)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved → {out_path}')


if __name__ == '__main__':
    main()

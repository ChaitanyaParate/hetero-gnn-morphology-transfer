"""
finetune_transfer.py  —  Kaggle-ready version
Fine-tune a pre-trained GNN locomotion policy on a new morphology.

On Kaggle:
  1. Upload all files in this folder as a Dataset
  2. Set BASE_CHECKPOINT_NAME = 'seed2_final.pt'
  3. Run: python finetune_transfer.py --target hexapod --steps 500000

Kaggle GPU: P100 (16GB). PyBullet is CPU-bound so GPU is only used
for backprop — ~800–1200 steps/s expected (vs 120 locally).
Estimated runtime: ~7 min per 500K steps on Kaggle P100.
"""

import argparse, os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from typing import List

# ── Path resolution (works both locally and on Kaggle) ────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Kaggle mounts datasets at /kaggle/input/<dataset-name>/
# When running as script, __file__ dir contains everything
BASE_CHECKPOINT_NAME = 'seed2_final.pt'
BASE_CHECKPOINT = os.path.join(HERE, BASE_CHECKPOINT_NAME)

from robot_env_bullet import RobotEnvBullet
from gnn_actor_critic import SlimHeteroGNNActorCritic
from urdf_to_graph import URDFGraphBuilder

# ── Target configs — all URDFs are in the SAME folder as this script ──────
TARGET_CONFIGS = {
    'hexapod': {
        'urdf':             os.path.join(HERE, 'hexapod_anymal.urdf'),
        'height_threshold': 0.15,
        'num_joints':       18,
        'obs_norm_dim':     42,   # 18*2 + 6
    },
    'aliengo': {
        'urdf':             os.path.join(HERE, 'aliengo_stripped.urdf'),
        'height_threshold': 0.22,
        'num_joints':       12,
        'obs_norm_dim':     30,
    },
    'go1': {
        'urdf':             os.path.join(HERE, 'go1_stripped.urdf'),
        'height_threshold': 0.18,
        'num_joints':       12,
        'obs_norm_dim':     30,
    },
}

# ── Running normaliser ─────────────────────────────────────────────────────
class RunningNorm:
    def __init__(self, shape, clip=10.0):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape, dtype=np.float64)
        self.count = 0.0001
        self.clip  = clip

    def update(self, x):
        m = np.mean(x, axis=0) if x.ndim > 1 else x
        v = np.var(x, axis=0)  if x.ndim > 1 else np.zeros_like(x)
        n = x.shape[0] if x.ndim > 1 else 1
        d = m - self.mean; tot = self.count + n
        self.mean += d * n / tot
        self.var   = (self.var * self.count + v * n + d**2 * self.count * n / tot) / tot
        self.count  = tot

    def normalize(self, x):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -self.clip, self.clip).astype(np.float32)


# ── Rollout buffer ─────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, num_steps, action_dim, device):
        self.num_steps  = num_steps
        self.action_dim = action_dim
        self.device     = device
        self.reset()

    def reset(self):
        self.graphs   = []
        self.actions  = torch.zeros(self.num_steps, self.action_dim, device=self.device)
        self.logprobs = torch.zeros(self.num_steps, device=self.device)
        self.rewards  = torch.zeros(self.num_steps, device=self.device)
        self.dones    = torch.zeros(self.num_steps, device=self.device)
        self.values   = torch.zeros(self.num_steps, device=self.device)
        self.ptr = 0

    def store(self, graph, action, logprob, reward, done, value):
        self.graphs.append(graph)
        self.actions[self.ptr]  = action.squeeze(0)
        self.logprobs[self.ptr] = logprob.squeeze()
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.values[self.ptr]   = value.squeeze()
        self.ptr += 1

    def compute_advantages(self, next_value, next_done, gamma=0.99, lam=0.95):
        adv  = torch.zeros(self.num_steps, device=self.device)
        last = 0.0
        nv   = next_value.squeeze()
        for t in reversed(range(self.num_steps)):
            nt   = 1.0 - (next_done if t == self.num_steps - 1 else self.dones[t + 1].item())
            nval = nv if t == self.num_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * nval * nt - self.values[t]
            last = delta + gamma * lam * nt * last
            adv[t] = last
        return adv, adv + self.values


# ── Observation builder ────────────────────────────────────────────────────
def make_graph(obs, obs_norm, n, gb, device):
    obs_norm.update(obs[:n*2+6])
    on = obs_norm.normalize(obs[:n*2+6])
    jp = on[:n].astype(np.float32)
    jv = on[n:n*2].astype(np.float32)
    lv = on[n*2:n*2+3].astype(np.float32)
    av = on[n*2+3:n*2+6].astype(np.float32)
    bq = obs[n*2+6:n*2+10].astype(np.float32)
    bg = obs[n*2+10:n*2+13].astype(np.float32)
    graph = gb.get_graph(jp, jv, body_quat=bq, body_grav=bg,
                         body_lin_vel=lv, body_ang_vel=av,
                         command=np.array([0.7, 0.0], np.float32))
    return graph.to(device)


# ── Main fine-tune function ────────────────────────────────────────────────
def finetune(target: str, total_steps: int, stage1_steps: int,
             num_steps: int, save_dir: str, checkpoint: str):

    cfg    = TARGET_CONFIGS[target]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'Fine-tuning → {target.upper()} | device={device}')
    print(f'Total steps: {total_steps:,}  | Stage1 (head-only): {stage1_steps:,}')
    print(f'Rollout steps/update: {num_steps} | Save dir: {save_dir}')
    print(f'{"="*60}\n')

    # ── Load base checkpoint ──
    print(f'Loading base checkpoint: {checkpoint}')
    ckpt   = torch.load(checkpoint, map_location='cpu', weights_only=False)
    base_sd = ckpt['agent']
    q_mean  = ckpt.get('obs_norm_mean', np.zeros(30))
    q_var   = ckpt.get('obs_norm_var',  np.ones(30))
    print(f'Base checkpoint: step={ckpt.get("global_step",0):,}, '
          f'params={sum(p.numel() for p in [v for v in base_sd.values() if v.dim()>0]):,}')

    n = cfg['num_joints']

    # ── Build model ──
    agent = SlimHeteroGNNActorCritic(
        node_dim=28, edge_dim=4, hidden_dim=48, num_joints=n
    ).to(device)

    # Load weights, expanding log_std for non-12-DOF morphologies
    sd = {k: v.clone() for k, v in base_sd.items()}
    if n != 12:
        avg = sd['log_std'].mean().item()
        exp = torch.full((n,), avg)
        exp[:min(12, n)] = sd['log_std'][:min(12, n)]
        sd['log_std'] = exp
    agent.load_state_dict(sd)
    print(f'Model: {sum(p.numel() for p in agent.parameters()):,} parameters')

    # ── Transfer obs normaliser ──
    d = cfg['obs_norm_dim']
    obs_norm = RunningNorm(shape=(d,))
    copy_len = min(30, d)
    obs_norm.mean[:copy_len] = q_mean[:copy_len]
    obs_norm.var[:copy_len]  = q_var[:copy_len]

    # ── Build env and graph builder ──
    urdf = cfg['urdf']
    env  = RobotEnvBullet(urdf, render_mode=None,
                          height_threshold=cfg['height_threshold'],
                          max_episode_steps=1000)
    gb   = URDFGraphBuilder(urdf, add_body_node=True)
    print(f'Env: {n} joints | obs={env.obs_dim} | urdf={os.path.basename(urdf)}\n')

    # ── Param groups for staged training ──
    gnn_params    = (list(agent.type_proj.parameters()) +
                     list(agent.conv1.parameters()) + list(agent.norm1.parameters()) +
                     list(agent.conv2.parameters()) + list(agent.norm2.parameters()))
    actor_params  = list(agent.actor_head.parameters()) + [agent.log_std]
    critic_params = list(agent.critic_head.parameters())

    # Stage 1 base LRs (head only)
    STAGE1_LRS = [3e-4, 3e-4]
    # Stage 2 base LRs (full)
    STAGE2_LRS = [1e-4, 1.5e-4, 1.5e-4]

    def make_stage1_optimizer():
        for p in gnn_params: p.requires_grad_(False)
        return optim.Adam([
            {'params': actor_params,  'lr': STAGE1_LRS[0], '_base_lr': STAGE1_LRS[0]},
            {'params': critic_params, 'lr': STAGE1_LRS[1], '_base_lr': STAGE1_LRS[1]},
        ], eps=1e-5)

    def make_stage2_optimizer():
        for p in gnn_params: p.requires_grad_(True)
        return optim.Adam([
            {'params': gnn_params,    'lr': STAGE2_LRS[0], '_base_lr': STAGE2_LRS[0]},
            {'params': actor_params,  'lr': STAGE2_LRS[1], '_base_lr': STAGE2_LRS[1]},
            {'params': critic_params, 'lr': STAGE2_LRS[2], '_base_lr': STAGE2_LRS[2]},
        ], eps=1e-5)

    stage = 1
    optimizer = make_stage1_optimizer()
    print('Stage 1: head-only training (backbone frozen)...')

    # ── Training loop ──
    results = []
    buffer  = RolloutBuffer(num_steps, n, device)
    obs, _  = env.reset(seed=42)
    global_step = 0
    ep_reward, ep_len = 0.0, 0
    episode_rewards: List[float] = []
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)

    while global_step < total_steps:

        # Stage transition
        if stage == 1 and global_step >= stage1_steps:
            stage = 2
            optimizer = make_stage2_optimizer()
            print(f'\nStage 2: full fine-tuning (all params) at step {global_step:,}...')

        # Linear LR decay based on ABSOLUTE step position
        frac = max(1.0 - global_step / total_steps, 0.05)
        for pg in optimizer.param_groups:
            pg['lr'] = pg['_base_lr'] * frac

        # ── Rollout collection ──
        buffer.reset()
        for _ in range(num_steps):
            global_step += 1
            ep_len += 1

            graph = make_graph(obs, obs_norm, n, gb, device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(graph)

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done_flag = terminated or truncated
            ep_reward += reward
            buffer.store(graph, action, logprob, float(reward), float(done_flag), value)

            if done_flag:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_len    = 0
                obs, _ = env.reset()

        # Bootstrap value
        with torch.no_grad():
            next_val = agent.get_value(make_graph(obs, obs_norm, n, gb, device))

        adv, returns = buffer.compute_advantages(next_val, float(done_flag))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── PPO update (4 epochs, 4 minibatches) ──
        indices = np.arange(num_steps)
        mb_size = num_steps // 4
        for _ in range(4):
            np.random.shuffle(indices)
            for s in range(0, num_steps, mb_size):
                idx        = indices[s:s + mb_size]
                mb_batch   = Batch.from_data_list([buffer.graphs[i] for i in idx]).to(device)
                mb_actions = buffer.actions[idx]
                mb_adv     = adv[idx]
                mb_ret     = returns[idx]
                mb_lp      = buffer.logprobs[idx]

                _, new_lp, ent, new_val = agent.get_action_and_value(mb_batch, mb_actions)
                new_val = new_val.view(-1)
                ratio   = (new_lp - mb_lp).exp()
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - 0.15, 1 + 0.15)
                ).mean()
                vf_loss = ((new_val - mb_ret) ** 2).mean() * 0.5
                loss    = pg_loss - 0.003 * ent.mean() + 0.5 * vf_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # ── Logging ──
        if episode_rewards:
            m   = float(np.mean(episode_rewards[-20:]))
            sps = int(global_step / max(time.time() - start_time, 1))
            lr0 = optimizer.param_groups[0]['lr']
            results.append({'step': global_step, 'mean_reward': round(m, 2)})
            stg = '1-head' if stage == 1 else '2-full'
            print(f'[{stg}] step={global_step:>7,} | rew={m:>8.2f} | '
                  f'sps={sps:>5} | lr={lr0:.2e}')

        # Save checkpoint every 100K steps
        if global_step % 100_000 < num_steps:
            mid_path = os.path.join(save_dir, f'finetuned_{target}_{global_step}.pt')
            torch.save({
                'global_step':   global_step,
                'agent':         agent.state_dict(),
                'obs_norm_mean': obs_norm.mean,
                'obs_norm_var':  obs_norm.var,
                'target':        target,
                'results':       results,
            }, mid_path)
            print(f'  → Checkpoint: {mid_path}')

    env.close()

    # ── Final save ──
    final_path = os.path.join(save_dir, f'finetuned_{target}_final.pt')
    torch.save({
        'global_step':   global_step,
        'agent':         agent.state_dict(),
        'obs_norm_mean': obs_norm.mean,
        'obs_norm_var':  obs_norm.var,
        'target':        target,
        'results':       results,
    }, final_path)

    curve_path = os.path.join(save_dir, f'curve_{target}.json')
    with open(curve_path, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'DONE  target={target}  steps={global_step:,}  time={elapsed/60:.1f} min')
    print(f'Final checkpoint : {final_path}')
    print(f'Learning curve   : {curve_path}')
    print(f'{"="*60}')

    return results


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune GNN on a target morphology')
    parser.add_argument('--target',       default='hexapod',
                        choices=['hexapod', 'aliengo', 'go1'])
    parser.add_argument('--steps',        type=int, default=500_000)
    parser.add_argument('--stage1-steps', type=int, default=100_000,
                        help='Steps with backbone frozen')
    parser.add_argument('--num-steps',    type=int, default=4096,
                        help='Rollout buffer size per PPO update')
    parser.add_argument('--save-dir',     default='/kaggle/working/finetune',
                        help='Output directory (use /kaggle/working on Kaggle)')
    parser.add_argument('--checkpoint',   default=BASE_CHECKPOINT,
                        help='Path to seed2_final.pt')
    args = parser.parse_args()

    finetune(
        target      = args.target,
        total_steps = args.steps,
        stage1_steps= args.stage1_steps,
        num_steps   = args.num_steps,
        save_dir    = args.save_dir,
        checkpoint  = args.checkpoint,
    )

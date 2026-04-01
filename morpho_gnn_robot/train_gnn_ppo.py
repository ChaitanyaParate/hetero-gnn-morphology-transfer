"""
train_gnn_ppo.py

CleanRL-style PPO adapted for GNN graph observations.
Runs on Kaggle (T4/P100 GPU, headless, PyBullet DIRECT mode).

Key changes from CleanRL ppo_continuous_action.py:
  1. obs is a list of PyG Data objects, not a flat tensor
  2. Minibatch creation uses Batch.from_data_list()
  3. Graph is built from obs[:24] (joint pos/vel) at each rollout step

Usage:
  python train_gnn_ppo.py
  python train_gnn_ppo.py --seed 1 --total-timesteps 1000000

On Kaggle: set WANDB_API_KEY in Kaggle secrets and --track 1.
"""

import argparse
import os
import random
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch

from urdf_to_graph      import URDFGraphBuilder
from robot_env_bullet   import RobotEnvBullet
from gnn_actor_critic   import GNNActorCritic


# -----------------------------------------------------------------------
# Running normalizer (Welford online algorithm)
# -----------------------------------------------------------------------
class RunningNorm:
    """
    Normalizes inputs to zero mean, unit variance using a running estimate.
    Used for both observations and rewards.

    WHY this matters: joint_pos in [-1.2, 1.2] and joint_vel in [-20, 20]
    are on different scales. Without normalization the GNN receives poorly
    scaled inputs and the critic cannot learn returns in the -5000 range.
    """
    def __init__(self, shape, clip: float = 10.0):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4
        self.clip  = clip

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var  = np.var(x,  axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_n    = x.shape[0] if x.ndim > 1 else 1
        delta      = batch_mean - self.mean
        tot        = self.count + batch_n
        self.mean += delta * batch_n / tot
        self.var   = (self.var * self.count + batch_var * batch_n +
                      delta**2 * self.count * batch_n / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8),
                       -self.clip, self.clip).astype(np.float32)


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
@dataclass
class Config:
    # Environment
    urdf_path: str = (
        "/mnt/newvolume/Programming/Python/Deep_Learning/"
        "Relational_Bias_for_Morphological_Generalization/"
        "morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf/anymal.urdf"
    )
    max_episode_steps: int = 1000

    # Training
    seed:             int   = 0
    total_timesteps:  int   = 500_000
    learning_rate:    float = 1e-4
    num_steps:        int   = 2048    # rollout length before each update
    num_minibatches:  int   = 32
    update_epochs:    int   = 10
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_coef:        float = 0.2
    ent_coef:         float = 0.00005   # low: let pg signal dominate; entropy is already high at init
    vf_coef:          float = 1.0
    max_grad_norm:    float = 0.5
    clip_vloss:       bool  = True
    resume_path: str = None

    # GNN
    hidden_dim:  int = 64

    # Logging
    track:       bool = False    # set True on Kaggle with wandb key
    run_name:    str  = ""
    save_every:  int  = 50_000   # checkpoint every N timesteps
    checkpoint_dir: str = "./checkpoints"

    @property
    def minibatch_size(self):
        return self.num_steps // self.num_minibatches


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser()
    for f_name, f_val in cfg.__dict__.items():
        if f_name.startswith("_"):
            continue
        t = type(f_val) if f_val is not None else str
        if isinstance(f_val, bool):
            parser.add_argument(f"--{f_name.replace('_','-')}", type=int, default=int(f_val))
        else:
            parser.add_argument(f"--{f_name.replace('_','-')}", type=t, default=f_val)
    args = parser.parse_args()
    for f_name in cfg.__dict__:
        if f_name.startswith("_"):
            continue
        v = getattr(args, f_name.replace("-", "_"), None)
        if v is not None:
            if isinstance(getattr(cfg, f_name), bool):
                setattr(cfg, f_name, bool(v))
            else:
                setattr(cfg, f_name, v)
    if not cfg.run_name:
        cfg.run_name = f"gnn_ppo_seed{cfg.seed}_{int(time.time())}"
    return cfg


# -----------------------------------------------------------------------
# Rollout buffer (graphs stored as list, everything else as tensors)
# -----------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, num_steps: int, action_dim: int, device: torch.device):
        self.num_steps  = num_steps
        self.action_dim = action_dim
        self.device     = device
        self.reset()

    def reset(self):
        self.graphs:   List[Data] = []
        self.actions   = torch.zeros(self.num_steps, self.action_dim,  device=self.device)
        self.logprobs  = torch.zeros(self.num_steps,                   device=self.device)
        self.rewards   = torch.zeros(self.num_steps,                   device=self.device)
        self.dones     = torch.zeros(self.num_steps,                   device=self.device)
        self.values    = torch.zeros(self.num_steps,                   device=self.device)
        self.ptr       = 0

    def store(self, graph, action, logprob, reward, done, value):
        self.graphs.append(graph)
        self.actions[self.ptr]  = action.squeeze(0)
        self.logprobs[self.ptr] = logprob.squeeze()
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.values[self.ptr]   = value.squeeze()
        self.ptr += 1

    def compute_advantages(
        self, next_value: torch.Tensor, next_done: float,
        gamma: float, gae_lambda: float
    ):
        """GAE advantage estimation."""
        advantages = torch.zeros(self.num_steps, device=self.device)
        last_gae   = 0.0
        next_val   = next_value.squeeze()

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                non_terminal = 1.0 - next_done
                nv           = next_val
            else:
                non_terminal = 1.0 - self.dones[t + 1]
                nv           = self.values[t + 1]

            delta     = self.rewards[t] + gamma * nv * non_terminal - self.values[t]
            last_gae  = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------
def train(cfg: Config):
    # ---- seeding ----
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- wandb ----
    if cfg.track:
        import wandb
        wandb.init(project="morpho_gnn_robot", name=cfg.run_name, config=cfg.__dict__)

    # ---- env + builder ----
    env     = RobotEnvBullet(cfg.urdf_path, max_episode_steps=cfg.max_episode_steps)
    builder = URDFGraphBuilder(cfg.urdf_path, add_body_node=True)

    assert env.joint_names == builder.joint_names, (
        "Joint ordering mismatch between env and builder. This will corrupt your training."
    )

    # Normalize only joint states (dims 0-30: pos, vel, lin_vel, ang_vel).
    # Orientation dims 30-37 (quaternion + gravity vector) are NOT normalized.
    # Quaternions are unit vectors on SO(3) -- applying (q-mean)/std destroys
    # the geometric meaning and gives the GNN garbage instead of orientation.
    # Gravity vector is already in [-1,1] by construction.
    # Joint states need normalization: pos in [-1.2,1.2], vel in [-20,20].
    OBS_NORM_DIM = 30   # normalize only first 30 dims
    obs_norm = RunningNorm(shape=(OBS_NORM_DIM,), clip=10.0)

    # ---- agent ----
    agent = GNNActorCritic(
        node_dim   = builder.node_dim,
        edge_dim   = builder.edge_dim,
        hidden_dim = cfg.hidden_dim,
        num_joints = builder.action_dim,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    start_global_step = 0

    if cfg.resume_path is not None:
        print(f"\nLoading checkpoint: {cfg.resume_path}")
        checkpoint = torch.load(cfg.resume_path, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_global_step = checkpoint.get("global_step", 0)
        episode_rewards   = checkpoint.get("episode_rewards", [])
        if "obs_norm_mean" in checkpoint:
            obs_norm.mean  = checkpoint["obs_norm_mean"]
            obs_norm.var   = checkpoint["obs_norm_var"]
            obs_norm.count = checkpoint["obs_norm_count"]
        print(f"Resumed from step {start_global_step}")

    print(f"\nAgent parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"Rollout steps: {cfg.num_steps} | Minibatch size: {cfg.minibatch_size}")
    print(f"Total updates: {cfg.total_timesteps // cfg.num_steps}\n")

    # ---- rollout buffer ----
    buffer = RolloutBuffer(cfg.num_steps, builder.action_dim, device)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ---- initial reset ----
    obs, _ = env.reset(seed=cfg.seed)
    done   = False

    global_step      = start_global_step
    update           = 0
    episode_rewards: List[float] = []
    episode_lengths: List[int]   = []
    ep_reward        = 0.0
    ep_length        = 0

    start_time = time.time()

    # ================================================================
    # Main training loop
    # ================================================================
    target_timesteps = cfg.total_timesteps
    while global_step < target_timesteps:

        # ---- learning rate annealing ----
        frac = 1.0 - (global_step - start_global_step) / (cfg.total_timesteps - start_global_step)
        frac = max(frac, 0.0)
        optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # --------------------------------------------------------
        # Rollout collection
        # --------------------------------------------------------
        buffer.reset()

        for step in range(cfg.num_steps):
            global_step += 1
            ep_length   += 1

            # Normalize only joint states (dims 0-30); orientation dims are raw
            obs_norm.update(obs[:OBS_NORM_DIM])
            obs_n = obs_norm.normalize(obs[:OBS_NORM_DIM])

            joint_pos = obs_n[:12]
            joint_vel = obs_n[12:24]
            body_quat = obs[30:34].astype(np.float32)   # unit quaternion -- do not normalize
            body_grav = obs[34:37].astype(np.float32)   # unit vector -- do not normalize
            graph     = builder.get_graph(joint_pos, joint_vel, body_quat, body_grav).to(device)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(graph)

            # step env
            action_np             = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done    = terminated or truncated
            ep_reward += reward

            # store raw (unscaled) reward -- critic learns actual return scale
            buffer.store(graph.to("cpu"), action.cpu(), logprob.cpu(),
                         reward, float(done), value.cpu())

            if done:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                ep_reward  = 0.0
                ep_length  = 0
                obs, _     = env.reset()

        # next value for GAE bootstrap
        with torch.no_grad():
            obs_norm.update(obs[:OBS_NORM_DIM])
            obs_n       = obs_norm.normalize(obs[:OBS_NORM_DIM])
            joint_pos   = obs_n[:12]
            joint_vel   = obs_n[12:24]
            body_quat   = obs[30:34].astype(np.float32)
            body_grav   = obs[34:37].astype(np.float32)
            next_graph  = builder.get_graph(joint_pos, joint_vel, body_quat, body_grav).to(device)
            next_value  = agent.get_value(next_graph)

        advantages, returns = buffer.compute_advantages(
            next_value.cpu(), float(done), cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --------------------------------------------------------
        # PPO update
        # --------------------------------------------------------
        update += 1
        indices = np.arange(cfg.num_steps)

        pg_losses, vf_losses, ent_losses = [], [], []
        clip_fracs = []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, cfg.num_steps, cfg.minibatch_size):
                mb_idx = indices[start : start + cfg.minibatch_size]

                # batch graphs for this minibatch
                mb_batch    = Batch.from_data_list([buffer.graphs[i] for i in mb_idx]).to(device)
                mb_actions  = buffer.actions[mb_idx].to(device)
                mb_adv      = advantages[mb_idx].to(device)
                mb_returns  = returns[mb_idx].to(device)
                mb_logprobs = buffer.logprobs[mb_idx].to(device)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_batch, mb_actions
                )
                newvalue = newvalue.view(-1)

                logratio  = newlogprob - mb_logprobs
                ratio     = logratio.exp()

                # clip fraction diagnostic
                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                if cfg.clip_vloss:
                    mb_vals = buffer.values[mb_idx].to(device)
                    v_clipped = mb_vals + (newvalue - mb_vals).clamp(
                        -cfg.clip_coef, cfg.clip_coef
                    )
                    vf_loss = torch.max(
                        (newvalue   - mb_returns) ** 2,
                        (v_clipped  - mb_returns) ** 2,
                    ).mean() * 0.5
                else:
                    vf_loss = ((newvalue - mb_returns) ** 2).mean() * 0.5

                ent_loss = entropy.mean()
                loss     = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * vf_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------
        sps = int(global_step / (time.time() - start_time))

        if episode_rewards:
            mean_ep_rew = np.mean(episode_rewards[-20:])
            mean_ep_len = np.mean(episode_lengths[-20:])
        else:
            mean_ep_rew = 0.0
            mean_ep_len = 0.0

        print(
            f"step={global_step:>8d} | "
            f"ep_rew={mean_ep_rew:>8.2f} | "
            f"ep_len={mean_ep_len:>6.0f} | "
            f"pg={np.mean(pg_losses):>7.4f} | "
            f"vf={np.mean(vf_losses):>7.4f} | "
            f"ent={np.mean(ent_losses):>6.4f} | "
            f"clip={np.mean(clip_fracs):.3f} | "
            f"sps={sps}"
        )

        if cfg.track:
            import wandb
            wandb.log({
                "charts/ep_reward_mean":   mean_ep_rew,
                "charts/ep_length_mean":   mean_ep_len,
                "losses/policy_loss":      np.mean(pg_losses),
                "losses/value_loss":       np.mean(vf_losses),
                "losses/entropy":          np.mean(ent_losses),
                "charts/clip_frac":        np.mean(clip_fracs),
                "charts/sps":              sps,
                "charts/learning_rate":    optimizer.param_groups[0]["lr"],
            }, step=global_step)

        # checkpoint
        if global_step % cfg.save_every < cfg.num_steps:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"gnn_ppo_{global_step}.pt")
            torch.save({
                "global_step":   global_step,
                "agent":         agent.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "episode_rewards": episode_rewards,
                "obs_norm_mean": obs_norm.mean,
                "obs_norm_var":  obs_norm.var,
                "obs_norm_count": obs_norm.count,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # ---- final save ----
    final_path = os.path.join(cfg.checkpoint_dir, "gnn_ppo_final.pt")
    torch.save({"global_step": global_step, "agent": agent.state_dict()}, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    env.close()
    if cfg.track:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

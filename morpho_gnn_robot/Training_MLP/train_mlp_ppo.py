import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core')))
from robot_env_bullet import RobotEnvBullet
from mlp_actor_critic import MLPActorCritic

class RunningNorm:

    def __init__(self, shape, clip: float=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0.0001
        self.clip = clip

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = np.var(x, axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_n = x.shape[0] if x.ndim > 1 else 1
        delta = batch_mean - self.mean
        total = self.count + batch_n
        self.mean += delta * batch_n / total
        self.var = (self.var * self.count + batch_var * batch_n + delta ** 2 * self.count * batch_n / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / (np.sqrt(self.var) + 1e-08)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

@dataclass
class Config:
    urdf_path: str = 'auto'
    max_episode_steps: int = 1000
    seed: int = 0
    total_timesteps: int = 12000000
    mlp_learning_rate: float = 0.00068
    actor_learning_rate: float = 0.00045
    critic_learning_rate: float = 0.00045
    num_steps: int = 4096
    num_minibatches: int = 4
    update_epochs: int = 6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.15
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_vloss: bool = True
    target_kl: float = 0.015
    resume_path: str = None
    resume_optimizer: bool = True
    hidden_dim: int = 256
    track: bool = False
    run_name: str = ''
    save_every: int = 70000
    checkpoint_dir: str = './checkpoints'

    @property
    def minibatch_size(self):
        return self.num_steps // self.num_minibatches

def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser()
    for f_name, f_val in cfg.__dict__.items():
        if f_name.startswith('_'):
            continue
        t = type(f_val) if f_val is not None else str
        if isinstance(f_val, bool):
            parser.add_argument(f'--{f_name.replace('_', '-')}', type=int, default=int(f_val))
        else:
            parser.add_argument(f'--{f_name.replace('_', '-')}', type=t, default=f_val)
    args = parser.parse_args()
    for f_name in cfg.__dict__:
        if f_name.startswith('_'):
            continue
        v = getattr(args, f_name.replace('-', '_'), None)
        if v is not None:
            if isinstance(getattr(cfg, f_name), bool):
                setattr(cfg, f_name, bool(v))
            else:
                setattr(cfg, f_name, v)
    if not cfg.run_name:
        cfg.run_name = f'mlp_ppo_seed{cfg.seed}_{int(time.time())}'
    if cfg.urdf_path == 'auto':
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Priority: local stripped URDF -> kaggle working dir -> ros2 ws
        for candidate in [
            os.path.join(base_dir, '..', 'URDFs', 'anymal_stripped.urdf'),
            os.path.join(base_dir, '..', 'URDFs', 'anymal.urdf'),
            '/kaggle/working/anymal.urdf',
            os.path.join(base_dir, '..', 'morpho_ros2_ws', 'src', 'morpho_robot', 'urdf', 'anymal.urdf'),
        ]:
            if os.path.exists(candidate):
                cfg.urdf_path = os.path.abspath(candidate)
                break
        else:
            raise FileNotFoundError('Could not find anymal.urdf. Pass --urdf-path explicitly.')
    return cfg

class RolloutBuffer:

    def __init__(self, num_steps: int, obs_dim: int, action_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        self.obs = torch.zeros(self.num_steps, self.obs_dim, device=self.device)
        self.actions = torch.zeros(self.num_steps, self.action_dim, device=self.device)
        self.logprobs = torch.zeros(self.num_steps, device=self.device)
        self.rewards = torch.zeros(self.num_steps, device=self.device)
        self.dones = torch.zeros(self.num_steps, device=self.device)
        self.values = torch.zeros(self.num_steps, device=self.device)
        self.ptr = 0

    def store(self, obs, action, logprob, reward, done, value):
        self.obs[self.ptr] = obs.squeeze(0)
        self.actions[self.ptr] = action.squeeze(0)
        self.logprobs[self.ptr] = logprob.squeeze()
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.squeeze()
        self.ptr += 1

    def compute_advantages(self, next_value: torch.Tensor, next_done: float, gamma: float, gae_lambda: float):
        advantages = torch.zeros(self.num_steps, device=self.device)
        last_gae = 0.0
        next_val = next_value.squeeze()
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                non_terminal = 1.0 - next_done
                nv = next_val
            else:
                non_terminal = 1.0 - self.dones[t + 1]
                nv = self.values[t + 1]
            delta = self.rewards[t] + gamma * nv * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return (advantages, returns)

def _policy_obs(obs: np.ndarray, obs_norm: RunningNorm, obs_norm_dim: int) -> np.ndarray:
    obs_norm.update(obs[:obs_norm_dim])
    normed = obs_norm.normalize(obs[:obs_norm_dim])
    return np.concatenate([normed, obs[obs_norm_dim:].astype(np.float32)], axis=0).astype(np.float32)

def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if cfg.track:
        import wandb
        wandb.init(project='morpho_mlp_robot', name=cfg.run_name, config=cfg.__dict__)
    env = RobotEnvBullet(cfg.urdf_path, max_episode_steps=cfg.max_episode_steps)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    obs_norm_dim = 30
    obs_norm = RunningNorm(shape=(obs_norm_dim,), clip=10.0)
    agent = MLPActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=cfg.hidden_dim).to(device)
    trunk_params = list(agent.trunk.parameters())
    actor_params = list(agent.actor_head.parameters()) + [agent.log_std]
    critic_params = list(agent.critic_head.parameters())
    optimizer = optim.Adam([{'params': trunk_params, 'lr': cfg.mlp_learning_rate}, {'params': actor_params, 'lr': cfg.actor_learning_rate}, {'params': critic_params, 'lr': cfg.critic_learning_rate}], eps=1e-05)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    start_global_step = 0
    episode_rewards: List[float] = []
    if cfg.resume_path is not None:
        print(f'\nLoading checkpoint: {cfg.resume_path}')
        checkpoint = torch.load(cfg.resume_path, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint['agent'])
        if cfg.resume_optimizer and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                base_lrs = [cfg.mlp_learning_rate, cfg.actor_learning_rate, cfg.critic_learning_rate]
                if len(optimizer.param_groups) == len(base_lrs):
                    for lr, pg in zip(base_lrs, optimizer.param_groups):
                        pg['lr'] = lr
            except Exception as exc:
                print(f'Warning: could not load optimizer state ({exc}). Using fresh optimizer.')
        elif not cfg.resume_optimizer:
            print('Resume: skipping optimizer state per resume_optimizer=0')
        start_global_step = checkpoint.get('global_step', 0)
        episode_rewards = checkpoint.get('episode_rewards', [])
        if 'obs_norm_mean' in checkpoint:
            obs_norm.mean = checkpoint['obs_norm_mean']
            obs_norm.var = checkpoint['obs_norm_var']
            obs_norm.count = checkpoint['obs_norm_count']
        print(f'Resumed from step {start_global_step}')
    print(f'\nAgent parameters: {sum((p.numel() for p in agent.parameters())):,}')
    print(f'Obs dim: {obs_dim} | Action dim: {action_dim}')
    print(f'Rollout steps: {cfg.num_steps} | Minibatch size: {cfg.minibatch_size}')
    print(f'Total updates: {cfg.total_timesteps // cfg.num_steps}\n')
    buffer = RolloutBuffer(cfg.num_steps, obs_dim, action_dim, device)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    obs, _ = env.reset(seed=cfg.seed)
    done = False
    global_step = start_global_step
    episode_lengths: List[int] = []
    episode_forward_vels: List[float] = []
    episode_cmd_lin_errors: List[float] = []
    episode_cmd_ang_errors: List[float] = []
    ep_reward = 0.0
    ep_length = 0
    ep_forward_vel_sum = 0.0
    ep_cmd_lin_error_sum = 0.0
    ep_cmd_ang_error_sum = 0.0
    start_time = time.time()
    target_timesteps = cfg.total_timesteps
    while global_step < target_timesteps:
        frac = 1.0 - (global_step - start_global_step) / (cfg.total_timesteps - start_global_step)
        frac = max(frac, 0.0)
        for base_lr, pg in zip(base_lrs, optimizer.param_groups):
            pg['lr'] = frac * base_lr
        buffer.reset()
        for _ in range(cfg.num_steps):
            global_step += 1
            ep_length += 1
            obs_in = _policy_obs(obs, obs_norm, obs_norm_dim)
            obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward
            ep_forward_vel_sum += float(obs[24 + env.forward_axis])
            ep_cmd_lin_error_sum += abs(float(obs[24 + env.forward_axis]) - float(obs[37]))
            ep_cmd_ang_error_sum += abs(float(obs[29]) - float(obs[38]))
            buffer.store(obs_t, action, logprob, reward, float(done), value)
            if done:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                episode_forward_vels.append(ep_forward_vel_sum / max(ep_length, 1))
                episode_cmd_lin_errors.append(ep_cmd_lin_error_sum / max(ep_length, 1))
                episode_cmd_ang_errors.append(ep_cmd_ang_error_sum / max(ep_length, 1))
                ep_reward = 0.0
                ep_length = 0
                ep_forward_vel_sum = 0.0
                ep_cmd_lin_error_sum = 0.0
                ep_cmd_ang_error_sum = 0.0
                obs, _ = env.reset()
        with torch.no_grad():
            next_obs_in = _policy_obs(obs, obs_norm, obs_norm_dim)
            next_obs_t = torch.tensor(next_obs_in, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = agent.get_value(next_obs_t)
        advantages, returns = buffer.compute_advantages(next_value, float(done), cfg.gamma, cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        indices = np.arange(cfg.num_steps)
        pg_losses, vf_losses, ent_losses = ([], [], [])
        clip_fracs, approx_kls = ([], [])
        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            early_stop = False
            for start in range(0, cfg.num_steps, cfg.minibatch_size):
                mb_idx = indices[start:start + cfg.minibatch_size]
                mb_obs = buffer.obs[mb_idx]
                mb_actions = buffer.actions[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_logprobs = buffer.logprobs[mb_idx]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                newvalue = newvalue.view(-1)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                with torch.no_grad():
                    clip_fracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                    approx_kls.append((ratio - 1.0 - logratio).mean().item())
                if cfg.target_kl and np.mean(approx_kls) > cfg.target_kl:
                    early_stop = True
                    break
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                if cfg.clip_vloss:
                    mb_vals = buffer.values[mb_idx]
                    v_clipped = mb_vals + (newvalue - mb_vals).clamp(-cfg.clip_coef, cfg.clip_coef)
                    vf_loss = torch.max((newvalue - mb_returns) ** 2, (v_clipped - mb_returns) ** 2).mean() * 0.5
                else:
                    vf_loss = ((newvalue - mb_returns) ** 2).mean() * 0.5
                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * vf_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
            if early_stop:
                break
        y_pred = buffer.values.detach().cpu().numpy()
        y_true = returns.detach().cpu().numpy()
        y_var = np.var(y_true)
        explained_var = np.nan if y_var == 0 else 1.0 - np.var(y_true - y_pred) / y_var
        elapsed = max(time.time() - start_time, 1e-08)
        run_steps = global_step - start_global_step
        sps = int(run_steps / elapsed)
        if episode_rewards:
            mean_ep_rew = np.mean(episode_rewards[-20:])
            mean_ep_len = np.mean(episode_lengths[-20:])
            mean_ep_fwd = np.mean(episode_forward_vels[-20:])
            mean_ep_lin_err = np.mean(episode_cmd_lin_errors[-20:])
            mean_ep_ang_err = np.mean(episode_cmd_ang_errors[-20:])
        else:
            mean_ep_rew = 0.0
            mean_ep_len = 0.0
            mean_ep_fwd = 0.0
            mean_ep_lin_err = 0.0
            mean_ep_ang_err = 0.0
        print(f'step={global_step:>8d} | ep_rew={mean_ep_rew:>8.2f} | ep_len={mean_ep_len:>6.0f} | cmd_lin_err={mean_ep_lin_err:>6.3f} | cmd_ang_err={mean_ep_ang_err:>6.3f} | pg={np.mean(pg_losses):>7.4f} | vf={np.mean(vf_losses):>7.4f} | ent={np.mean(ent_losses):>6.4f} | clip={np.mean(clip_fracs):.3f} | kl={np.mean(approx_kls):.5f} | ev={explained_var:>6.3f} | lr_m={optimizer.param_groups[0]["lr"]:.2e} | sps={sps}')
        if cfg.track:
            import wandb
            wandb.log({'charts/ep_reward_mean': mean_ep_rew, 'charts/ep_length_mean': mean_ep_len, 'charts/ep_cmd_lin_error_mean': mean_ep_lin_err, 'charts/ep_cmd_ang_error_mean': mean_ep_ang_err, 'losses/policy_loss': np.mean(pg_losses), 'losses/value_loss': np.mean(vf_losses), 'losses/entropy': np.mean(ent_losses), 'charts/clip_frac': np.mean(clip_fracs), 'losses/approx_kl': np.mean(approx_kls), 'losses/explained_variance': explained_var, 'charts/sps': sps, 'charts/learning_rate_mlp': optimizer.param_groups[0]['lr']}, step=global_step)
        if global_step % cfg.save_every < cfg.num_steps:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f'mlp_ppo_{global_step}.pt')
            torch.save({'global_step': global_step, 'agent': agent.state_dict(), 'optimizer': optimizer.state_dict(), 'episode_rewards': episode_rewards, 'obs_norm_mean': obs_norm.mean, 'obs_norm_var': obs_norm.var, 'obs_norm_count': obs_norm.count}, ckpt_path)
            print(f'  Checkpoint saved: {ckpt_path}')
    final_path = os.path.join(cfg.checkpoint_dir, 'mlp_ppo_final.pt')
    torch.save({'global_step': global_step, 'agent': agent.state_dict(), 'obs_norm_mean': obs_norm.mean, 'obs_norm_var': obs_norm.var, 'obs_norm_count': obs_norm.count}, final_path)
    print(f'\nTraining complete. Final checkpoint: {final_path}')
    env.close()
    if cfg.track:
        import wandb
        wandb.finish()
if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)
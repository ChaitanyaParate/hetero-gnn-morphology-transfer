"""
eval_comprehensive.py
Full evaluation suite covering:
  A) Zero-shot transfer on all morphologies (with fixed spawn pose)
  B) Terrain robustness: flat vs slope vs uneven ground (quadruped)
  C) Fine-tuned checkpoint evaluation (if available)

Outputs:
  eval_comprehensive_results.json  — machine-readable results
  eval_comprehensive_summary.txt   — human-readable for paper

Usage:
  python eval_comprehensive.py              # full suite
  python eval_comprehensive.py --only terrain  # only terrain test
  python eval_comprehensive.py --only transfer # only zero-shot
  python eval_comprehensive.py --finetuned hexapod:path/to/ckpt.pt
"""

import sys, os, argparse, json, time, statistics
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core')))
from robot_env_bullet import RobotEnvBullet
from gnn_actor_critic import SlimHeteroGNNActorCritic
from urdf_to_graph import URDFGraphBuilder

# ─── Checkpoint config ────────────────────────────────────────────────────
BASE_CKPT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../../training_with_different_seed_notebooks/seed2_final.pt'))

# ─── Robot configs ─────────────────────────────────────────────────────────
ROBOTS = {
    'anymal_quad': {
        'urdf':             '../URDFs/anymal.urdf',
        'height_threshold': 0.25,
        'label':            'ANYmal Quadruped (training)',
    },
    'anymal_hex': {
        'urdf':             '../URDFs/hexapod_anymal.urdf',
        'height_threshold': 0.15,
        'label':            'ANYmal Hexapod (zero-shot)',
    },
    'aliengo': {
        'urdf':             '../URDFs/aliengo_stripped.urdf',
        'height_threshold': 0.22,
        'label':            'Unitree Aliengo (zero-shot)',
    },
    'go1': {
        'urdf':             '../URDFs/go1_stripped.urdf',
        'height_threshold': 0.18,
        'label':            'Unitree Go1 (zero-shot)',
    },
    # Aliases used by finetune_transfer.py target names
    'hexapod': {
        'urdf':             '../URDFs/hexapod_anymal.urdf',
        'height_threshold': 0.15,
        'label':            'ANYmal Hexapod (fine-tuned)',
    },
}

TERRAIN_TESTS = [
    {'name': 'flat',          'terrain': 'flat',   'slope_angle': 0.0,  'height_noise': 0.0},
    {'name': 'slope_5deg',    'terrain': 'slope',  'slope_angle': 0.087,'height_noise': 0.0},   # ~5°
    {'name': 'slope_10deg',   'terrain': 'slope',  'slope_angle': 0.175,'height_noise': 0.0},   # ~10°
    {'name': 'uneven_2cm',    'terrain': 'uneven', 'slope_angle': 0.0,  'height_noise': 0.02},
    {'name': 'uneven_4cm',    'terrain': 'uneven', 'slope_angle': 0.0,  'height_noise': 0.04},
]


def load_base_model(ckpt_path, n_joints):
    """Load seed2_final.pt and expand for n_joints."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = {k: v.clone() for k, v in ckpt['agent'].items()}
    q_mean = ckpt.get('obs_norm_mean', np.zeros(30))
    q_var  = ckpt.get('obs_norm_var',  np.ones(30))
    if n_joints != 12:
        avg = sd['log_std'].mean().item()
        exp = torch.full((n_joints,), avg)
        exp[:12] = sd['log_std']
        sd['log_std'] = exp
    model = SlimHeteroGNNActorCritic(node_dim=28, edge_dim=4, hidden_dim=48, num_joints=n_joints)
    model.load_state_dict(sd)
    model.eval()
    return model, q_mean, q_var


def build_obs_norm(n, q_mean, q_var):
    """Build normaliser adapted to n-joint morphology."""
    dim = n * 2 + 6
    h_mean = np.zeros(dim); h_var = np.ones(dim)
    for i in range(n):
        s = i % 12
        h_mean[i]     = q_mean[s]; h_var[i]     = q_var[s]
        h_mean[n + i] = q_mean[12 + s]; h_var[n + i] = q_var[12 + s]
    h_mean[2*n:] = q_mean[24:30]; h_var[2*n:] = q_var[24:30]
    def norm(x): return np.clip((x[:dim] - h_mean) / (np.sqrt(h_var) + 1e-8), -10, 10)
    return norm


def run_episodes(model, env, gb, norm_fn, n, n_episodes=20, seed_base=42):
    """Run n_episodes and return stats."""
    rng = np.random.default_rng(seed_base)
    rewards, lengths, successes, fall_steps = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        ep_r, ep_l, fell = 0.0, 0, False
        fell_at = None
        while True:
            on = norm_fn(obs).astype(np.float32)
            jp = on[:n]; jv = on[n:n*2]
            lv = on[n*2:n*2+3]; av = on[n*2+3:n*2+6]
            bq = obs[n*2+6:n*2+10].astype(np.float32)
            bg = obs[n*2+10:n*2+13].astype(np.float32)
            pyg = gb.get_graph(jp, jv, body_quat=bq, body_grav=bg,
                               body_lin_vel=lv, body_ang_vel=av,
                               command=np.array([0.7, 0.0], np.float32))
            with torch.no_grad():
                h, _ = model._encode(pyg)
                jh   = model._joint_embeddings(h, pyg)
                act  = model.actor_head(jh).view(1, n)[0].numpy()

            obs, r, term, trunc, info = env.step(act)
            ep_r += r; ep_l += 1

            if term:
                reason = info.get('term_reason', '')
                fell   = reason in ('height', 'contact', 'orientation')
                if fell: fell_at = ep_l
                break
            if trunc:
                break

        rewards.append(ep_r)
        lengths.append(ep_l)
        successes.append(not fell)
        if fell_at: fall_steps.append(fell_at)

    return {
        'mean':         round(statistics.mean(rewards), 2),
        'std':          round(statistics.stdev(rewards) if len(rewards) > 1 else 0.0, 2),
        'mean_steps':   round(statistics.mean(lengths), 1),
        'success_rate': round(sum(successes) / len(successes) * 100, 1),
        'n_episodes':   n_episodes,
        'mean_fall_step': round(statistics.mean(fall_steps), 1) if fall_steps else None,
    }


# ─── Evaluation routines ──────────────────────────────────────────────────
def eval_transfer(ckpt_path, n_episodes=20):
    """Zero-shot transfer evaluation across all morphologies."""
    results = {}
    for key, cfg in ROBOTS.items():
        print(f'\n─── {cfg["label"]} ───')
        urdf = cfg['urdf']
        gb   = URDFGraphBuilder(urdf, add_body_node=True)
        n    = gb.action_dim
        model, q_mean, q_var = load_base_model(ckpt_path, n)
        norm_fn = build_obs_norm(n, q_mean, q_var)
        env = RobotEnvBullet(urdf, render_mode=None,
                             height_threshold=cfg['height_threshold'],
                             max_episode_steps=1000)
        stats = run_episodes(model, env, gb, norm_fn, n, n_episodes)
        env.close()
        results[key] = {'label': cfg['label'], **stats}
        print(f'  mean={stats["mean"]} ± {stats["std"]} | '
              f'steps={stats["mean_steps"]} | success={stats["success_rate"]}%')
    return results


def eval_terrain(ckpt_path, n_episodes=20):
    """Terrain robustness evaluation on the ANYmal quadruped."""
    urdf = '../URDFs/anymal.urdf'
    gb   = URDFGraphBuilder(urdf, add_body_node=True)
    n    = gb.action_dim
    model, q_mean, q_var = load_base_model(ckpt_path, n)
    norm_fn = build_obs_norm(n, q_mean, q_var)

    results = {}
    for tcfg in TERRAIN_TESTS:
        print(f'\n─── Terrain: {tcfg["name"]} ───')
        env = RobotEnvBullet(urdf, render_mode=None,
                             height_threshold=0.25,
                             max_episode_steps=1000,
                             terrain=tcfg['terrain'],
                             slope_angle=tcfg['slope_angle'],
                             height_noise_scale=tcfg['height_noise'])
        stats = run_episodes(model, env, gb, norm_fn, n, n_episodes)
        env.close()
        results[tcfg['name']] = stats
        print(f'  mean={stats["mean"]} ± {stats["std"]} | '
              f'steps={stats["mean_steps"]} | success={stats["success_rate"]}%')
    return results


def eval_finetuned(ckpt_path, morphology, n_episodes=20):
    """Evaluate a fine-tuned checkpoint."""
    cfg  = ROBOTS[morphology]
    urdf = cfg['urdf']
    gb   = URDFGraphBuilder(urdf, add_body_node=True)
    n    = gb.action_dim

    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SlimHeteroGNNActorCritic(node_dim=28, edge_dim=4, hidden_dim=48, num_joints=n)
    model.load_state_dict(ckpt['agent'])
    model.eval()

    # Use the normaliser that was built during fine-tuning
    raw_mean = ckpt.get('obs_norm_mean', np.zeros(n*2+6))
    raw_var  = ckpt.get('obs_norm_var',  np.ones(n*2+6))
    # build_obs_norm expects a 30-dim base (from 12-joint quadruped)
    base_mean = raw_mean[:30] if len(raw_mean) >= 30 else np.pad(raw_mean, (0, 30 - len(raw_mean)), constant_values=0)
    base_var  = raw_var[:30]  if len(raw_var)  >= 30 else np.pad(raw_var,  (0, 30 - len(raw_var)),  constant_values=1)
    norm_fn = build_obs_norm(n, base_mean, base_var)

    label = cfg.get('label', morphology)
    print(f'\n─── Fine-tuned {label} (step={ckpt.get("global_step",0):,}) ───')
    env = RobotEnvBullet(urdf, render_mode=None,
                         height_threshold=cfg['height_threshold'],
                         max_episode_steps=1000)
    stats = run_episodes(model, env, gb, norm_fn, n, n_episodes)
    env.close()
    print(f'  mean={stats["mean"]} ± {stats["std"]} | '
          f'steps={stats["mean_steps"]} | success={stats["success_rate"]}%')
    return stats


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only',      choices=['transfer', 'terrain', 'all'],
                        default='all')
    parser.add_argument('--episodes',  type=int, default=20)
    parser.add_argument('--finetuned', nargs='+', default=[],
                        help='morphology:path pairs e.g. hexapod:./finetune_checkpoints/finetuned_hexapod_500000.pt')
    parser.add_argument('--ckpt',      default=BASE_CKPT,
                        help='Base checkpoint path')
    args = parser.parse_args()

    all_results = {}

    if args.only in ('transfer', 'all'):
        print('\n' + '='*60)
        print('PHASE A: Zero-Shot Transfer Evaluation')
        print('='*60)
        all_results['transfer'] = eval_transfer(args.ckpt, args.episodes)

    if args.only in ('terrain', 'all'):
        print('\n' + '='*60)
        print('PHASE C: Terrain Robustness Evaluation')
        print('='*60)
        all_results['terrain'] = eval_terrain(args.ckpt, args.episodes)

    # Fine-tuned checkpoints
    for spec in args.finetuned:
        morphology, ckpt_path = spec.split(':', 1)
        stats = eval_finetuned(ckpt_path, morphology, args.episodes)
        all_results.setdefault('finetuned', {})[morphology] = stats

    # Save
    out_path = os.path.join(os.path.dirname(__file__), 'eval_comprehensive_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n✓ Saved: {out_path}')

    # Print summary table
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    if 'transfer' in all_results:
        print('\nZero-Shot Transfer:')
        print(f'  {"Robot":<35} {"Mean Rew":>10} {"±Std":>8} {"Steps":>8} {"Success":>8}')
        print('  ' + '-'*65)
        for key, s in all_results['transfer'].items():
            print(f'  {s["label"]:<35} {s["mean"]:>10.2f} {s["std"]:>8.2f} '
                  f'{s["mean_steps"]:>8.1f} {s["success_rate"]:>7.1f}%')

    if 'terrain' in all_results:
        print('\nTerrain Robustness (ANYmal Quadruped):')
        print(f'  {"Terrain":<20} {"Mean Rew":>10} {"±Std":>8} {"Steps":>8} {"Success":>8}')
        print('  ' + '-'*55)
        for name, s in all_results['terrain'].items():
            print(f'  {name:<20} {s["mean"]:>10.2f} {s["std"]:>8.2f} '
                  f'{s["mean_steps"]:>8.1f} {s["success_rate"]:>7.1f}%')

    if 'finetuned' in all_results:
        print('\nFine-tuned Results:')
        for morphology, s in all_results['finetuned'].items():
            print(f'  {morphology}: {s["mean"]} ± {s["std"]}, success={s["success_rate"]}%')

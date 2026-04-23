from robot_env_bullet import RobotEnvBullet
from urdf_to_graph import URDFGraphBuilder
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anymal_stripped.urdf')
builder = URDFGraphBuilder(URDF, add_body_node=True)
env = RobotEnvBullet(URDF, max_episode_steps=800)
obs, _ = env.reset()
assert env.joint_names == builder.joint_names, f'Joint mismatch!\n  env:     {env.joint_names}\n  builder: {builder.joint_names}'
print('joint_names: MATCH')
print(f'foot_indices:  {env.foot_indices}')
print(f'height_thresh: {env.height_threshold}')
print(f'action_scale:  {env._action_scale}')
print()
print('=== Zero-action (standing still) ===')
for i in range(5):
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_dim))
    fwd_vel = float(obs[24 + env.forward_axis])
    print(f'step {i}: reward={reward:.4f}  fwd_vel={fwd_vel:.4f}  height={info['base_height']:.3f}  term={terminated}')
    if terminated or truncated:
        print(f'  EARLY TERMINATION: {info['term_reason']}')
        break
print()
obs, _ = env.reset()
print('=== Random actions ===')
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    fwd_vel = float(obs[24 + env.forward_axis])
    act_mag = float(np.mean(np.abs(action)))
    print(f'step {i}: reward={reward:.4f}  fwd_vel={fwd_vel:.4f}  |action|={act_mag:.3f}  term={terminated}')
    if terminated or truncated:
        print(f'  EARLY TERMINATION: {info['term_reason']}')
        break
env.close()
print('\nSmoke test done.')
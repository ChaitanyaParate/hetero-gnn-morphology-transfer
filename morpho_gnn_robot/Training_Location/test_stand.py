import numpy as np
from robot_env_bullet import RobotEnvBullet

env = RobotEnvBullet('anymal_stripped.urdf', render_mode=None, height_threshold=0.22)
obs, _ = env.reset()
for step in range(100):
    action = np.zeros(env.action_dim)
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        print(f"Step {step} terminated! Reason: {info.get('term_reason')}, Height: {info.get('base_height'):.3f}")
        break
print("Done")

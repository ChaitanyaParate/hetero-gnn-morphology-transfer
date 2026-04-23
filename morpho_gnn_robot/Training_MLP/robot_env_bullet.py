import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mlp_actor_critic import MLPActorCritic
from scipy.spatial.transform import Rotation as R
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError('pip install pybullet')
CONTROLLABLE = {'revolute', 'continuous', 'prismatic'}
KP = 150.0
KD = 5.0
PLANE_LATERAL_FRICTION = 1.3
PLANE_SPINNING_FRICTION = 0.05
PLANE_ROLLING_FRICTION = 0.001
LINK_LATERAL_FRICTION = 1.3
LINK_SPINNING_FRICTION = 0.03
LINK_ROLLING_FRICTION = 0.001
NOMINAL_POSE_PER_JOINT = {'LF_HAA': 0.0, 'LF_HFE': 0.6, 'LF_KFE': -1.2, 'RF_HAA': 0.0, 'RF_HFE': 0.6, 'RF_KFE': -1.2, 'LM_HAA': 0.0, 'LM_HFE': 0.6, 'LM_KFE': -1.2, 'RM_HAA': 0.0, 'RM_HFE': 0.6, 'RM_KFE': -1.2, 'LH_HAA': 0.0, 'LH_HFE': -0.6, 'LH_KFE': 1.2, 'RH_HAA': 0.0, 'RH_HFE': -0.6, 'RH_KFE': 1.2}

class RobotEnvBullet(gym.Env):
    metadata = {'render_modes': ['human', 'direct']}
    FALL_PENALTY = 250.0

    def __init__(self, urdf_path: str, max_episode_steps: int=1000, render_mode: str=None, forward_axis: int=0):
        super().__init__()
        self.urdf_path = urdf_path
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.forward_axis = forward_axis
        self._parse_urdf()
        self.action_dim = len(self.joint_names)
        self.obs_dim = self.action_dim * 2 + 13
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.obs_noise_scale = 0.01
        self.vel_noise_scale = 0.02
        self.action_lag_prob = 0.5
        self.action_history = []
        if render_mode == 'human':
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)
            p.setPhysicsEngineParameter(enableFileCaching=0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._robot_id = None
        self._step_count = 0
        self._fell = False
        self.prev_action = np.zeros(self.action_dim)

    def _parse_urdf(self):
        import xml.etree.ElementTree as ET
        root = ET.parse(self.urdf_path).getroot()
        names: list = []
        efforts: dict = {}
        for joint in root.findall('joint'):
            jtype = joint.attrib.get('type', 'fixed')
            if jtype not in CONTROLLABLE:
                continue
            name = joint.attrib['name']
            names.append(name)
            lim = joint.find('limit')
            efforts[name] = float(lim.attrib.get('effort', 40.0)) if lim is not None else 40.0
        self.joint_names = sorted(names)
        self.effort_limits = np.array([efforts[n] for n in self.joint_names])

    def _load_robot(self):
        if self._robot_id is not None:
            p.removeBody(self._robot_id)
        start_pos = [0.0, 0.0, 0.65]
        start_orient = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        robot_id = p.loadURDF(self.urdf_path, start_pos, start_orient, useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)
        self._robot_id = robot_id
        num_joints = p.getNumJoints(robot_id)
        self._joint_idx: dict = {}
        self.foot_indices = []
        self.heavy_indices = []
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            if '_FOOT' in link_name:
                self.foot_indices.append(i)
            elif 'HIP' in link_name or 'THIGH' in link_name:
                self.heavy_indices.append(i)
            if name in self.joint_names:
                self._joint_idx[name] = i
                p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)
        assert len(self._joint_idx) == len(self.joint_names), f'URDF has {len(self.joint_names)} controllable joints but PyBullet found {len(self._joint_idx)}. Check URDF path.'
        self._pybullet_indices = np.array([self._joint_idx[n] for n in self.joint_names], dtype=np.int32)
        self._nominal_pos = np.array([NOMINAL_POSE_PER_JOINT.get(name, 0.0) for name in self.joint_names])
        for name, idx in self._joint_idx.items():
            val = NOMINAL_POSE_PER_JOINT.get(name, 0.0)
            p.resetJointState(self._robot_id, int(idx), val)
        for i in range(p.getNumJoints(self._robot_id)):
            p.changeDynamics(self._robot_id, i, linearDamping=0.04, angularDamping=0.04, lateralFriction=LINK_LATERAL_FRICTION, spinningFriction=LINK_SPINNING_FRICTION, rollingFriction=LINK_ROLLING_FRICTION)
        p.changeDynamics(self._robot_id, -1, linearDamping=0.04, angularDamping=0.04, lateralFriction=LINK_LATERAL_FRICTION, spinningFriction=LINK_SPINNING_FRICTION, rollingFriction=LINK_ROLLING_FRICTION)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        self._robot_id = None
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 400.0)
        p.setPhysicsEngineParameter(numSolverIterations=50, numSubSteps=1)
        plane_id = p.loadURDF('plane.urdf')
        p.changeDynamics(plane_id, -1, lateralFriction=PLANE_LATERAL_FRICTION, spinningFriction=PLANE_SPINNING_FRICTION, rollingFriction=PLANE_ROLLING_FRICTION)
        self._load_robot()
        for _ in range(240):
            states = p.getJointStates(self._robot_id, self._pybullet_indices.tolist())
            curr_pos = np.array([s[0] for s in states])
            curr_vel = np.array([s[1] for s in states])
            torques = KP * (self._nominal_pos - curr_pos) - KD * curr_vel
            torques = np.clip(torques, -self.effort_limits, self.effort_limits)
            for idx, torque in zip(self._pybullet_indices, torques):
                p.setJointMotorControl2(self._robot_id, int(idx), p.TORQUE_CONTROL, force=float(torque))
            p.stepSimulation()
        self._step_count = 0
        self._fell = False
        self.prev_action = np.zeros(self.action_dim)
        self._prev_pos = np.array(p.getBasePositionAndOrientation(self._robot_id)[0])
        self.action_history = [np.zeros(self.action_dim, dtype=np.float32)] * 2
        obs = self._get_obs()
        return (obs, {})

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.action_history.append(action.copy())
        if len(self.action_history) > 3:
            self.action_history.pop(0)
        if np.random.rand() < self.action_lag_prob:
            effective_action = self.action_history[-2]
        else:
            effective_action = action
        action_scale = 0.8
        target_pos = self._nominal_pos + effective_action * action_scale
        smooth_penalty = np.sum((action - self.prev_action) ** 2)
        self.prev_action = action.copy()
        states = p.getJointStates(self._robot_id, self._pybullet_indices.tolist())
        curr_pos = np.array([s[0] for s in states])
        curr_vel = np.array([s[1] for s in states])
        torques = KP * (target_pos - curr_pos) - KD * curr_vel
        torques = np.clip(torques, -self.effort_limits, self.effort_limits)
        for idx, torque in zip(self._pybullet_indices, torques):
            p.setJointMotorControl2(self._robot_id, int(idx), p.TORQUE_CONTROL, force=float(torque))
        for _ in range(2):
            p.stepSimulation()
        self._step_count += 1
        obs = self._get_obs()
        contact_penalty = 0.0
        contact_termination = False
        contacts = p.getContactPoints(bodyA=self._robot_id)
        for contact in contacts:
            link_idx = contact[3]
            if link_idx == -1:
                contact_penalty = -20.0
                contact_termination = True
                break
            if link_idx not in self.foot_indices:
                contact_penalty = -5.0
                if link_idx in self.heavy_indices:
                    contact_penalty = -20.0
                    contact_termination = True
                    break
        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        base_height = base_pos[2]
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn)
        reward = self._compute_reward(obs, torques, smooth_penalty, base_height=base_height)
        reward += contact_penalty
        self._fell = base_height < 0.2 or contact_termination
        bad_orientation = abs(roll) > 0.8 or abs(pitch) > 0.8
        terminated = self._fell or bad_orientation
        truncated = self._step_count >= self.max_episode_steps
        if terminated:
            reward -= self.FALL_PENALTY
        info = {'base_height': base_height, 'step': self._step_count, 'fell': self._fell}
        return (obs, reward, terminated, truncated, info)

    def _get_obs(self) -> np.ndarray:
        states = p.getJointStates(self._robot_id, self._pybullet_indices.tolist())
        joint_pos = np.array([s[0] for s in states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in states], dtype=np.float32)
        lin_vel, ang_vel = p.getBaseVelocity(self._robot_id)
        lin_vel = np.array(lin_vel, dtype=np.float32)
        ang_vel = np.array(ang_vel, dtype=np.float32)
        _, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        quat = np.array(base_orn, dtype=np.float32)
        rot_mat = np.array(p.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
        lin_vel_body = rot_mat.T @ lin_vel
        ang_vel_body = rot_mat.T @ ang_vel
        gravity_body = rot_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        r = R.from_quat(quat)
        euler = r.as_euler('xyz')
        neutral_r = R.from_euler('xyz', [euler[0], euler[1], 0.0])
        neutral_quat = neutral_r.as_quat().astype(np.float32)
        obs = np.concatenate([joint_pos, joint_vel, lin_vel_body, ang_vel_body, neutral_quat, gravity_body])
        noise = np.zeros_like(obs)
        n_j = self.action_dim
        noise[:n_j] = np.random.normal(0, self.obs_noise_scale, n_j)
        noise[n_j:n_j * 2] = np.random.normal(0, self.vel_noise_scale, n_j)
        noise[n_j * 2:n_j * 2 + 6] = np.random.normal(0, 0.01, 6)
        return (obs + noise).astype(np.float32)

    def _compute_reward(self, obs, torques, smooth_penalty, base_height):
        lin_vel = obs[24:27]
        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn)
        forward_vel = float(lin_vel[self.forward_axis])
        lateral_vel = float(lin_vel[1 - self.forward_axis])
        yaw_rate = float(obs[29])
        r_vel = 300.0 * forward_vel
        target_height = 0.45
        height_error = base_height - target_height
        r_height = 10.0 * np.exp(-20.0 * height_error ** 2)
        r_stability = -2.0 * (roll ** 2 + pitch ** 2) - 2.0 * abs(lateral_vel) - 1.0 * yaw_rate ** 2
        torque_pen = -0.0005 * float(np.sum(torques ** 2))
        smooth_pen = -0.001 * smooth_penalty
        r_moving = -15.0 if abs(forward_vel) < 0.05 else 0.0
        r = r_vel + r_height + r_stability + torque_pen + smooth_pen + r_moving
        return float(np.clip(r, -50.0, 300.0))

    def close(self):
        if p.isConnected(self._physics_client):
            p.disconnect(self._physics_client)
if __name__ == '__main__':
    URDF = '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf/anymal.urdf'
    print('Launching RobotEnvBullet smoke test...')
    env = RobotEnvBullet(URDF, max_episode_steps=200, render_mode=None)
    obs, _ = env.reset()
    print(f'  Joint names     : {env.joint_names}')
    print(f'  Effort limits   : {env.effort_limits}')
    print(f'  Obs shape       : {obs.shape}  expected (37,)')
    print(f'  Action dim      : {env.action_dim}  expected 12')
    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f'  Episode ended at step {step} | fell={info['fell']}')
            obs, _ = env.reset()
            break
    print(f'  100-step reward sum : {total_reward:.3f}')
    print(f'  Final base height   : {info['base_height']:.3f} m')
    import sys
    sys.path.insert(0, '.')
    try:
        from urdf_to_graph import URDFGraphBuilder
        builder = URDFGraphBuilder(URDF)
        assert env.joint_names == builder.joint_names, f'Joint ordering mismatch!\n  env: {env.joint_names}\n  builder: {builder.joint_names}'
        print('  Joint ordering matches URDFGraphBuilder -- obs[:24] maps correctly to graph.')
    except ImportError:
        print('  (urdf_to_graph not in path -- skip ordering check)')
    env.close()
    print('\nSmoke test passed.')
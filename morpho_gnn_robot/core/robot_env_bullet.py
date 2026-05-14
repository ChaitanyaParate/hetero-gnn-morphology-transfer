import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError('pip install pybullet')
import contextlib, os

CONTROLLABLE = {'revolute', 'continuous', 'prismatic'}
KP = 150.0
KD = 5.0
PLANE_LATERAL_FRICTION = 1.3
PLANE_SPINNING_FRICTION = 0.05
PLANE_ROLLING_FRICTION = 0.001
LINK_LATERAL_FRICTION = 1.3
LINK_SPINNING_FRICTION = 0.03
LINK_ROLLING_FRICTION = 0.001
NOMINAL_POSE_PER_JOINT = {
    # ANYmal / Hexapod (LF/RF/LH/RH/LM/RM legs)
    'LF_HAA': 0.0,  'LF_HFE':  0.6, 'LF_KFE': -1.2,
    'RF_HAA': 0.0,  'RF_HFE':  0.6, 'RF_KFE': -1.2,
    'LM_HAA': 0.0,  'LM_HFE':  0.6, 'LM_KFE': -1.2,
    'RM_HAA': 0.0,  'RM_HFE':  0.6, 'RM_KFE': -1.2,
    'LH_HAA': 0.0,  'LH_HFE': -0.6, 'LH_KFE':  1.2,
    'RH_HAA': 0.0,  'RH_HFE': -0.6, 'RH_KFE':  1.2,
    # Unitree quadrupeds (Aliengo / Go1 / A1 — FL/FR/RL/RR with hip/thigh/calf)
    'FL_hip_joint':  0.0,  'FL_thigh_joint':  0.70, 'FL_calf_joint': -1.40,
    'FR_hip_joint':  0.0,  'FR_thigh_joint':  0.70, 'FR_calf_joint': -1.40,
    'RL_hip_joint':  0.0,  'RL_thigh_joint':  0.70, 'RL_calf_joint': -1.40,
    'RR_hip_joint':  0.0,  'RR_thigh_joint':  0.70, 'RR_calf_joint': -1.40,
}

@contextlib.contextmanager
def _suppress_bullet():
    """Silence C-level PyBullet b3Warning messages (stderr fd redirect)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr  = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull_fd)

class RobotEnvBullet(gym.Env):
    metadata = {'render_modes': ['human', 'direct']}
    FALL_PENALTY = 250.0   # match MLP

    def __init__(self, urdf_path: str, max_episode_steps: int=1000, render_mode: str=None,
                 forward_axis: int=0, height_threshold: float=0.25,
                 terrain: str='flat',          # 'flat' | 'slope' | 'uneven'
                 slope_angle: float=0.0,        # radians, applied as floor tilt (max ~0.18 ≈ 10°)
                 height_noise_scale: float=0.0, # uneven terrain amplitude in metres
                 action_smooth_alpha: float=0.0 # EMA smoothing: 0=off, 0.8=strong smooth
                 ):
        super().__init__()
        self.urdf_path = urdf_path
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.forward_axis = forward_axis
        self.height_threshold = height_threshold
        self.terrain = terrain
        self.slope_angle = slope_angle
        self.height_noise_scale = height_noise_scale
        self.action_smooth_alpha = action_smooth_alpha
        self._parse_urdf()
        self.action_dim = len(self.joint_names)
        self.obs_dim = self.action_dim * 2 + 15
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.command = np.array([0.0, 0.0], dtype=np.float32)
        self.obs_noise_scale = 0.01
        self.vel_noise_scale = 0.02
        self.action_lag_prob = 0.0
        self.action_history = []
        self._smoothed_action = np.zeros(self.action_dim)  # for EMA smoothing
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
        self.prev_smooth_action = np.zeros(self.action_dim)
        self._action_scale = 0.6

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
        start_pos = [0.0, 0.0, 0.5]
        start_orient = p.getQuaternionFromEuler([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0.0])
        with _suppress_bullet():
            robot_id = p.loadURDF(self.urdf_path, start_pos, start_orient, useFixedBase=False,
                                  flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_IGNORE_VISUAL_SHAPES)

        self._robot_id = robot_id
        num_joints = p.getNumJoints(robot_id)
        self._joint_idx: dict = {}
        self.foot_indices = []
        self.heavy_indices = []
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            if any((x in link_name.upper() for x in ['FOOT', 'ADAPTER', 'SHANK'])):
                self.foot_indices.append(i)
            elif 'HIP' in link_name.upper() or 'THIGH' in link_name.upper():
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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # --- Terrain setup ---
        if self.terrain == 'slope' and self.slope_angle != 0.0:
            # Tilt the ground plane
            slope_quat = p.getQuaternionFromEuler([0.0, self.slope_angle, 0.0])
            with _suppress_bullet():
                plane_id = p.loadURDF('plane.urdf', [0, 0, 0], slope_quat)
        elif self.terrain == 'uneven' and self.height_noise_scale > 0.0:
            # Heightfield terrain centered at origin
            N = 64  # grid resolution (64x64 = 6.4m x 6.4m at 0.1m/cell)
            np.random.seed(42)  # reproducible terrain across episodes
            hfield = (np.random.rand(N, N) * 2 - 1) * self.height_noise_scale
            hfield_list = hfield.flatten().tolist()
            shape_id = p.createCollisionShape(
                p.GEOM_HEIGHTFIELD,
                meshScale=[0.1, 0.1, 1.0],
                heightfieldData=hfield_list,
                numHeightfieldRows=N,
                numHeightfieldColumns=N,
                replaceHeightfieldIndex=-1
            )
            # Center heightfield at origin; robot spawns at (0,0)
            plane_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape_id,
                basePosition=[0.0, 0.0, 0.0],
                baseOrientation=[0, 0, 0, 1]
            )
            p.changeDynamics(plane_id, -1, lateralFriction=PLANE_LATERAL_FRICTION)
        else:
            with _suppress_bullet():
                plane_id = p.loadURDF('plane.urdf')
            p.changeDynamics(plane_id, -1, lateralFriction=PLANE_LATERAL_FRICTION,
                             spinningFriction=PLANE_SPINNING_FRICTION, rollingFriction=PLANE_ROLLING_FRICTION)
        self._load_robot()
        for name, idx in self._joint_idx.items():
            target = NOMINAL_POSE_PER_JOINT.get(name, 0.0)
            p.setJointMotorControl2(self._robot_id, int(idx), p.POSITION_CONTROL, targetPosition=target, force=200.0)
        for _ in range(100):
            p.stepSimulation()
        for name, idx in self._joint_idx.items():
            p.setJointMotorControl2(self._robot_id, int(idx), p.VELOCITY_CONTROL, force=0)
        self._step_count = 0
        self._fell = False
        self.prev_action = np.zeros(self.action_dim)
        self.prev_smooth_action = np.zeros(self.action_dim)
        self._smoothed_action = np.zeros(self.action_dim)  # EMA buffer
        self._prev_pos = np.array(p.getBasePositionAndOrientation(self._robot_id)[0])
        self.action_history = [np.zeros(self.action_dim, dtype=np.float32)] * 2
        
        # Lock to forward-walk mode to break out of crouch local optimum
        self.mode = 1
        # Always command forward walking with randomized speed
        self.command = np.array([np.random.uniform(0.5, 1.0), 0.0], dtype=np.float32)
            
        obs = self._get_obs()
        return (obs, {})

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        # Action smoothing: use instance alpha (0=off for training, >0 for inference on novel morphologies)
        alpha = self.action_smooth_alpha if self.action_smooth_alpha > 0.0 else 0.8
        applied_action = alpha * action + (1.0 - alpha) * self.prev_smooth_action
        self.prev_smooth_action = applied_action.copy()
        self.action_history.append(applied_action.copy())
        if len(self.action_history) > 3:
            self.action_history.pop(0)
        if np.random.rand() < self.action_lag_prob:
            effective_action = self.action_history[-2]
        else:
            effective_action = action
        target_pos = self._nominal_pos + effective_action * self._action_scale
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
        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        base_height = base_pos[2]
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn)
        contact_penalty = 0.0
        contact_termination = False
        contacts = p.getContactPoints(bodyA=self._robot_id)
        for contact in contacts:
            if contact[2] == self._robot_id:
                continue
            link_idx = contact[3]
            if link_idx not in self.foot_indices:
                contact_penalty = -0.5
                contact_termination = True
                break
        reward = self._compute_reward(obs, torques, smooth_penalty, base_height=base_height, contact_penalty=contact_penalty)
        self._fell = base_height < self.height_threshold
        bad_orientation = abs(roll) > 0.8 or abs(pitch) > 0.8
        terminated = self._fell or bad_orientation or contact_termination
        truncated = self._step_count >= self.max_episode_steps
        term_reason = 'running'
        if base_height < self.height_threshold:
            term_reason = 'height'
        elif contact_termination:
            term_reason = 'contact'
        elif bad_orientation:
            term_reason = 'orientation'
        elif truncated:
            term_reason = 'truncated'
        info = {'base_height': base_height, 'step': self._step_count, 'fell': self._fell, 'term_reason': term_reason}
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
        obs = np.concatenate([joint_pos, joint_vel, lin_vel_body, ang_vel_body, neutral_quat, gravity_body, self.command])
        noise = np.zeros_like(obs)
        n_j = self.action_dim
        noise[:n_j] = np.random.normal(0, self.obs_noise_scale, n_j)
        noise[n_j:n_j * 2] = np.random.normal(0, self.vel_noise_scale, n_j)
        noise[n_j * 2:n_j * 2 + 6] = np.random.normal(0, 0.01, 6)
        return (obs + noise).astype(np.float32)

    def _compute_reward(self, obs, torques, smooth_penalty, base_height, contact_penalty=0.0):
        n = self.action_dim  # number of controllable joints (varies per morphology)
        # obs layout: [joint_pos(n), joint_vel(n), lin_vel(3), ang_vel(3), quat(4), grav(3), cmd(2)]
        lin_vel = obs[n*2 : n*2+3]
        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        forward_vel = float(lin_vel[self.forward_axis])
        lateral_vel = float(lin_vel[1 - self.forward_axis])
        yaw_rate = float(obs[n*2+3+2])  # ang_vel[2] = yaw rate
        
        cmd_vx, cmd_wy = self.command
        
        r_alive = 0.1
        
        # Command tracking rewards
        lin_vel_error = (forward_vel - cmd_vx)**2
        ang_vel_error = (yaw_rate - cmd_wy)**2
        r_tracking_lin = np.exp(-2.0 * lin_vel_error)
        r_tracking_ang = np.exp(-2.0 * ang_vel_error)
        # Boosted linear weight (4.0 vs old 1.5) to break crouching local optimum
        r_vel = 4.0 * r_tracking_lin + 1.5 * r_tracking_ang
        
        # Standing-still penalty: punish near-zero velocity when commanded to walk
        if cmd_vx > 0.1 and abs(forward_vel) < 0.15:
            r_vel -= 0.5
        
        if base_height < self.height_threshold:
            r_vel = 0.0
            r_alive = -self.FALL_PENALTY
            
        r_stability = -1.0 * (roll ** 2 + pitch ** 2) - 0.5 * abs(lateral_vel)
        torque_pen = -5e-06 * float(np.sum(torques ** 2))
        r = r_alive + r_vel + r_stability + torque_pen + contact_penalty
        return float(np.clip(r, -5.0, 5.0))

    def close(self):
        if p.isConnected(self._physics_client):
            p.disconnect(self._physics_client)
if __name__ == '__main__':
    URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'morpho_ros2_ws', 'src', 'morpho_robot', 'urdf', 'anymal.urdf')
    print('Launching RobotEnvBullet smoke test...')
    env = RobotEnvBullet(URDF, max_episode_steps=200, render_mode=None)
    obs, _ = env.reset()
    print(f'  Joint names     : {env.joint_names}')
    print(f'  Effort limits   : {env.effort_limits}')
    print(f'  Obs shape       : {obs.shape}  expected (39,)')
    print(f'  Action dim      : {env.action_dim}  expected 12')
    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f'  Episode ended at step {step} | fell={info["fell"]}')
            obs, _ = env.reset()
            break
    print(f'  100-step reward sum : {total_reward:.3f}')
    print(f'  Final base height   : {info["base_height"]:.3f} m')
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
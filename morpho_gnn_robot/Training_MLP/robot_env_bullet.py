"""
robot_env_bullet.py

Gym environment for legged robot locomotion using PyBullet DIRECT mode.
Designed for Kaggle headless training -- no display, no GUI.

Observation layout (flat, 37-dim):
  [0:12]  joint positions   (rad)
  [12:24] joint velocities  (rad/s)
  [24:27] base linear  velocity (m/s)
  [27:30] base angular velocity (rad/s)

Joint ordering: alphabetically sorted by name.
This MUST match URDFGraphBuilder(urdf_path).joint_names for graph construction.
Both sort alphabetically -- they will always agree.

Action: 12-dim in [-1, 1]. Scaled by effort_limit inside step().
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError("pip install pybullet")


CONTROLLABLE = {"revolute", "continuous", "prismatic"}

# PD gains -- higher stiffness needed to hold ANYmal against gravity
KP = 150.0
KD = 5.0

# Contact/friction tuning
PLANE_LATERAL_FRICTION = 1.3
PLANE_SPINNING_FRICTION = 0.05
PLANE_ROLLING_FRICTION = 0.001
LINK_LATERAL_FRICTION = 1.3
LINK_SPINNING_FRICTION = 0.03
LINK_ROLLING_FRICTION = 0.001

# ANYmal nominal standing pose -- front and hind legs have OPPOSITE sign conventions
NOMINAL_POSE_PER_JOINT = {
    "LF_HAA":  0.0,  "LF_HFE":  0.6,  "LF_KFE": -1.2,
    "RF_HAA":  0.0,  "RF_HFE":  0.6,  "RF_KFE": -1.2,
    "LH_HAA":  0.0,  "LH_HFE": -0.6,  "LH_KFE":  1.2,
    "RH_HAA":  0.0,  "RH_HFE": -0.6,  "RH_KFE":  1.2,
}


class RobotEnvBullet(gym.Env):
    """
    Parameters
    ----------
    urdf_path         : absolute path to robot URDF
    max_episode_steps : episode terminates after this many steps
    render_mode       : "human" opens GUI; None (or "direct") stays headless
    forward_axis      : 0=x, 1=y -- direction of desired locomotion
    """

    metadata = {"render_modes": ["human", "direct"]}

    # Reward weights
    # NOTE: torque cost is small but non-zero to regularize energy without
    # collapsing to a stand-still policy.
    FALL_PENALTY = 250.0  # Reduced to encourage exploration/risk-taking during recovery

    def __init__(
        self,
        urdf_path:         str,
        max_episode_steps: int   = 1000,
        render_mode:       str   = None,
        forward_axis:      int   = 0,
    ):
        super().__init__()
        self.urdf_path         = urdf_path
        self.max_episode_steps = max_episode_steps
        self.render_mode       = render_mode
        self.forward_axis      = forward_axis

        # Parse URDF once to get joint metadata
        self._parse_urdf()

        # Observation layout (36-dim):
        #   [0:12]  joint positions   (rad)
        #   [12:24] joint velocities  (rad/s)
        #   [24:27] base linear velocity (m/s)
        #   [27:30] base angular velocity (rad/s)
        #   [30:34] base orientation as quaternion (qx, qy, qz, qw)
        #   [34:37] projected gravity vector (body frame)
        # Quaternion + gravity vector together give the policy unambiguous
        # information about whether it is upright. Without this, the policy
        # cannot observe roll/pitch and cannot learn to stand or recover.
        self.obs_dim    = 12 + 12 + 3 + 3 + 4 + 3   # = 37
        self.action_dim = len(self.joint_names)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32
        )

        # Domain Randomization & Robustness params
        self.obs_noise_scale = 0.01  # 1% noise on joint pos
        self.vel_noise_scale = 0.02  # 2% noise on velocities
        self.action_lag_prob = 0.5   # 50% chance of a 1-step action delay
        self.action_history = []

        

        # Connect to PyBullet
        if render_mode == "human":
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)
            p.setPhysicsEngineParameter(enableFileCaching=0)

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._robot_id    = None
        self._step_count  = 0
        self._fell        = False
        self.prev_action = np.zeros(self.action_dim)

    # ------------------------------------------------------------------
    def _parse_urdf(self):
        """Read URDF to get joint names, types, and effort limits."""
        import xml.etree.ElementTree as ET
        root = ET.parse(self.urdf_path).getroot()

        names:   list = []
        efforts: dict = {}

        for joint in root.findall("joint"):
            jtype = joint.attrib.get("type", "fixed")
            if jtype not in CONTROLLABLE:
                continue
            name = joint.attrib["name"]
            names.append(name)
            lim = joint.find("limit")
            efforts[name] = float(lim.attrib.get("effort", 40.0)) if lim is not None else 40.0

        # Sort alphabetically -- same ordering as URDFGraphBuilder
        self.joint_names   = sorted(names)
        self.effort_limits = np.array([efforts[n] for n in self.joint_names])

    # ------------------------------------------------------------------
    def _load_robot(self):
        """(Re)load robot in PyBullet. Called on every reset()."""
        if self._robot_id is not None:
            p.removeBody(self._robot_id)

        # Start slightly above ground; adjust if URDF origin differs
        start_pos    = [0.0, 0.0, 0.65]
        start_orient = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        robot_id     = p.loadURDF(
            self.urdf_path,
            start_pos, start_orient,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        self._robot_id = robot_id

        # Map joint names to PyBullet joint indices
        num_joints   = p.getNumJoints(robot_id)
        self._joint_idx: dict = {}

        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            name  = info[1].decode("utf-8")
            if name in self.joint_names:
                self._joint_idx[name] = i
                # CRITICAL: disable default position control motor
                # Without this, PyBullet fights your torque commands
                p.setJointMotorControl2(
                    robot_id, i,
                    p.VELOCITY_CONTROL,
                    force=0
                )

        assert len(self._joint_idx) == len(self.joint_names), (
            f"URDF has {len(self.joint_names)} controllable joints but "
            f"PyBullet found {len(self._joint_idx)}. Check URDF path."
        )

        # Ordered PyBullet indices matching self.joint_names
        self._pybullet_indices = np.array(
            [self._joint_idx[n] for n in self.joint_names], dtype=np.int32
        )

        # Nominal standing pose in same alphabetical order as joint_names
        self._nominal_pos = np.array([
            NOMINAL_POSE_PER_JOINT.get(name, 0.0)
            for name in self.joint_names
        ])

        # ---- Initialize joints to standing pose ----
        for name, idx in self._joint_idx.items():
            val = NOMINAL_POSE_PER_JOINT.get(name, 0.0)
            p.resetJointState(self._robot_id, int(idx), val)

        for i in range(p.getNumJoints(self._robot_id)):
            p.changeDynamics(
                self._robot_id,
                i,
                linearDamping=0.04,
                angularDamping=0.04,
                lateralFriction=LINK_LATERAL_FRICTION,
                spinningFriction=LINK_SPINNING_FRICTION,
                rollingFriction=LINK_ROLLING_FRICTION,
            )

        p.changeDynamics(
            self._robot_id,
            -1,  # base link
            linearDamping=0.04,
            angularDamping=0.04,
            lateralFriction=LINK_LATERAL_FRICTION,
            spinningFriction=LINK_SPINNING_FRICTION,
            rollingFriction=LINK_ROLLING_FRICTION,
        )


    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        self._robot_id = None   # resetSimulation already removed all bodies
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 400.0) # Higher freq for 200Hz control (dt=2/400=5ms)
        p.setPhysicsEngineParameter(numSolverIterations=50, numSubSteps=1)
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(
            plane_id,
            -1,
            lateralFriction=PLANE_LATERAL_FRICTION,
            spinningFriction=PLANE_SPINNING_FRICTION,
            rollingFriction=PLANE_ROLLING_FRICTION,
        )

        self._load_robot()

        # Settle: hold nominal pose with PD so robot is actually standing at episode start
        for _ in range(240):
            states   = p.getJointStates(self._robot_id, self._pybullet_indices.tolist())
            curr_pos = np.array([s[0] for s in states])
            curr_vel = np.array([s[1] for s in states])
            torques  = KP * (self._nominal_pos - curr_pos) - KD * curr_vel
            torques  = np.clip(torques, -self.effort_limits, self.effort_limits)
            for idx, torque in zip(self._pybullet_indices, torques):
                p.setJointMotorControl2(
                    self._robot_id, int(idx),
                    p.TORQUE_CONTROL, force=float(torque)
                )
            p.stepSimulation()

        self._step_count = 0
        self._fell       = False
        self.prev_action = np.zeros(self.action_dim)
        self._prev_pos   = np.array(p.getBasePositionAndOrientation(self._robot_id)[0])
        
        # Buffer for action lag simulation
        self.action_history = [np.zeros(12, dtype=np.float32)] * 2

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        action : [num_joints] in [-1, 1]
                 Interpreted as offsets from nominal standing pose (radians).
                 Scaled by 0.1 rad so max deviation = ±0.1 rad per joint.
                 PD controller converts target positions to torques.
        """
        action = np.clip(action, -1.0, 1.0)

        # target position = nominal pose + scaled action offset
        # Action scale: 0.1 rad max deviation from nominal.
        # At 0.5 rad: random policy (sigma=0.37) generates 27Nm unintended
        # forces per joint, immediately destabilising the body before the
        # policy has any chance to learn. At 0.1 rad: max unintended force
        # is 5.5Nm -- robot stays near nominal during random exploration.
        # action_scale = 0.80 rad: push exploration toward larger strides.
        # HFE joint needs ~0.3-0.4 rad swing range for a proper step.
        # Previously 0.1 rad was needed because policy had no orientation obs
        # and random large actions caused immediate falls.
        # Now orientation (gravity vector) is in obs so the policy can
        # stabilize while learning to walk. 0.35 gives locomotion range
        # without producing joint limit violations.
        # 1. Action Lag Simulation (Simulate 1-step network delay)
        # Randomly choose between current and previous action to simulate jitter
        self.action_history.append(action.copy())
        if len(self.action_history) > 3:
            self.action_history.pop(0)

        if np.random.rand() < self.action_lag_prob:
            effective_action = self.action_history[-2] # Uses previous action
        else:
            effective_action = action

        action_scale = 0.80
        target_pos = self._nominal_pos + effective_action * action_scale

        smooth_penalty = np.sum((action - self.prev_action) ** 2)
        self.prev_action = action.copy()

        # Read current joint states for PD computation
        states    = p.getJointStates(self._robot_id, self._pybullet_indices.tolist())
        curr_pos  = np.array([s[0] for s in states])
        curr_vel  = np.array([s[1] for s in states])

        # PD torques
        torques = KP * (target_pos - curr_pos) - KD * curr_vel
        torques = np.clip(torques, -self.effort_limits, self.effort_limits)

        for idx, torque in zip(self._pybullet_indices, torques):
            p.setJointMotorControl2(
                self._robot_id, int(idx),
                p.TORQUE_CONTROL,
                force=float(torque)
            )

        for _ in range(2):
            p.stepSimulation()

        self._step_count += 1

        # --- observation ---
        obs = self._get_obs()

        # --- base state ---
        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        base_height = base_pos[2]
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn)

        # --- reward FIRST (mandatory) ---
        reward = self._compute_reward(
            obs,
            torques,
            smooth_penalty,
            base_height=base_height
        )

        # --- termination ---
        self._fell = base_height < 0.20
        bad_orientation = abs(roll) > 0.8 or abs(pitch) > 0.8

        terminated = self._fell or bad_orientation
        truncated  = self._step_count >= self.max_episode_steps

        # --- penalty AFTER reward exists ---
        if terminated:
            reward -= self.FALL_PENALTY

        info = {
            "base_height": base_height,
            "step": self._step_count,
            "fell": self._fell,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build 37-dim observation including orientation.

        Velocities are rotated into the BODY frame so the policy learns
        heading-invariant locomotion.  obs[24] always means 'forward
        relative to the robot', regardless of world yaw.
        """
        states    = p.getJointStates(self._robot_id, self._pybullet_indices.tolist())
        joint_pos = np.array([s[0] for s in states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in states], dtype=np.float32)

        lin_vel, ang_vel = p.getBaseVelocity(self._robot_id)
        lin_vel = np.array(lin_vel, dtype=np.float32)   # world frame
        ang_vel = np.array(ang_vel, dtype=np.float32)   # world frame

        _, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        quat = np.array(base_orn, dtype=np.float32)   # (qx, qy, qz, qw)

        # Rotation matrix: world -> body  (R^T maps world vectors to body frame)
        rot_mat = np.array(p.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
        lin_vel_body = rot_mat.T @ lin_vel   # body-frame linear velocity
        ang_vel_body = rot_mat.T @ ang_vel   # body-frame angular velocity

        # Projected gravity in body frame
        gravity_body = rot_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        obs = np.concatenate([joint_pos, joint_vel, lin_vel_body, ang_vel_body, quat, gravity_body])
        
        # Add Domain Randomization Noise
        noise = np.zeros_like(obs)
        noise[:12]  = np.random.normal(0, self.obs_noise_scale, 12)  # Joint pos noise
        noise[12:24] = np.random.normal(0, self.vel_noise_scale, 12) # Joint vel noise
        noise[24:30] = np.random.normal(0, 0.01, 6)                  # Base vel noise
        
        return (obs + noise).astype(np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self, obs, torques, smooth_penalty, base_height):
        lin_vel = obs[24:27]
        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn)

        forward_vel = float(lin_vel[self.forward_axis])
        lateral_vel = float(lin_vel[1 - self.forward_axis])
        yaw_rate    = float(obs[29])

        # 1. Forward Speed Reward (Continuous Linear)
        r_vel = 300.0 * forward_vel
        
        # 2. Base Height Reward (Gaussian bonus for nominal height 0.45m)
        target_height = 0.45
        height_error = base_height - target_height
        r_height = 3.0 * np.exp(-15.0 * (height_error**2))
        
        # 3. Stability Penalties
        r_stability = (
            -1.0 * (roll**2 + pitch**2)     # Upright stance
            -2.0 * abs(lateral_vel)         # Heading alignment
            -1.0 * (yaw_rate ** 2)          # Yaw stability
        )
        
        # 4. Efficiency & Smoothing
        torque_pen = -0.0005 * float(np.sum(torques ** 2))
        smooth_pen = -0.001 * smooth_penalty
        
        # 5. Stagnation Break (Directly punish standing still)
        r_moving = -15.0 if abs(forward_vel) < 0.05 else 0.0
        
        r = r_vel + r_height + r_stability + torque_pen + smooth_pen + r_moving
        
        # Prevent extremely negative rewards from glitching
        return float(np.clip(r, -50.0, 300.0))

    # ------------------------------------------------------------------
    def close(self):
        if p.isConnected(self._physics_client):
            p.disconnect(self._physics_client)


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    URDF = (
        "/mnt/newvolume/Programming/Python/Deep_Learning/"
        "Relational_Bias_for_Morphological_Generalization/"
        "morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf/anymal.urdf"
    )

    print("Launching RobotEnvBullet smoke test...")
    env = RobotEnvBullet(URDF, max_episode_steps=200, render_mode=None)
    obs, _ = env.reset()

    print(f"  Joint names     : {env.joint_names}")
    print(f"  Effort limits   : {env.effort_limits}")
    print(f"  Obs shape       : {obs.shape}  expected (37,)")
    print(f"  Action dim      : {env.action_dim}  expected 12")

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample() * 0.1   # gentle random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Episode ended at step {step} | fell={info['fell']}")
            obs, _ = env.reset()
            break

    print(f"  100-step reward sum : {total_reward:.3f}")
    print(f"  Final base height   : {info['base_height']:.3f} m")

    # Verify joint ordering matches URDFGraphBuilder
    import sys
    sys.path.insert(0, ".")
    try:
        from urdf_to_graph import URDFGraphBuilder
        builder = URDFGraphBuilder(URDF)
        assert env.joint_names == builder.joint_names, (
            f"Joint ordering mismatch!\n  env: {env.joint_names}\n  builder: {builder.joint_names}"
        )
        print("  Joint ordering matches URDFGraphBuilder -- obs[:24] maps correctly to graph.")
    except ImportError:
        print("  (urdf_to_graph not in path -- skip ordering check)")

    env.close()
    print("\nSmoke test passed.")
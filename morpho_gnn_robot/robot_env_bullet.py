"""
robot_env_bullet.py

Gym environment for legged robot locomotion using PyBullet DIRECT mode.
Designed for Kaggle headless training -- no display, no GUI.

Observation layout (flat, 30-dim):
  [0:12]  joint positions   (rad)
  [12:24] joint velocities  (rad/s)
  [24:27] base linear  velocity (m/s)
  [27:30] base angular velocity (rad/s)

Joint ordering: alphabetically sorted by name.
This MUST match URDFGraphBuilder(urdf_path).joint_names for graph construction.
Both sort alphabetically -- they will always agree.

Action: 12-dim in [-1, 1]. Scaled by effort_limit inside step().
"""


import os
import time
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
    # NOTE: torque cost removed -- with KP=150 torques reach 75 Nm per joint,
    # making τ² sum ~67,500 per step. At 1e-4 that is -6.75 per step, -6750
    # over 1000 steps. The agent learns to do nothing to minimize torque cost
    # rather than walk. Remove it until locomotion emerges; add back later.
    FALL_PENALTY = 10.0    # one-time penalty, small enough not to dominate

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
            jtype = info[2]   # 0=revolute, 1=prismatic, 4=fixed

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
                angularDamping=0.04
            )

        p.changeDynamics(
            self._robot_id,
            -1,  # base link
            linearDamping=0.04,
            angularDamping=0.04
        )


    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        self._robot_id = None   # resetSimulation already removed all bodies
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.loadURDF("plane.urdf")

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
        # action_scale = 0.35 rad: sufficient range for trot gait on ANYmal.
        # HFE joint needs ~0.3-0.4 rad swing range for a proper step.
        # Previously 0.1 rad was needed because policy had no orientation obs
        # and random large actions caused immediate falls.
        # Now orientation (gravity vector) is in obs so the policy can
        # stabilize while learning to walk. 0.35 gives locomotion range
        # without producing joint limit violations.
        target_pos = self._nominal_pos + action * 0.35

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

        for _ in range(4):
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
        """Build 37-dim observation including orientation."""
        states    = p.getJointStates(self._robot_id, self._pybullet_indices.tolist())
        joint_pos = np.array([s[0] for s in states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in states], dtype=np.float32)

        lin_vel, ang_vel = p.getBaseVelocity(self._robot_id)
        lin_vel = np.array(lin_vel, dtype=np.float32)
        ang_vel = np.array(ang_vel, dtype=np.float32)

        _, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        quat = np.array(base_orn, dtype=np.float32)   # (qx, qy, qz, qw)

        # Projected gravity in body frame: tells the policy which way is "down"
        # relative to its own orientation. This is the clearest standing signal.
        # gravity_world = [0, 0, -1] (normalized)
        # rot_matrix maps world->body, so gravity_body = R^T * [0,0,-1]
        rot_mat = np.array(p.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
        gravity_body = rot_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        return np.concatenate([joint_pos, joint_vel, lin_vel, ang_vel, quat, gravity_body])

    # ------------------------------------------------------------------
    def _compute_reward(self, obs: np.ndarray, torques: np.ndarray, smooth_penalty: float, base_height: float) -> float:
        lin_vel  = obs[24:27]

        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        roll, pitch, _     = p.getEulerFromQuaternion(base_orn)

        forward_vel = float(lin_vel[self.forward_axis])
        lateral_vel = float(lin_vel[1 - self.forward_axis])

        # NO alive bonus -- it creates a local optimum where the policy learns
        # to stand still and collect it rather than walk. The ONLY path to
        # positive reward is forward_vel > 0.
        target_vel = 0.6
        vel_error = (forward_vel - target_vel)
        r = (
            10.0 * np.exp(-vel_error**2)   # ONLY reward forward
            - 1.0 * abs(lateral_vel)
            - 2.0 * (roll**2 + pitch**2)
            - 2.0 * max(0.0, 0.45 - base_height)
            - 0.02 * smooth_penalty
        )

        return float(r)

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
    print(f"  Obs shape       : {obs.shape}  expected (30,)")
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
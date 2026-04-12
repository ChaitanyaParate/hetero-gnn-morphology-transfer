#!/usr/bin/env python3
"""
MLP_policy_node.py
------------------
Deploys a trained flat-observation PPO checkpoint into ROS2 Jazzy.

Subscribes  : /joint_states, /odom
Publishes   : /model/robot/joint/*/cmd_pos (std_msgs/Float64)
Control Hz  : 20 Hz (timer-driven, non-blocking)

This is the MLP counterpart to gnn_policy_node.py.
It reconstructs the same 37-dim observation used during MLP training:
  [0:12]  joint positions
  [12:24] joint velocities
  [24:27] base linear velocity
  [27:30] base angular velocity
  [30:34] base orientation quaternion
  [34:37] projected gravity in body frame
"""

import argparse
import threading
import xml.etree.ElementTree as ET
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

try:
    from urdf_to_graph import URDFGraphBuilder
except ImportError as exc:
    print(f"[FATAL] Cannot import project modules: {exc}")
    print("        cd into the directory containing urdf_to_graph.py first.")
    raise SystemExit(1)


CONTROL_HZ = 200
POSITION_SCALE = 0.80  # must match action_scale in robot_env_bullet.py
JOINT_COMMAND_FMT = "/model/robot/joint/{}/cmd_pos"
HIDDEN_DIM = 256
OBS_NORM_DIM = 30
ACTION_SMOOTH_ALPHA = 0.75  # Reduced smoothing for faster response
MAX_CMD_STEP = 0.25        # Relaxed step limit to allow standing up quickly
STARTUP_HOLD_TICKS = 200       # Hold nominal pose for 1 second to settle

NOMINAL_POSE_PER_JOINT = {
    "LF_HAA": 0.0,  "LF_HFE": 0.6,  "LF_KFE": -1.2,
    "RF_HAA": 0.0,  "RF_HFE": 0.6,  "RF_KFE": -1.2,
    "LH_HAA": 0.0,  "LH_HFE": -0.6, "LH_KFE": 1.2,
    "RH_HAA": 0.0,  "RH_HFE": -0.6, "RH_KFE": 1.2,
}


def _layer_init(layer, std=0.01, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPActorCritic(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            _layer_init(torch.nn.Linear(obs_dim, hidden_dim), std=1.0),
            torch.nn.Tanh(),
            _layer_init(torch.nn.Linear(hidden_dim, hidden_dim), std=1.0),
            torch.nn.Tanh(),
        )
        self.actor_head = torch.nn.Sequential(
            _layer_init(torch.nn.Linear(hidden_dim, hidden_dim), std=0.5),
            torch.nn.Tanh(),
            _layer_init(torch.nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -0.7))
        self.critic_head = torch.nn.Sequential(
            _layer_init(torch.nn.Linear(hidden_dim, hidden_dim), std=0.5),
            torch.nn.Tanh(),
            _layer_init(torch.nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self._encode(obs)
        return self.critic_head(h)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        h = self._encode(obs)
        mean = self.actor_head(h)
        std = self.log_std.exp().clamp(min=0.15, max=0.8)
        std = std.unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(h)
        return action, log_prob, entropy, value


class MLPPolicyNode(Node):
    def __init__(
        self,
        checkpoint_path: str,
        urdf_path: str,
        device_str: str,
        action_remap: str,
        odom_in_base_frame: bool,
    ):
        super().__init__("MLP_policy_node")
        self.device = torch.device(device_str)
        self.get_logger().info(f"Device: {self.device}")

        self.get_logger().info(f"Parsing URDF: {urdf_path}")
        self.builder = URDFGraphBuilder(urdf_path, add_body_node=True)
        self.get_logger().info(
            f"Joint order: {self.builder.joint_names} | joints={self.builder.num_joints}"
        )

        self._action_remap_idx = self._build_action_remap(action_remap)
        if self._action_remap_idx is None:
            self.get_logger().info("Action remap: none")
        else:
            self.get_logger().info(f"Action remap: {action_remap}")

        self._odom_in_base_frame = odom_in_base_frame
        self.get_logger().info(
            f"Odom twist frame: {'base' if self._odom_in_base_frame else 'world'}"
        )

        # In training, only obs[:30] was normalized. Reuse checkpoint stats here.
        self._obs_norm_mean = np.zeros((OBS_NORM_DIM,), dtype=np.float32)
        self._obs_norm_var = np.ones((OBS_NORM_DIM,), dtype=np.float32)
        self._obs_norm_count = 1.0
        self._norm_available = False

        self._joint_lower, self._joint_upper = self._load_joint_limits(urdf_path)

        self.model = self._load_checkpoint(checkpoint_path)
        self.model.eval()

        self._lock = threading.Lock()
        self._joint_pos = np.zeros(self.builder.num_joints, dtype=np.float32)
        self._joint_vel = np.zeros(self.builder.num_joints, dtype=np.float32)
        self._base_lin_vel = np.zeros(3, dtype=np.float32)
        self._base_ang_vel = np.zeros(3, dtype=np.float32)
        self._base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._prev_action = np.zeros(self.builder.action_dim, dtype=np.float32)
        self._prev_cmd_pos = np.array(
            [NOMINAL_POSE_PER_JOINT.get(j, 0.0) for j in self.builder.joint_names],
            dtype=np.float32,
        )
        self._ticks = 0
        self._startup_hold_ticks = STARTUP_HOLD_TICKS
        self._policy_activated = False
        self._last_telemetry_time = self.get_clock().now()
        
        self._joint_ready = False
        self._odom_ready = False
        self._cmd_initialized = False

        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)
        self.create_subscription(Odometry, "/odom", self._cb_odom, 10)

        self._joint_pubs = {
            jname: self.create_publisher(Float64, JOINT_COMMAND_FMT.format(jname), 10)
            for jname in self.builder.joint_names
        }
        self.create_timer(1.0 / CONTROL_HZ, self._control_cb)

        self.get_logger().info(
            f"Ready. Publishing {self.builder.num_joints} joint commands to '{JOINT_COMMAND_FMT}' at {CONTROL_HZ} Hz."
        )

    def _load_joint_limits(self, urdf_path: str):
        lower = np.full(self.builder.num_joints, -np.inf, dtype=np.float32)
        upper = np.full(self.builder.num_joints, np.inf, dtype=np.float32)

        root = ET.parse(urdf_path).getroot()
        limits_by_name = {}
        for joint in root.findall("joint"):
            name = joint.attrib.get("name")
            lim = joint.find("limit")
            if name and lim is not None:
                lo = float(lim.attrib.get("lower", "-1e9"))
                hi = float(lim.attrib.get("upper", "1e9"))
                limits_by_name[name] = (lo, hi)

        for i, jname in enumerate(self.builder.joint_names):
            if jname in limits_by_name:
                lower[i], upper[i] = limits_by_name[jname]

        return lower, upper

    def _build_action_remap(self, mode: str):
        if mode == "none":
            return None

        if mode == "rotate_cw":
            leg_map = {"LF": "LH", "RF": "LF", "RH": "RF", "LH": "RH"}
        elif mode == "rotate_ccw":
            leg_map = {"LF": "RF", "RF": "RH", "RH": "LH", "LH": "LF"}
        else:
            raise ValueError(f"Unknown action_remap mode: {mode}")

        name_to_idx = {name: idx for idx, name in enumerate(self.builder.joint_names)}
        remap = []
        for jname in self.builder.joint_names:
            parts = jname.split("_", 1)
            if len(parts) != 2:
                remap.append(name_to_idx[jname])
                continue
            leg, suffix = parts
            source_leg = leg_map.get(leg, leg)
            source_name = f"{source_leg}_{suffix}"
            remap.append(name_to_idx.get(source_name, name_to_idx[jname]))

        return np.array(remap, dtype=np.int64)

    def _load_checkpoint(self, path: str) -> MLPActorCritic:
        self.get_logger().info(f"Loading checkpoint: {path}")
        raw = torch.load(path, map_location=self.device, weights_only=False)

        model = MLPActorCritic(
            obs_dim=37,
            action_dim=self.builder.action_dim,
            hidden_dim=HIDDEN_DIM,
        ).to(self.device)

        if isinstance(raw, dict):
            if "agent" in raw:
                state = raw["agent"]
                step = raw.get("global_step", "unknown")
                self.get_logger().info(f"Training checkpoint, step={step}")
                if "obs_norm_mean" in raw and "obs_norm_var" in raw:
                    self._obs_norm_mean = np.array(raw["obs_norm_mean"], dtype=np.float32)
                    self._obs_norm_var = np.array(raw["obs_norm_var"], dtype=np.float32)
                    self._obs_norm_count = float(raw.get("obs_norm_count", 1.0))
                    self._norm_available = True
                else:
                    self.get_logger().error(
                        "Checkpoint has no obs_norm stats. Refusing to run policy.",
                    )
            else:
                state = raw
                self.get_logger().info("Bare state dict checkpoint")
            model.load_state_dict(state, strict=True)
        elif isinstance(raw, MLPActorCritic):
            self.get_logger().info("Full model object checkpoint")
            return raw.to(self.device)
        else:
            raise RuntimeError(
                f"Unrecognised checkpoint type: {type(raw)}. "
                "Expected dict with key 'agent', a bare state dict, or an MLPActorCritic object."
            )

        n_params = sum(p.numel() for p in model.parameters())
        self.get_logger().info(f"Model loaded: {n_params:,} parameters")
        return model

    def _normalize_policy_obs(self, obs: np.ndarray) -> np.ndarray:
        head = obs[:OBS_NORM_DIM]
        tail = obs[OBS_NORM_DIM:]
        denom = np.sqrt(self._obs_norm_var) + 1e-8
        head_n = np.clip((head - self._obs_norm_mean) / denom, -10.0, 10.0)
        return np.concatenate([head_n, tail], axis=0).astype(np.float32)

    def _cb_joint_states(self, msg: JointState):
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        pos = np.zeros(self.builder.num_joints, dtype=np.float32)
        vel = np.zeros(self.builder.num_joints, dtype=np.float32)
        missing = []

        for j, jname in enumerate(self.builder.joint_names):
            if jname in name_to_idx:
                i = name_to_idx[jname]
                pos[j] = msg.position[i] if msg.position else 0.0
                vel[j] = msg.velocity[i] if msg.velocity else 0.0
            else:
                missing.append(jname)

        if missing:
            self.get_logger().warn(f"Missing joints in /joint_states: {missing}", throttle_duration_sec=5.0)

        with self._lock:
            self._joint_pos = pos
            self._joint_vel = vel
            self._joint_ready = True
            if not self._cmd_initialized:
                self._prev_cmd_pos = pos.copy()
                self._cmd_initialized = True

    def _cb_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        tw = msg.twist.twist
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        qn = np.linalg.norm(quat)
        if qn > 1e-8:
            quat = quat / qn
        else:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        with self._lock:
            self._base_quat = quat
            self._base_lin_vel = np.array([tw.linear.x, tw.linear.y, tw.linear.z], dtype=np.float32)
            self._base_ang_vel = np.array([tw.angular.x, tw.angular.y, tw.angular.z], dtype=np.float32)
            self._odom_ready = True

    def _control_cb(self):
        if not self._joint_ready:
            return

        with self._lock:
            pos = self._joint_pos.copy()
            vel = self._joint_vel.copy()
            base_lin_vel = self._base_lin_vel.copy()
            base_ang_vel = self._base_ang_vel.copy()
            base_quat = self._base_quat.copy()
            prev_cmd = self._prev_cmd_pos.copy()

        if not self._odom_ready:
            base_lin_vel = np.zeros(3, dtype=np.float32)
            base_ang_vel = np.zeros(3, dtype=np.float32)
            base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            self.get_logger().warn(
                "No /odom yet; using zeros/identity for base state.",
                throttle_duration_sec=5.0,
            )

        if self._startup_hold_ticks > 0:
            self._startup_hold_ticks -= 1
            for i, jname in enumerate(self.builder.joint_names):
                msg = Float64()
                msg.data = float(prev_cmd[i])
                self._joint_pubs[jname].publish(msg)
            return

        if not self._norm_available:
            self.get_logger().error(
                "Checkpoint missing obs_norm stats. Use a step checkpoint, not mlp_ppo_final.pt.",
                throttle_duration_sec=5.0,
            )
            for i, jname in enumerate(self.builder.joint_names):
                msg = Float64()
                msg.data = float(prev_cmd[i])
                self._joint_pubs[jname].publish(msg)
            return

        rot_mat = np.array(
            [
                [1 - 2 * (base_quat[1] ** 2 + base_quat[2] ** 2), 2 * (base_quat[0] * base_quat[1] - base_quat[2] * base_quat[3]), 2 * (base_quat[0] * base_quat[2] + base_quat[1] * base_quat[3])],
                [2 * (base_quat[0] * base_quat[1] + base_quat[2] * base_quat[3]), 1 - 2 * (base_quat[0] ** 2 + base_quat[2] ** 2), 2 * (base_quat[1] * base_quat[2] - base_quat[0] * base_quat[3])],
                [2 * (base_quat[0] * base_quat[2] - base_quat[1] * base_quat[3]), 2 * (base_quat[1] * base_quat[2] + base_quat[0] * base_quat[3]), 1 - 2 * (base_quat[0] ** 2 + base_quat[1] ** 2)],
            ],
            dtype=np.float32,
        )

        if not self._odom_in_base_frame:
            base_lin_vel = rot_mat.T @ base_lin_vel
            base_ang_vel = rot_mat.T @ base_ang_vel

        gravity_body = rot_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # 3. Startup & Sensor Blocking Logic
        if not self._odom_ready or not self._joint_ready:
            # Still waiting for sensors. Hold nominal pose.
            for jname in self.builder.joint_names:
                if jname in self._joint_pubs:
                    self._joint_pubs[jname].publish(Float64(data=NOMINAL_POSE_PER_JOINT[jname]))
            return

        if self._ticks < STARTUP_HOLD_TICKS:
            # Hold nominal pose but sensors ARE ready. Just settling on ground.
            for jname in self.builder.joint_names:
                if jname in self._joint_pubs:
                    self._joint_pubs[jname].publish(Float64(data=NOMINAL_POSE_PER_JOINT[jname]))
            self._ticks += 1
            return

        # 4. Telemetry (1Hz)
        now = self.get_clock().now()
        if (now - self._last_telemetry_time).nanoseconds > 1e9:
            # gravity_body[0] is roughly -sin(pitch). 
            # If robot tilts forward (nose down), gravity_body[0] becomes positive.
            pitch_est = -gravity_body[0]
            self.get_logger().info(
                f"Telemetery | Pitch: {pitch_est:.2f} | FwdVel: {base_lin_vel[0]:.2f} | Ticks: {self._ticks}"
            )
            self._last_telemetry_time = now

        # 5. Neural Network Inference
        obs = np.concatenate([pos, vel, base_lin_vel, base_ang_vel, base_quat, gravity_body]).astype(np.float32)

        # 4. Neural Network Inference
        obs = np.concatenate([pos, vel, base_lin_vel, base_ang_vel, base_quat, gravity_body]).astype(np.float32)
        obs_n = self._normalize_policy_obs(obs)
        obs_t = torch.tensor(obs_n, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Deterministic policy action for deployment.
            mean = self.model.actor_head(self.model._encode(obs_t))
            action = mean

        action_np = action.squeeze(0).cpu().numpy()

        if self._action_remap_idx is not None:
            action_np = action_np[self._action_remap_idx]

        action_np = np.clip(action_np, -1.0, 1.0)
        action_np = (1.0 - ACTION_SMOOTH_ALPHA) * self._prev_action + ACTION_SMOOTH_ALPHA * action_np
        self._prev_action = action_np.copy()

        # 6. Safety Ramp: gradually allow larger movements
        # During the first 400 ticks (2s) of policy activation, clamp the action magnitude
        ramp_ticks = 400
        policy_ticks = self._ticks - STARTUP_HOLD_TICKS
        ramp_factor = min(1.0, float(policy_ticks) / ramp_ticks)

        cmd_pos = []
        for i, jname in enumerate(self.builder.joint_names):
            # target = nominal + scaled action * ramp
            target = NOMINAL_POSE_PER_JOINT.get(jname, 0.0) + float(action_np[i] * POSITION_SCALE * ramp_factor)
            target = float(np.clip(target, self._joint_lower[i], self._joint_upper[i]))
            delta = target - float(self._prev_cmd_pos[i])
            delta = float(np.clip(delta, -MAX_CMD_STEP, MAX_CMD_STEP))
            target = float(self._prev_cmd_pos[i] + delta)
            self._prev_cmd_pos[i] = target
            cmd_pos.append(target)
            msg = Float64()
            msg.data = float(target)
            self._joint_pubs[jname].publish(msg)

        self.get_logger().debug(
            f"Cmd pos (rad): {np.round(np.array(cmd_pos), 3).tolist()}",
            throttle_duration_sec=1.0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt file")
    parser.add_argument("--urdf", required=True, help="Path to URDF")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--action_remap",
        default="none",
        choices=["none", "rotate_cw", "rotate_ccw"],
        help="Remap actions to rotate gait direction",
    )
    parser.add_argument(
        "--odom_in_base_frame",
        default=1,
        type=int,
        choices=[0, 1],
        help="Interpret /odom twist in base frame (1) or world frame (0)",
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = MLPPolicyNode(
        args.checkpoint,
        args.urdf,
        args.device,
        args.action_remap,
        bool(args.odom_in_base_frame),
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

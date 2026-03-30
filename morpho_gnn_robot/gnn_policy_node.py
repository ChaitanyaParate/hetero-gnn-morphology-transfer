#!/usr/bin/env python3
"""
gnn_policy_node.py  -- position-control version for Gazebo Harmonic
Publishes Float64 position targets to per-joint gz-bridged topics.

Policy output is a position OFFSET from nominal pose:
    target_pos = nominal_pos + action * 0.5   (matches robot_env_bullet.step)

Each joint gets its own ROS2 topic bridged to Gazebo JointPositionController.
"""

import argparse
import sys
import threading
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

try:
    from gnn_actor_critic import GNNActorCritic
    from urdf_to_graph import URDFGraphBuilder
except ImportError as e:
    print(f"[FATAL] {e}\n  cd into the directory with gnn_actor_critic.py first.")
    sys.exit(1)

CONTROL_HZ = 20
HIDDEN_DIM = 64

# ANYmal B nominal standing pose (matches robot_env_bullet.py NOMINAL_POSE_PER_JOINT)
# Keys must cover all 12 alphabetical joint names
NOMINAL_POSE = {
    "LF_HAA":  0.0,  "LF_HFE":  0.6,  "LF_KFE": -1.2,
    "LH_HAA":  0.0,  "LH_HFE": -0.6,  "LH_KFE":  1.2,
    "RF_HAA":  0.0,  "RF_HFE":  0.6,  "RF_KFE": -1.2,
    "RH_HAA":  0.0,  "RH_HFE": -0.6,  "RH_KFE":  1.2,
}


class GNNPolicyNode(Node):

    def __init__(self, checkpoint_path: str, urdf_path: str, device_str: str):
        super().__init__("gnn_policy_node")
        self.device = torch.device(device_str)

        # ── URDF graph builder ───────────────────────────────────────────────
        self.builder = URDFGraphBuilder(urdf_path, add_body_node=True)
        self.get_logger().info(f"Joints (alphabetical): {self.builder.joint_names}")

        # Nominal pose array in alphabetical joint order
        self._nominal = np.array(
            [NOMINAL_POSE[n] for n in self.builder.joint_names], dtype=np.float32
        )

        # ── checkpoint ───────────────────────────────────────────────────────
        self.model = self._load_checkpoint(checkpoint_path)
        self.model.eval()

        # ── runtime obs: init to nominal so first commands are sane ──────────
        self._lock      = threading.Lock()
        self._joint_pos = self._nominal.copy()
        self._joint_vel = np.zeros(self.builder.num_joints, dtype=np.float32)
        self._obs_ready = False

        # ── one Float64 publisher per joint ──────────────────────────────────
        # Topic pattern must match parameter_bridge config and URDF plugin tags
        self._pubs = {
            name: self.create_publisher(
                Float64, f"/model/robot/joint/{name}/cmd_pos", 10
            )
            for name in self.builder.joint_names
        }
        self.get_logger().info(
            f"Publishing to {len(self._pubs)} joint command topics."
        )

        self.create_subscription(JointState, "/joint_states", self._cb_joints, 10)
        self.create_timer(1.0 / CONTROL_HZ, self._control_cb)
        self.get_logger().info(
            f"Ready at {CONTROL_HZ} Hz. Waiting for /joint_states..."
        )

    # -----------------------------------------------------------------------
    def _load_checkpoint(self, path: str) -> GNNActorCritic:
        """Loads checkpoint saved by train_gnn_ppo.py (key: 'agent')."""
        self.get_logger().info(f"Loading: {path}")
        raw = torch.load(path, map_location=self.device, weights_only=False)

        model = GNNActorCritic(
            node_dim   = self.builder.node_dim,    # 13
            edge_dim   = self.builder.edge_dim,    # 4
            hidden_dim = HIDDEN_DIM,               # 64
            num_joints = self.builder.action_dim,  # 12
        ).to(self.device)

        if isinstance(raw, dict):
            state = raw["agent"] if "agent" in raw else raw
            self.get_logger().info(
                f"Checkpoint step={raw.get('global_step', 'unknown')}"
            )
            model.load_state_dict(state, strict=True)
        elif isinstance(raw, GNNActorCritic):
            return raw.to(self.device)
        else:
            raise RuntimeError(f"Unknown checkpoint type: {type(raw)}")

        n = sum(p.numel() for p in model.parameters())
        self.get_logger().info(f"Model loaded: {n:,} params")
        return model

    # -----------------------------------------------------------------------
    def _cb_joints(self, msg: JointState):
        """Reorder /joint_states into alphabetical order matching builder."""
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        pos = np.zeros(self.builder.num_joints, dtype=np.float32)
        vel = np.zeros(self.builder.num_joints, dtype=np.float32)
        for j, jname in enumerate(self.builder.joint_names):
            if jname in name_to_idx:
                i      = name_to_idx[jname]
                pos[j] = msg.position[i] if msg.position else 0.0
                vel[j] = msg.velocity[i] if msg.velocity else 0.0
        with self._lock:
            self._joint_pos = pos
            self._joint_vel = vel
            self._obs_ready = True

    # -----------------------------------------------------------------------
    def _control_cb(self):
        """20 Hz: obs -> graph -> GNN -> position targets -> publish."""
        if not self._obs_ready:
            return

        with self._lock:
            pos = self._joint_pos.copy()
            vel = self._joint_vel.copy()

        graph = self.builder.get_graph(pos, vel).to(self.device)

        with torch.no_grad():
            # action shape: [1, 12], values in approx [-2, 2] (Normal sample)
            action, _, _, _ = self.model.get_action_and_value(graph)

        # Clip and convert to position targets (matches training env exactly)
        action_np  = np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)
        target_pos = self._nominal + action_np * 0.5    # [12] radians

        for i, name in enumerate(self.builder.joint_names):
            msg      = Float64()
            msg.data = float(target_pos[i])
            self._pubs[name].publish(msg)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--urdf",       required=True)
    parser.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    args, _ = parser.parse_known_args()
    rclpy.init()
    node = GNNPolicyNode(args.checkpoint, args.urdf, args.device)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
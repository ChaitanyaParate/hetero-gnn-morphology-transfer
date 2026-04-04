#!/usr/bin/env python3
"""
gnn_policy_node.py  (corrected)
--------------------------------
Deploys a trained GNN PPO checkpoint into ROS2 Jazzy.

Subscribes  : /joint_states          (sensor_msgs/JointState)
Publishes   : /effort_commands       (std_msgs/Float64MultiArray)
Control Hz  : 20 Hz (timer-driven, non-blocking)

Bugs fixed vs first version
----------------------------
1. GNNActorCritic constructor used wrong kwarg names (node_feature_dim etc.)
   Correct signature: GNNActorCritic(node_dim, edge_dim, hidden_dim, num_joints)

2. URDFGraphBuilder.build() does not exist.
   Correct: instantiate builder, then call builder.get_graph(pos, vel) every step.

3. self.model(graph) called a nonexistent forward().
   Correct: model.get_action_and_value(graph) -> (action, log_prob, entropy, value)

4. Checkpoint loaded with key "model_state_dict" but train_gnn_ppo.py saves key "agent".

5. Action shape comment was wrong.
   get_action_and_value() already strips the body node internally (_joint_embeddings).
   Output shape is [1, 12], NOT [13, 1].

Usage
-----
    # Must use system Python for ROS2 Jazzy
    pyenv shell system
    source /opt/ros/jazzy/setup.bash
    source ~/morpho_ros2_ws/install/setup.bash

    cd /dir/containing/gnn_actor_critic.py/and/urdf_to_graph.py

    python3 gnn_policy_node.py \
        --checkpoint /path/to/gnn_ppo_550912.pt \
        --urdf /path/to/anymal.urdf \
        --device cpu
"""

import argparse
import sys
import threading
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

try:
    from gnn_actor_critic import GNNActorCritic
    from urdf_to_graph import URDFGraphBuilder
except ImportError as e:
    print(f"[FATAL] Cannot import project modules: {e}")
    print("        cd into the directory containing gnn_actor_critic.py and urdf_to_graph.py first.")
    sys.exit(1)
    


# ---------------------------------------------------------------------------
# CONFIGURATION  -- only change these
# ---------------------------------------------------------------------------
CONTROL_HZ          = 20      # Hz, must not block
TORQUE_SCALE        = 40.0    # action in [-1,1] * TORQUE_SCALE = Nm (ANYmal B ~40 Nm)
JOINT_COMMAND_TOPIC = "/effort_commands"   # check: ros2 topic list | grep -i effort
HIDDEN_DIM          = 64      # must match what you trained with (Config.hidden_dim default)


class GNNPolicyNode(Node):

    def __init__(self, checkpoint_path: str, urdf_path: str, device_str: str):
        super().__init__("gnn_policy_node")
        self.device = torch.device(device_str)
        self.get_logger().info(f"Device: {self.device}")

        # ── 1. Build graph structure from URDF (done once) ──────────────────
        # URDFGraphBuilder parses joints, builds static features and edge index.
        # We reuse the same builder every control step via get_graph(pos, vel).
        self.get_logger().info(f"Parsing URDF: {urdf_path}")
        self.builder = URDFGraphBuilder(urdf_path, add_body_node=True)
        self.get_logger().info(
            f"Graph: {self.builder.num_nodes} nodes, "
            f"{self.builder._edge_index.shape[1]} edges, "
            f"{self.builder.num_joints} controllable joints"
        )
        self.get_logger().info(f"Joint order: {self.builder.joint_names}")

        # ── 2. Load checkpoint ───────────────────────────────────────────────
        self.model = self._load_checkpoint(checkpoint_path)
        self.model.eval()

        # ── 3. Runtime state (updated by /joint_states callback) ─────────────
        self._lock      = threading.Lock()
        self._joint_pos = np.zeros(self.builder.num_joints)
        self._joint_vel = np.zeros(self.builder.num_joints)
        self._obs_ready = False

        # ── 4. ROS2 plumbing ─────────────────────────────────────────────────
        self.create_subscription(
            JointState, "/joint_states", self._cb_joint_states, 10
        )
        self._pub = self.create_publisher(
            Float64MultiArray, JOINT_COMMAND_TOPIC, 10
        )
        self.create_timer(1.0 / CONTROL_HZ, self._control_cb)

        self.get_logger().info(
            f"Ready. Publishing {self.builder.num_joints} joint commands "
            f"to '{JOINT_COMMAND_TOPIC}' at {CONTROL_HZ} Hz."
        )

    # -----------------------------------------------------------------------
    def _load_checkpoint(self, path: str) -> GNNActorCritic:
        """
        Handles the checkpoint format produced by train_gnn_ppo.py:
            torch.save({
                "global_step": ...,
                "agent":       agent.state_dict(),   <-- key is "agent"
                "optimizer":   ...,
            }, path)
        Also handles a bare state dict in case you saved differently.
        """
        self.get_logger().info(f"Loading checkpoint: {path}")
        raw = torch.load(path, map_location=self.device, weights_only=False)

        # Instantiate with the EXACT same args used in train_gnn_ppo.py:
        #   GNNActorCritic(
        #       node_dim   = builder.node_dim,    # 13
        #       edge_dim   = builder.edge_dim,    # 4
        #       hidden_dim = cfg.hidden_dim,      # 64 (default in Config)
        #       num_joints = builder.action_dim,  # 12
        #   )
        model = GNNActorCritic(
            node_dim   = self.builder.node_dim,    # 13  (STATIC_DIM=11 + RUNTIME_DIM=2)
            edge_dim   = self.builder.edge_dim,    # 4
            hidden_dim = HIDDEN_DIM,               # 64 -- change if you overrode this
            num_joints = self.builder.action_dim,  # 12
        ).to(self.device)

        if isinstance(raw, dict):
            # Standard training checkpoint from train_gnn_ppo.py
            if "agent" in raw:
                state = raw["agent"]
                step  = raw.get("global_step", "unknown")
                self.get_logger().info(f"Training checkpoint, step={step}")
            else:
                # Bare state dict (e.g. saved with torch.save(model.state_dict(), path))
                state = raw
                self.get_logger().info("Bare state dict checkpoint")
            model.load_state_dict(state, strict=True)

        elif isinstance(raw, GNNActorCritic):
            # Full model object saved with torch.save(model, path)
            self.get_logger().info("Full model object checkpoint")
            return raw.to(self.device)

        else:
            raise RuntimeError(
                f"Unrecognised checkpoint type: {type(raw)}. "
                "Expected dict with key 'agent', a bare state dict, or a GNNActorCritic object."
            )

        n_params = sum(p.numel() for p in model.parameters())
        self.get_logger().info(f"Model loaded: {n_params:,} parameters")
        return model

    # -----------------------------------------------------------------------
    def _cb_joint_states(self, msg: JointState):
        """
        Reorder incoming /joint_states into the alphabetical order that
        URDFGraphBuilder and RobotEnvBullet both use.
        ROS2 does NOT guarantee topic message joint order matches URDF order.
        """
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        pos = np.zeros(self.builder.num_joints)
        vel = np.zeros(self.builder.num_joints)
        missing = []

        for j, jname in enumerate(self.builder.joint_names):
            if jname in name_to_idx:
                i      = name_to_idx[jname]
                pos[j] = msg.position[i] if msg.position else 0.0
                vel[j] = msg.velocity[i] if msg.velocity else 0.0
            else:
                missing.append(jname)

        if missing:
            self.get_logger().warn(
                f"Missing joints in /joint_states: {missing}",
                throttle_duration_sec=5.0,
            )

        with self._lock:
            self._joint_pos = pos
            self._joint_vel = vel
            self._obs_ready = True

    # -----------------------------------------------------------------------
    def _control_cb(self):
        """
        20 Hz control loop:
          joint_states -> PyG graph -> GNN -> action [1, 12] -> publish
        """
        if not self._obs_ready:
            return

        with self._lock:
            pos = self._joint_pos.copy()
            vel = self._joint_vel.copy()

        # builder.get_graph() handles static feat concat + body node correctly.
        # This is the same call used in train_gnn_ppo.py rollout collection.
        graph = self.builder.get_graph(pos, vel).to(self.device)

        with torch.no_grad():
            # get_action_and_value returns: (action, log_prob, entropy, value)
            # action shape: [B, num_joints] = [1, 12]
            # The body node (index 0) is ALREADY excluded by _joint_embeddings()
            # inside GNNActorCritic -- do NOT index into it manually.
            action, _, _, _ = self.model.get_action_and_value(graph)

        # action: [1, 12] -- squeeze batch dim, clip
        action_np = action.squeeze(0).cpu().numpy()   # [12]
        action_np = np.clip(action_np, -1.0, 1.0)

        # ----------------------------------------------------------------
        # CRITICAL: the policy was trained with PD control inside step().
        # action is a POSITION OFFSET from nominal pose, NOT a raw torque.
        # robot_env_bullet.step() does:
        #   target_pos = nominal_pos + action * 0.5
        #   torque = KP * (target_pos - curr_pos) - KD * curr_vel
        #
        # For Gazebo you have two choices:
        #   A) Publish to a JointGroupPositionController topic as target angles
        #      (easiest -- Gazebo's position controller does the PD internally)
        #   B) Implement the PD loop here and publish raw torques to
        #      JointGroupEffortController
        #
        # This node publishes raw [-1,1] offsets for now so you can verify
        # signal flow first. Switch to A or B after confirming connectivity.
        # ----------------------------------------------------------------
        msg = Float64MultiArray()
        msg.data = action_np.tolist()
        self._pub.publish(msg)

        self.get_logger().debug(
            f"Actions: {np.round(action_np, 3).tolist()}",
            throttle_duration_sec=1.0,
        )


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt file")
    parser.add_argument("--urdf",       required=True, help="Path to URDF")
    parser.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    args, _ = parser.parse_known_args()   # _ discards ROS2 remapping args

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
import argparse
import sys
import threading
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
import math
from torch_geometric.data import Batch
try:
    from gnn_actor_critic import GNNActorCritic
    from urdf_to_graph import URDFGraphBuilder
except ImportError as e:
    print(f'[FATAL] Cannot import project modules: {e}')
    sys.exit(1)
CONTROL_HZ = 200
POSITION_SCALE = 0.6
JOINT_COMMAND_FMT = '/model/robot/joint/{}/cmd_pos'
HIDDEN_DIM = 48
NOMINAL_POSE_PER_JOINT = {'LF_HAA': 0.0, 'LF_HFE': 0.6, 'LF_KFE': -1.2, 'RF_HAA': 0.0, 'RF_HFE': 0.6, 'RF_KFE': -1.2, 'LH_HAA': 0.0, 'LH_HFE': -0.6, 'LH_KFE': 1.2, 'RH_HAA': 0.0, 'RH_HFE': -0.6, 'RH_KFE': 1.2}
ACTION_SMOOTH_ALPHA = 0.6   # reduced from 0.8 — allows faster leg motion changes
MAX_CMD_STEP = 0.5          # increased: was 0.2 which suppressed front-leg oscillations
STARTUP_HOLD_TICKS = 400
# Yaw-rate PI controller: corrects circular drift via HAA joint offsets (post-action).
# HAA (hip abductor) joints are nudged to steer the body without changing the GNN command.
YAW_RATE_KP  = 0.35   # conservative P gain — avoid disrupting forward motion
YAW_RATE_KI  = 0.02   # small I gain — slow accumulation to fix steady-state circle
YAW_RATE_LPF = 0.90   # heavy low-pass — smooth out footfall noise in yaw rate

class RunningNorm:

    def __init__(self, shape, clip: float=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.clip = clip

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-08), -self.clip, self.clip).astype(np.float32)

class GNNPolicyNode(Node):

    def __init__(self, checkpoint_path: str, urdf_path: str, device_str: str):
        super().__init__('gnn_policy_node')
        self.device = torch.device(device_str)
        self.get_logger().info(f'Device: {self.device}')
        self.builder = URDFGraphBuilder(urdf_path, add_body_node=True)
        self.get_logger().info(f'Graph Built: {self.builder.num_joints} controllable joints')
        self.obs_norm = RunningNorm(shape=(30,))
        self.model = self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self._lock = threading.Lock()
        self._joint_pos = np.zeros(self.builder.num_joints)
        self._joint_vel = np.zeros(self.builder.num_joints)
        self._raw_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._world_lin_vel = np.zeros(3)
        self._world_ang_vel = np.zeros(3)
        self._prev_action = np.zeros(self.builder.action_dim, dtype=np.float32)
        self._prev_cmd_pos = np.array([NOMINAL_POSE_PER_JOINT.get(j, 0.0) for j in self.builder.joint_names], dtype=np.float32)
        self._target_cmd = np.array([0.5, 0.0], dtype=np.float32)  # default: walk forward until LLM sets a goal
        self._ticks = 0
        self._yaw_rate_filtered = 0.0   # low-pass filtered yaw rate for stable correction
        self._yaw_integral = 0.0        # integral of yaw error for steady-state correction
        self._startup_hold_ticks = STARTUP_HOLD_TICKS
        self._obs_ready = False
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(JointState, '/joint_states', self._cb_joint_states, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._cb_odom, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, '/goal_pose', self._cb_goal_pose, 10)
        self._joint_pubs = {jname: self.create_publisher(Float64, JOINT_COMMAND_FMT.format(jname), 10) for jname in self.builder.joint_names}
        self.create_timer(1.0 / CONTROL_HZ, self._control_cb)
        self.get_logger().info(f'Ready. Spinning {CONTROL_HZ} Hz control loop.')

    def _load_checkpoint(self, path: str) -> GNNActorCritic:
        self.get_logger().info(f'Loading checkpoint: {path}')
        raw = torch.load(path, map_location=self.device, weights_only=False)
        model = GNNActorCritic(node_dim=self.builder.node_dim, edge_dim=self.builder.edge_dim, hidden_dim=HIDDEN_DIM, num_joints=self.builder.action_dim).to(self.device)
        if isinstance(raw, dict):
            if 'agent' in raw:
                model.load_state_dict(raw['agent'], strict=True)
                self.get_logger().info(f'Loaded epoch step {raw.get('global_step', 'unknown')}')
            else:
                model.load_state_dict(raw, strict=True)
            if 'obs_norm_mean' in raw:
                self.obs_norm.mean = raw['obs_norm_mean']
                self.obs_norm.var = raw['obs_norm_var']
                self.get_logger().info('Loaded RunningNorm observation scales.')
        return model

    def _get_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]], dtype=np.float32)

    def _cb_joint_states(self, msg: JointState):
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        p, v = (np.zeros(self.builder.num_joints), np.zeros(self.builder.num_joints))
        for j, jname in enumerate(self.builder.joint_names):
            if jname in name_to_idx:
                i = name_to_idx[jname]
                p[j] = msg.position[i] if msg.position else 0.0
                v[j] = msg.velocity[i] if msg.velocity else 0.0
        with self._lock:
            self._joint_pos = p
            self._joint_vel = v
            self._obs_ready = True

    def _cb_odom(self, msg: Odometry):
        with self._lock:
            self._world_lin_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            self._world_ang_vel = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
            self._raw_quat = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

    def _cb_goal_pose(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        dist = math.hypot(x, y)
        if dist < 0.1:
            vx = 0.0
            wy = 0.0
        else:
            vx = min(1.0, max(-0.5, x))
            wy = min(1.0, max(-1.0, math.atan2(y, x)))
        with self._lock:
            self._target_cmd = np.array([vx, wy], dtype=np.float32)
        
        action_str = "standing still"
        if vx > 0.1: action_str = "walking forward"
        elif vx < -0.1: action_str = "walking backward"
        elif abs(wy) > 0.1: action_str = "turning"
        
        self.get_logger().info(f'Received new command: [vx={vx:.2f}, wy={wy:.2f}] -> Robot should be {action_str}')

    def _control_cb(self):
        if not self._obs_ready:
            return
        with self._lock:
            p = self._joint_pos.copy()
            v = self._joint_vel.copy()
            raw_quat = self._raw_quat.copy()
            world_lin_vel = self._world_lin_vel.copy()
            world_ang_vel = self._world_ang_vel.copy()
            prev_cmd = self._prev_cmd_pos.copy()
            target_cmd = self._target_cmd.copy()
        if self._startup_hold_ticks > 0:
            self._startup_hold_ticks -= 1
            for i, jname in enumerate(self.builder.joint_names):
                msg = Float64()
                msg.data = float(prev_cmd[i])
                self._joint_pubs[jname].publish(msg)
            return
        if self._ticks == 0:
            self.get_logger().info('🚀 GNN POLICY ACTIVE: Taking control.')
        self._ticks += 1
        rot_mat = self._get_rotation_matrix(raw_quat)
        blv = rot_mat.T @ world_lin_vel
        bav = rot_mat.T @ world_ang_vel
        bgv = rot_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        r = R.from_quat(raw_quat)
        euler = r.as_euler('xyz')
        neutral_r = R.from_euler('xyz', [euler[0], euler[1], 0.0])
        bqv = neutral_r.as_quat().astype(np.float32)
        raw_state = np.concatenate([p, v, blv, bav]).astype(np.float32)
        norm_state = self.obs_norm.normalize(raw_state)
        n_p = norm_state[0:12]
        n_v = norm_state[12:24]
        n_blv = norm_state[24:27]
        n_bav = norm_state[27:30]
        # Feed GNN the command, clamped to plausible ranges.
        # vx: clamp to training range [0.5, 1.0] for forward walking.
        # wy: allow small turning signal (clamped to ±0.3) — completely zeroing wy
        #     prevents the robot from turning when commanded. The GNN only saw wy=0
        #     during training, but small wy values are close to its distribution.
        gnn_cmd = np.array([
            float(np.clip(target_cmd[0], 0.5, 1.0)),          # clamp vx to training range
            float(np.clip(target_cmd[1], -0.3, 0.3)),         # allow small wy for turns
        ], dtype=np.float32)
        graph = self.builder.get_graph(joint_pos=n_p, joint_vel=n_v, body_quat=bqv, body_grav=bgv, body_lin_vel=n_blv, body_ang_vel=n_bav, command=gnn_cmd)
        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.no_grad():
            joint_h = self.model._joint_embeddings(self.model._encode(batch)[0], batch)
            action = self.model.actor_head(joint_h).view(1, self.model.num_joints)
        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)

        # --- Yaw-rate feedback (post-action HAA correction) ---
        # KEY: GNN was trained with wy=ALWAYS 0.0, so injecting wy != 0 is OOD.
        # Instead, correct yaw drift by offsetting the HAA (hip abductor) joints
        # directly AFTER getting the GNN action. This is in-distribution.
        actual_yaw_rate = float(bav[2])
        self._yaw_rate_filtered = (
            YAW_RATE_LPF * self._yaw_rate_filtered
            + (1.0 - YAW_RATE_LPF) * actual_yaw_rate
        )
        yaw_error = self._yaw_rate_filtered - float(target_cmd[1])  # target_cmd[1] is always ~0
        # Integrate yaw error (clamped to prevent windup)
        self._yaw_integral = float(np.clip(self._yaw_integral + yaw_error * 0.005, -0.3, 0.3))
        # HAA correction: when drifting right (-yaw_rate), abduct left legs more
        # and adduct right legs — pushes body back left.
        # HAA joint indices: LF=0, LH=3, RF=6, RH=9
        haa_correction = float(np.clip(
            YAW_RATE_KP * yaw_error + YAW_RATE_KI * self._yaw_integral,
            -0.12, 0.12))  # conservative clamp — large values block forward motion
        action_np[[0, 3]] -= haa_correction   # left legs: +correction when drifting right
        action_np[[6, 9]] += haa_correction   # right legs: −correction when drifting right
        action_np = np.clip(action_np, -1.0, 1.0)
        # Clamp vx to training range [0.5, 1.0] — GNN never saw vx < 0.5
        vx_cmd = float(np.clip(target_cmd[0], 0.5, 1.0))
        if self._ticks % 200 == 0:
            self.get_logger().info(
                f'yaw={actual_yaw_rate:.3f} filt={self._yaw_rate_filtered:.3f} '
                f'err={yaw_error:.3f} integ={self._yaw_integral:.3f} '
                f'haa_corr={haa_correction:+.3f} vx={vx_cmd:.2f}'
            )
        # Apply action smoothing
        action_np = (1.0 - ACTION_SMOOTH_ALPHA) * self._prev_action + ACTION_SMOOTH_ALPHA * action_np
        self._prev_action = action_np.copy()
        ramp_ticks = 400
        ramp_factor = max(0.0, min(1.0, float(self._ticks) / ramp_ticks))
        cmd_pos = []
        for i, jname in enumerate(self.builder.joint_names):
            nominal = NOMINAL_POSE_PER_JOINT.get(jname, 0.0)
            target = nominal + float(action_np[i]) * POSITION_SCALE * ramp_factor
            delta = target - float(self._prev_cmd_pos[i])
            delta = float(np.clip(delta, -MAX_CMD_STEP, MAX_CMD_STEP))
            target = float(self._prev_cmd_pos[i] + delta)
            self._prev_cmd_pos[i] = target
            cmd_pos.append(target)
            msg = Float64()
            msg.data = float(target)
            self._joint_pubs[jname].publish(msg)
        self.get_logger().debug(f'Action (rad): {np.round(cmd_pos, 3)}', throttle_duration_sec=1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args, _ = parser.parse_known_args()
    rclpy.init()
    node = GNNPolicyNode(args.checkpoint, args.urdf, args.device)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
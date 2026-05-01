import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))

def parse_target_token(target: str) -> Tuple[str, Optional[int]]:
    m = re.fullmatch('([A-Za-z0-9\\-]+)(?:_(\\d+))?', target.strip())
    if not m:
        return (target.strip(), None)
    label = m.group(1)
    idx = int(m.group(2)) if m.group(2) is not None else None
    return (label, idx)

class SkillTranslatorNode(Node):

    def __init__(self):
        super().__init__('skill_translator_node')
        self.declare_parameter('llm_action_topic', '/llm_action')
        self.declare_parameter('scene_graph_topic', '/scene_graph')
        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter('active_skill_topic', '/active_skill')
        self.declare_parameter('goal_frame_id', 'base_link')
        self.declare_parameter('camera_half_fov_rad', 0.5236)
        self.declare_parameter('stop_distance_m', 0.7)
        self.declare_parameter('default_forward_goal_m', 1.0)
        self.llm_action_topic = self.get_parameter('llm_action_topic').value
        self.scene_graph_topic = self.get_parameter('scene_graph_topic').value
        self.goal_pose_topic = self.get_parameter('goal_pose_topic').value
        self.active_skill_topic = self.get_parameter('active_skill_topic').value
        self.goal_frame_id = self.get_parameter('goal_frame_id').value
        self.camera_half_fov_rad = float(self.get_parameter('camera_half_fov_rad').value)
        self.stop_distance_m = float(self.get_parameter('stop_distance_m').value)
        self.default_forward_goal_m = float(self.get_parameter('default_forward_goal_m').value)
        self._latest_scene: Dict[str, Any] = {}
        self.create_subscription(String, self.scene_graph_topic, self._scene_cb, 10)
        self.create_subscription(String, self.llm_action_topic, self._action_cb, 10)
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_pose_topic, 10)
        self.skill_pub = self.create_publisher(String, self.active_skill_topic, 10)
        self.get_logger().info(f'skill_translator_node ready | in_action={self.llm_action_topic} | in_scene={self.scene_graph_topic} | out_goal={self.goal_pose_topic} | out_skill={self.active_skill_topic}')

    def _scene_cb(self, msg: String) -> None:
        try:
            scene = json.loads(msg.data)
            if isinstance(scene, dict):
                self._latest_scene = scene
            else:
                self.get_logger().warn('Scene graph JSON is not an object', throttle_duration_sec=5.0)
        except Exception as exc:
            self.get_logger().warn(f'Invalid scene_graph JSON: {exc}', throttle_duration_sec=5.0)

    def _action_cb(self, msg: String) -> None:
        try:
            action = json.loads(msg.data)
            if not isinstance(action, dict):
                raise ValueError('llm_action must be a JSON object')
        except Exception as exc:
            self.get_logger().warn(f'Invalid llm_action JSON: {exc}', throttle_duration_sec=5.0)
            return
        skill = str(action.get('skill', 'unknown'))
        target = str(action.get('target', '')).strip()
        params = action.get('params', {})
        if not isinstance(params, dict):
            params = {}
        self._publish_active_skill(skill)
        goal_xy = self._extract_goal_from_params(params)
        if goal_xy is None:
            goal_xy = self._resolve_goal_from_scene(target)
        if goal_xy is None:
            goal_xy = self._fallback_goal(skill, target)
        if goal_xy is None:
            self.get_logger().warn(f'No goal resolved for target="{target}". Need params {{x,y}} or a matching object in scene_graph.', throttle_duration_sec=3.0)
            return
        self._publish_goal_pose(goal_xy[0], goal_xy[1])

    def _publish_active_skill(self, skill: str) -> None:
        out = String()
        out.data = skill
        self.skill_pub.publish(out)

    def _extract_goal_from_params(self, params: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        key_pairs = [('x', 'y'), ('goal_x', 'goal_y'), ('target_x', 'target_y')]
        for kx, ky in key_pairs:
            if kx in params and ky in params:
                try:
                    return (float(params[kx]), float(params[ky]))
                except Exception:
                    return None
        p_objects = params.get('objects')
        if isinstance(p_objects, list) and p_objects:
            goal = self._goal_from_object(p_objects[0])
            if goal is not None:
                return goal
        return None

    def _resolve_goal_from_scene(self, target: str) -> Optional[Tuple[float, float]]:
        objects = self._latest_scene.get('objects', [])
        if not isinstance(objects, list) or not objects:
            return None
        label, idx = parse_target_token(target)
        candidates: List[Dict[str, Any]] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if label and str(obj.get('label', '')).lower() != label.lower():
                continue
            candidates.append(obj)
        if not candidates:
            return None
        candidates.sort(key=lambda o: float(o.get('confidence', 0.0)), reverse=True)
        if idx is None:
            obj = candidates[0]
        else:
            sel = max(1, idx) - 1
            if sel >= len(candidates):
                obj = candidates[-1]
            else:
                obj = candidates[sel]
        return self._goal_from_object(obj)

    def _goal_from_object(self, obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        try:
            distance = float(obj.get('distance_m', 0.0))
            bearing = float(obj.get('bearing', 0.0))
        except Exception:
            return None
        yaw = max(-1.0, min(1.0, bearing)) * self.camera_half_fov_rad
        forward = max(0.0, distance - self.stop_distance_m)
        x = forward * math.cos(yaw)
        y = forward * math.sin(yaw)
        return (x, y)

    def _fallback_goal(self, skill: str, target: str) -> Optional[Tuple[float, float]]:
        s = skill.strip().lower()
        t = target.strip().lower()
        # Forward/navigate fallbacks
        if s.startswith('navigate') and t in ('goal', 'forward', ''):
            self.get_logger().warn(f'Using fallback forward goal: x={self.default_forward_goal_m:.2f}, y=0.00', throttle_duration_sec=3.0)
            return (self.default_forward_goal_m, 0.0)
        # Explicit skill mappings -> (x_forward, y_lateral) in base_link
        # GNN policy's _cb_goal_pose converts these to: vx=min(1,x), wy=atan2(y,x)
        if s in ('trot', 'walk', 'move_forward', 'forward', 'go_forward'):
            return (self.default_forward_goal_m, 0.0)
        if s in ('turn_left', 'rotate_left', 'rotate_ccw'):
            # Gentle arc-left: wy = atan2(0.4, 0.6) ≈ 34° — curves left without spinning
            return (0.6, 0.4)
        if s in ('turn_right', 'rotate_right', 'rotate_cw'):
            return (0.6, -0.4)
        if s in ('stop', 'stand', 'halt', 'standstill'):
            # Send a very small forward goal so robot holds stance, not zero (zero causes drift)
            return (0.05, 0.0)
        if s in ('backward', 'move_backward', 'reverse'):
            return (-0.3, 0.0)
        # Generic fallback: go forward
        self.get_logger().warn(f'Unknown skill "{skill}": defaulting to forward goal.', throttle_duration_sec=5.0)
        return (self.default_forward_goal_m, 0.0)

    def _publish_goal_pose(self, x: float, y: float) -> None:
        yaw = math.atan2(y, x) if abs(x) + abs(y) > 1e-06 else 0.0
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.goal_frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.goal_pub.publish(msg)
        self.get_logger().info(f'Published goal_pose: frame={self.goal_frame_id}, x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = SkillTranslatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
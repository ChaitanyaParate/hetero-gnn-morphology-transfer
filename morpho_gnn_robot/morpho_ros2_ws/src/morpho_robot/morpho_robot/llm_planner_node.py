import json
import time
import ollama
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def reactive_fallback(scene: dict) -> dict:
    """Rule-based planner used when Ollama is unavailable. Reads depth sensor data."""
    dist = scene.get('obstacle_distances', {})
    front   = dist.get('front',   10.0)
    closest = dist.get('closest', 10.0)
    left    = dist.get('left',    10.0)
    right   = dist.get('right',   10.0)
    TURN_DIST     = 1.2  # m — only turn when obstacle is truly close (avoids triggering on own legs at ~1.5m)
    SIDE_WALL_MIN = 0.8  # m — min clearance before blocking a turn direction
    # When obstacle is ahead, turn toward open side.
    if front < TURN_DIST or closest < TURN_DIST:
        prefer_left   = left >= right
        right_blocked = right < SIDE_WALL_MIN  # right wall would cause roll when turning left
        left_blocked  = left  < SIDE_WALL_MIN  # left wall would cause roll when turning right
        if prefer_left and not right_blocked:
            skill = 'turn_left'   # open left, right side has clearance
        elif not prefer_left and not left_blocked:
            skill = 'turn_right'  # open right, left side has clearance
        else:
            skill = 'turn_left' if left >= right else 'turn_right'
        return {'skill': skill, 'target': 'obstacle', 'params': {}}
    return {'skill': 'trot', 'target': 'waypoint', 'params': {'x': 5.0, 'y': 0.0, 'velocity': 0.35}}

def call_llm_planner(task: str, scene: dict, model: str) -> dict:
    try:
        response = ollama.chat(model=model, format='json', options={'temperature': 0.1}, messages=[{'role': 'system', 'content': 'You are a robot planner. Output ONLY JSON with keys: skill (string), target (string), params (dict).'}, {'role': 'user', 'content': f'Task: {task}\nScene: {json.dumps(scene)}'}])
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f'[WARN] LLM unavailable: {e}. Using reactive obstacle avoidance.')
        return reactive_fallback(scene)

class LLMPlannerNode(Node):

    def __init__(self):
        super().__init__('llm_planner_node')
        self.declare_parameter('task', 'navigate to the goal')
        self.declare_parameter('llm_model', 'qwen2.5:7b')
        self.declare_parameter('replan_interval_s', 5.0)
        self.declare_parameter('scene_graph_topic', '/scene_graph')
        self.declare_parameter('action_topic', '/llm_action')
        self.task = self.get_parameter('task').get_parameter_value().string_value
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        self.interval = self.get_parameter('replan_interval_s').get_parameter_value().double_value
        self.scene_graph_topic = self.get_parameter('scene_graph_topic').get_parameter_value().string_value
        self.action_topic = self.get_parameter('action_topic').get_parameter_value().string_value
        self.sub = self.create_subscription(String, self.scene_graph_topic, self.scene_callback, 10)
        self.pub = self.create_publisher(String, self.action_topic, 10)
        self.get_logger().info(f'llm_planner_node running | model={self.llm_model} | in={self.scene_graph_topic} | out={self.action_topic} | interval={self.interval:.1f}s')
        self.last_llm_call = 0   # throttle for slow LLM inference
        self.last_react_call = 0 # throttle for fast reactive fallback (0.5s)

    def scene_callback(self, msg):
        try:
            now = time.time()
            scene = json.loads(msg.data)

            # Fast reactive path: run every 0.5s regardless of LLM throttle
            # This ensures obstacle detection reacts within 0.5s (~0.175m at 0.35m/s)
            if now - self.last_react_call >= 0.5:
                self.last_react_call = now
                plan = reactive_fallback(scene)

                # Only call LLM if: no obstacle urgency AND enough time has passed
                dist = scene.get('obstacle_distances', {})
                front = dist.get('front', 10.0)
                closest = dist.get('closest', 10.0)
                path_clear = front > 3.0 and closest > 3.0
                if path_clear and now - self.last_llm_call >= self.interval:
                    self.last_llm_call = now
                    plan = call_llm_planner(self.task, scene, self.llm_model)

                out = String()
                out.data = json.dumps(plan)
                self.pub.publish(out)
                self.get_logger().info(f'Plan published: {out.data}')
        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
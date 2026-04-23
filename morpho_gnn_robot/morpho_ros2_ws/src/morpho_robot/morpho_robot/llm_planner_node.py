import json
import time
import ollama
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def call_llm_planner(task: str, scene: dict, model: str) -> dict:
    try:
        response = ollama.chat(model=model, format='json', options={'temperature': 0.1}, messages=[{'role': 'system', 'content': 'You are a robot planner. Output ONLY JSON with keys: skill (string), target (string), params (dict).'}, {'role': 'user', 'content': f'Task: {task}\nScene: {json.dumps(scene)}'}])
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f'[WARN] Local LLM server directly unavailable: {e}. Outputting default navigation fallback.')
        return {'skill': 'trot', 'target': 'waypoint', 'params': {'x': 5.0, 'y': 0.0, 'velocity': 0.35}}

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
        self.last_call = 0

    def scene_callback(self, msg):
        try:
            if time.time() - self.last_call < self.interval:
                return
            self.last_call = time.time()
            scene = json.loads(msg.data)
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
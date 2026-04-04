import rclpy, ollama, json
from rclpy.node import Node
from std_msgs.msg import String

def call_llm_planner(task: str, scene: dict) -> dict:
    response = ollama.chat(
        model="qwen2.5:7b",
        format="json",
        options={"temperature": 0.1},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a robot task planner. Output a JSON object with exactly "
                    "two keys: 'plan' (list of steps) and 'status' (string). "
                    "Each step must have: skill (string), target (string), "
                    "constraints (dict with 'avoid' list of strings)."
                )
            },
            {
                "role": "user",
                "content": f"Task: {task}\nScene: {json.dumps(scene)}"
            }
        ]
    )
    return json.loads(response["message"]["content"])

class LLMPlannerNode(Node):
    def __init__(self):
        super().__init__('llm_planner_node')
        self.declare_parameter('task', 'navigate to the goal')
        self.task = self.get_parameter('task').get_parameter_value().string_value

        self.sub = self.create_subscription(
            String, '/scene_graph', self.scene_callback, 10)
        self.pub = self.create_publisher(String, '/llm_plan', 10)
        self.get_logger().info('llm_planner_node running')

    def scene_callback(self, msg):
        try:
            scene = json.loads(msg.data)
            plan = call_llm_planner(self.task, scene)
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
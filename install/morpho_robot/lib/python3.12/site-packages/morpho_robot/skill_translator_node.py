import rclpy
from rclpy.node import Node

class SkillTranslatorNode(Node):
    def __init__(self):
        super().__init__('skill_translator_node')
        self.get_logger().info('skill_translator_node stub running')

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(SkillTranslatorNode())
    rclpy.shutdown()
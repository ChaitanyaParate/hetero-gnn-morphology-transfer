import rclpy
from rclpy.node import Node

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info('vision_node stub running')

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisionNode())
    rclpy.shutdown()
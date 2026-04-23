import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import torch
from urdf_to_graph import data as base_graph

class GraphStateUpdater(Node):

    def __init__(self):
        super().__init__('graph_state_updater')
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.graph = base_graph
        self.joint_to_node = {}

    def joint_callback(self, msg):
        num_nodes = self.graph.x.shape[0]
        dynamic = torch.zeros((num_nodes, 3))
        for i, name in enumerate(msg.name):
            if name not in self.joint_to_node:
                continue
            node_idx = self.joint_to_node[name]
            pos = msg.position[i] if i < len(msg.position) else 0.0
            vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
            eff = msg.effort[i] if i < len(msg.effort) else 0.0
            dynamic[node_idx] = torch.tensor([pos, vel, eff])
        self.graph.x = torch.cat([base_graph.x, dynamic], dim=1)
        self.publish_graph()

    def publish_graph(self):
        self.get_logger().info(f'Graph updated: {self.graph.x.shape}')

def main(args=None):
    rclpy.init(args=args)
    node = GraphStateUpdater()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
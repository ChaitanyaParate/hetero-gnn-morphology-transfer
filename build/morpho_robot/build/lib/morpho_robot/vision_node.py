#!/usr/bin/env python3
# vision_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import json
import numpy as np
import argparse
import cv2
class VisionNode(Node):
    def __init__(self, args):
        super().__init__('vision_node')
        
        self.bridge = CvBridge()
        self.model = YOLO(args.yolo_model)  # downloads on first run (~6MB)
        
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(String, '/scene_graph', 10)
        
        # Approximate focal length for distance estimation (tune per camera)
        self.focal_length_px = 554.0  # for 640px wide, 60deg FOV
        
        self.get_logger().info('Vision node ready')

    def estimate_distance(self, bbox_height_px, real_height_m=0.3):
        # Pinhole model: distance = (real_height * focal_length) / bbox_height
        if bbox_height_px < 1:
            return -1.0
        return (real_height_m * self.focal_length_px) / bbox_height_px

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge failed: {e}')
            return

        results = self.model(cv_image, verbose=False)[0]

        annotated = results.plot()
        cv2.imshow("detections", annotated)
        cv2.waitKey(1)

        objects = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if confidence < 0.4:
                continue

            bbox_height = y2 - y1
            cx = (x1 + x2) / 2.0
            # Normalize x position: -1 (left) to +1 (right)
            bearing = (cx - 320.0) / 320.0
            distance = self.estimate_distance(bbox_height)

            objects.append({
                'label': label,
                'confidence': round(confidence, 2),
                'bearing': round(bearing, 3),
                'distance_m': round(distance, 2),
                'bbox': [round(x1), round(y1), round(x2), round(y2)]
            })

        scene = {'objects': objects, 'timestamp': msg.header.stamp.sec}
        
        out = String()
        out.data = json.dumps(scene)
        self.pub.publish(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.4)
    args = parser.parse_args()

    rclpy.init()
    node = VisionNode(args)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
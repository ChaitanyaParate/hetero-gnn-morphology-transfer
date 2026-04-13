#!/usr/bin/env python3
# vision_node.py

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import json
import numpy as np
import argparse
#export QT_QPA_FONTDIR=/usr/share/fonts
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'
import cv2
from sensor_msgs.msg import CameraInfo
class VisionNode(Node):
    def __init__(self, args):
        super().__init__('vision_node')
        
        self.bridge = CvBridge()
        self.model = YOLO(args.yolo_model)  # downloads on first run (~6MB)
        self.conf_threshold = args.conf
        
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            qos_profile_sensor_data
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.cam_info_callback,
            qos_profile_sensor_data
        )

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.depth_image = None
        self._has_display = bool(os.environ.get('DISPLAY'))
        self.pub = self.create_publisher(String, '/scene_graph', 10)
        
        # Approximate focal length for distance estimation (tune per camera)
        self.focal_length_px = 554.0  # for 640px wide, 60deg FOV
        
        self.get_logger().info('Vision node ready')
    
    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)
            self.depth_image = depth
            self.get_logger().info('Depth frame received', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def cam_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Convert RGB → BGR manually if needed
            if cv_image.shape[-1] == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            cv_image = cv2.convertScaleAbs(cv_image, alpha=1.5, beta=20)
            print(cv_image.min(), cv_image.max())
            
        except Exception as e:
            self.get_logger().error(f'cv_bridge failed: {e}')
            return

        depth_snapshot = None if self.depth_image is None else self.depth_image.copy()

        if depth_snapshot is None:
            self.get_logger().warn('Depth NOT received yet', throttle_duration_sec=1.0)
            if self._has_display:
                h, w = cv_image.shape[:2]
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.imshow("depth", blank)
                cv2.imshow("walls", blank)
                cv2.waitKey(1)
            return

        results = self.model(cv_image, verbose=False)[0]

        if self._has_display:
            cv2.imshow("raw_rgb", cv_image)

        annotated = results.plot()

        if depth_snapshot is not None:
            h, w = depth_snapshot.shape

            # Method 1: center ray distance
            center_u = w // 2
            center_v = h // 2
            center_depth = float(depth_snapshot[center_v, center_u])

            # Method 2: closest obstacle in center ROI
            roi = depth_snapshot[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
            valid_roi = roi[np.isfinite(roi) & (roi > 0)]
            min_dist = float(np.min(valid_roi)) if valid_roi.size > 0 else float('nan')

            # Method 3: left/right depth medians (Cropped to middle-height to ignore ground)
            h_start, h_end = h // 3, 2 * h // 3
            left = depth_snapshot[h_start:h_end, :w // 3]
            right = depth_snapshot[h_start:h_end, 2 * w // 3:]
            left_valid = left[np.isfinite(left) & (left > 0)]
            right_valid = right[np.isfinite(right) & (right > 0)]
            left_dist = float(np.median(left_valid)) if left_valid.size > 0 else float('nan')
            right_dist = float(np.median(right_valid)) if right_valid.size > 0 else float('nan')

            self.get_logger().info(
                f"Front: {center_depth:.2f} m | Closest: {min_dist:.2f} m | "
                f"Left: {left_dist:.2f} m | Right: {right_dist:.2f} m",
                throttle_duration_sec=1.0,
            )

            if self._has_display:
                cv2.circle(annotated, (center_u, center_v), 5, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f"Front {center_depth:.2f}m",
                    (center_u + 10, center_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        if depth_snapshot is not None and self._has_display:
            depth_vis = depth_snapshot.copy()
            depth_vis[np.isnan(depth_vis)] = 0
            depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("depth", depth_vis)

            wall_mask = (depth_snapshot < 2.0) & (depth_snapshot > 0)
            mask_vis = wall_mask.astype(np.uint8) * 255
            cv2.imshow("walls", mask_vis)

        objects = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if confidence < self.conf_threshold:
                continue

            cx = (x1 + x2) / 2.0
            # Normalize x position: -1 (left) to +1 (right)

            if depth_snapshot is not None and self.fx is not None:
                u = int(cx)
                v = int((y1 + y2) / 2.0)

                h, w = depth_snapshot.shape

                if u < 0 or u >= w or v < 0 or v >= h:
                    continue

                u_min = max(u - 2, 0)
                u_max = min(u + 3, w)
                v_min = max(v - 2, 0)
                v_max = min(v + 3, h)

                window = depth_snapshot[v_min:v_max, u_min:u_max]
                Z = np.nanmedian(window)

                if np.isnan(Z) or Z <= 0:
                    continue

                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
            else:
                continue

            if self._has_display:
                cv2.circle(annotated, (u, v), 4, (0, 255, 0), -1)
                cv2.putText(
                    annotated,
                    f"{Z:.2f}m",
                    (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            objects.append({
                'label': label,
                'confidence': round(confidence, 2),
                'position': {
                    'x': round(float(X), 3),
                    'y': round(float(Y), 3),
                    'z': round(float(Z), 3)
                },
                'bbox': [round(x1), round(y1), round(x2), round(y2)]
            })

        if self._has_display:
            cv2.imshow("detections", annotated)
            cv2.waitKey(1)

        scene = {
            'objects': objects,
            'timestamp': int(msg.header.stamp.sec),
            'obstacle_distances': {
                'left': float(round(left_dist, 3)) if not np.isnan(left_dist) else 10.0,
                'right': float(round(right_dist, 3)) if not np.isnan(right_dist) else 10.0,
                'front': float(round(center_depth, 3)) if not np.isnan(center_depth) else 10.0,
                'closest': float(round(min_dist, 3)) if not np.isnan(min_dist) else 10.0
            }
        }
        
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
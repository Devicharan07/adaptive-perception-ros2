#!/usr/bin/env python3

import os
import csv
import time
from datetime import datetime
import cv2
import numpy as np
import psutil
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge, CvBridgeError

from adaptive_perception.detector import ArUcoDetectorModule
from adaptive_perception.profiler import Profiler
from adaptive_perception.confidence import ConfidenceEvaluator
from adaptive_perception.pose import PoseEstimator

# --- EVALUATION LOGGING CONFIGURATION ---
ENABLE_CSV_LOG = True
CSV_LOG_DIR = os.path.expanduser("~/Adaptive_Perception_Project/experiments/logs")


class BaselinePerceptionNode(Node):
    def __init__(self):
        super().__init__('baseline_perception_node')

        self.bridge = CvBridge()

        # System Metrics Monitor Initialization
        self.process = psutil.Process(os.getpid())

        # Parameters
        self.declare_parameter('marker_length', 0.15)
        self.marker_length = self.get_parameter('marker_length').value

        # Baseline Pipeline Modules (Identical to Adaptive)
        self.profiler = Profiler()
        self.detector_module = ArUcoDetectorModule(dictionary_name='DICT_ARUCO_ORIGINAL')
        self.confidence_evaluator = ConfidenceEvaluator()
        self.pose_estimator = PoseEstimator()

        # Fixed Baseline Constants
        self.SUBPIX_CONFIDENCE = 0.75

        # State Buffers
        self.camera_matrix = None
        self.dist_coeffs = None
        self.frame_counter = 0
        self.last_frame_time = None
        self.use_subpix_last = False  # Synchronized subpix decision buffer from previous frame

        # ROS Communications (Identical Topics)
        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)

        self.pub_pose = self.create_publisher(
            PoseStamped, '/adaptive_perception/pose', 10)
        self.pub_debug_img = self.create_publisher(
            Image, '/adaptive_perception/debug_image', 10)
        self.pub_latency = self.create_publisher(
            Float32, '/adaptive_perception/latency_ms', 10)
        self.pub_status = self.create_publisher(
            String, '/adaptive_perception/status', 10)

        # CSV Logging Initialization (Identical Schema + pipeline_mode + ROI Dimensions)
        self.csv_path = None
        if ENABLE_CSV_LOG:
            os.makedirs(CSV_LOG_DIR, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file_name = f"baseline_hover_{timestamp_str}.csv"
            self.csv_path = os.path.join(CSV_LOG_DIR, csv_file_name)
            
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame_id', 'timestamp', 'detection_success', 'pose_success', 'confidence', 
                    'latency_ms', 'mode', 'alpha', 'scale', 'subpix', 'streak', 
                    'marker_x', 'marker_y', 'marker_size_px',
                    'fps', 'cpu_percent', 'memory_percent', 'roi_switch', 
                    'miss_count', 'marker_area_px', 'bbox_width_px', 'bbox_height_px',
                    'pipeline_mode',
                    'roi_x', 'roi_y', 'roi_width', 'roi_height'
                ])
            self.get_logger().info(f"Logging baseline experiment to: {self.csv_path}")

        self.get_logger().info("Baseline Perception Pipeline: Fixed Full-Resolution Execution Active.")

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)

    def image_callback(self, msg: Image):
        current_time = time.time()
        self.frame_counter += 1
        
        # Calculate instantaneous FPS
        fps = 0.0
        if self.last_frame_time is not None:
            time_delta = current_time - self.last_frame_time
            if time_delta > 0:
                fps = 1.0 / time_delta
        self.last_frame_time = current_time

        self.profiler.start()

        try:
            if msg.encoding == 'rgb8':
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # 1. Stage 1: Full Image Processing Input (Fixed Scale = 1.0)
        input_image = cv_img
        img_h, img_w = cv_img.shape[:2]

        roi_x = 0
        roi_y = 0
        roi_width = int(img_w)
        roi_height = int(img_h)

        # 2. Perception Detection
        success, corners, ids = self.detector_module.detect(input_image)
        if success and corners is not None:
            corners = list(corners)

        primary_corners = corners[0] if (success and len(corners) > 0) else None

        # Calculate spatial & geometric metrics for experiment logging
        marker_x, marker_y, marker_size_px = -1.0, -1.0, -1.0
        bbox_width_px, bbox_height_px, marker_area_px = -1.0, -1.0, -1.0

        if success and primary_corners is not None:
            pts = np.squeeze(primary_corners)
            center = np.mean(pts, axis=0)
            marker_x, marker_y = float(center[0]), float(center[1])
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            bbox_width_px = float(x_max - x_min)
            bbox_height_px = float(y_max - y_min)
            marker_size_px = float(max(bbox_width_px, bbox_height_px))
            marker_area_px = float(bbox_width_px * bbox_height_px)

        # 3. Corner Refinement Mode (Evaluates use_subpix_last from previous frame state)
        subpix_applied_this_frame = False
        if success and self.use_subpix_last and primary_corners is not None:
            subpix_applied_this_frame = True
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            pts = np.squeeze(primary_corners).astype(np.float32)
            refined_pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
            primary_corners = refined_pts.reshape(1, 4, 2)
            corners[0] = primary_corners

        # 4. Continuous Confidence Calculation
        confidence = self.confidence_evaluator.compute_confidence(corners, cv_img.shape)

        # 5. Update SubPix decision buffer for the NEXT frame (Matching adaptive controller timing)
        if confidence < self.SUBPIX_CONFIDENCE:
            self.use_subpix_last = True
        else:
            self.use_subpix_last = False

        # 6. Pose Estimation (Identical)
        pose_success = False
        rvec, tvec, quat = None, None, None
        if success and self.camera_matrix is not None:
            pose_success, rvec, tvec, quat = self.pose_estimator.estimate(
                corners, self.camera_matrix, self.dist_coeffs, self.marker_length
            )

            if pose_success and quat is not None:
                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                
                tvec_flat = np.array(tvec, dtype=np.float64).flatten()
                quat_flat = np.array(quat, dtype=np.float64).flatten()

                pose_msg.pose.position.x = tvec_flat[0]
                pose_msg.pose.position.y = tvec_flat[1]
                pose_msg.pose.position.z = tvec_flat[2]

                pose_msg.pose.orientation.x = quat_flat[0]
                pose_msg.pose.orientation.y = quat_flat[1]
                pose_msg.pose.orientation.z = quat_flat[2]
                pose_msg.pose.orientation.w = quat_flat[3]

                self.pub_pose.publish(pose_msg)

        # 7. Profiling & Telemetry
        latency_ms = self.profiler.stop()

        lat_msg = Float32(data=latency_ms)
        self.pub_latency.publish(lat_msg)

        status_str = f"Mode: FULL_IMAGE | Alpha: 2.00 | Scale: 1.0x | Subpix: {subpix_applied_this_frame} | Latency: {latency_ms:.2f}ms"
        self.pub_status.publish(String(data=status_str))

        # 8. CSV Logging Step (Identical Schema + ROI Dimensions)
        if ENABLE_CSV_LOG and self.csv_path:
            ts = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            cpu_pct = self.process.cpu_percent(interval=None)
            mem_pct = self.process.memory_percent()

            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.frame_counter,
                    f"{ts:.4f}",
                    int(success),
                    int(pose_success),
                    f"{confidence:.3f}",
                    f"{latency_ms:.2f}",
                    "FULL_IMAGE",
                    "2.00",
                    "1.0",
                    int(subpix_applied_this_frame),
                    0,  # streak = 0
                    f"{marker_x:.1f}",
                    f"{marker_y:.1f}",
                    f"{marker_size_px:.1f}",
                    f"{fps:.2f}",
                    f"{cpu_pct:.1f}",
                    f"{mem_pct:.2f}",
                    0,  # roi_switch = 0
                    0,  # miss_count = 0
                    f"{marker_area_px:.1f}",
                    f"{bbox_width_px:.1f}",
                    f"{bbox_height_px:.1f}",
                    "BASELINE",
                    roi_x,
                    roi_y,
                    roi_width,
                    roi_height
                ])

        # 9. Debug Visualization
        debug_img = cv_img.copy()

        debug_img = self.detector_module.draw(
            debug_img, corners, ids, 
            camera_matrix=self.camera_matrix, 
            dist_coeffs=self.dist_coeffs, 
            rvec=rvec, 
            tvec=tvec, 
            length=0.1
        )

        debug_img = np.ascontiguousarray(debug_img, dtype=np.uint8)

        try:
            debug_msg = Image()
            debug_msg.header.frame_id = msg.header.frame_id
            debug_msg.header.stamp = msg.header.stamp
            debug_msg.height = debug_img.shape[0]
            debug_msg.width = debug_img.shape[1]
            debug_msg.encoding = "bgr8"
            debug_msg.is_bigendian = 0
            debug_msg.step = debug_img.shape[1] * 3
            debug_msg.data = debug_img.tobytes()

            self.pub_debug_img.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Debug Image Publish Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BaselinePerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
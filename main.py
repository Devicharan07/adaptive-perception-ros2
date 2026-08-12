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
from adaptive_perception.controller import AdaptiveController
from adaptive_perception.roi import ROIManager

# --- EVALUATION LOGGING CONFIGURATION ---
ENABLE_CSV_LOG = True
CSV_LOG_DIR = os.path.expanduser("~/Adaptive_Perception_Project/experiments/logs")


class MainPerceptionNode(Node):
    def __init__(self):
        super().__init__('main_perception_node')

        self.bridge = CvBridge()

        # System Metrics Monitor Initialization
        self.process = psutil.Process(os.getpid())

        # Parameters
        self.declare_parameter('marker_length', 0.15)
        self.marker_length = self.get_parameter('marker_length').value

        # Pipeline Modules
        self.profiler = Profiler()
        self.detector_module = ArUcoDetectorModule(dictionary_name='DICT_ARUCO_ORIGINAL')
        self.confidence_evaluator = ConfidenceEvaluator()
        self.pose_estimator = PoseEstimator()

        self.roi_manager = ROIManager(alpha=2.0, min_padding_px=20)
        self.controller = AdaptiveController(streak_threshold=3, max_displacement=250.0, min_confidence=0.55)

        # State Buffers
        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_corners = None
        self.last_latency_ms = 0.0
        self.frame_counter = 0
        self.last_frame_time = None
        self.last_header_time = None
        self.raw_miss_count = 0
        self.previous_mode = "FULL_IMAGE"

        # ROS Communications
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

        # CSV Logging Initialization (Automatic Timestamp Generator)
        self.csv_path = None
        if ENABLE_CSV_LOG:
            os.makedirs(CSV_LOG_DIR, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file_name = f"experiment_run_{timestamp_str}.csv"
            self.csv_path = os.path.join(CSV_LOG_DIR, csv_file_name)
            
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame_id', 'timestamp', 'detection_success', 'pose_success', 'confidence',
                    'latency_ms', 'latency_derived_fps', 'callback_interval_ms', 'callback_rate_hz', 'header_rate_hz',
                    'mode_used', 'alpha_used', 'scale_used', 'subpix_used',
                    'mode_next', 'alpha_next', 'scale_next', 'subpix_next', 'streak_next',
                    'marker_x', 'marker_y', 'marker_size_px',
                    'raw_miss_start', 'raw_miss_count', 'raw_reacquisition_event', 'raw_frames_to_reacquire',
                    'adaptive_reset_event', 'controller_miss_count',
                    'marker_area_px', 'bbox_width_px', 'bbox_height_px',
                    'roi_x_used', 'roi_y_used', 'roi_width_used', 'roi_height_used',
                    'detector_input_width', 'detector_input_height', 'detector_input_pixels',
                    'decode_convert_ms', 'roi_compute_ms', 'crop_ms', 'resize_ms', 'detector_ms',
                    'subpix_ms', 'confidence_ms', 'controller_ms', 'pose_ms',
                    'cpu_user_ms', 'cpu_system_ms', 'cpu_total_ms', 'process_cpu_util_pct', 'rss_mb',
                    'pipeline_mode'
                ])
            self.get_logger().info(f"Logging experiment to: {self.csv_path}")

        self.get_logger().info("Adaptive Perception Pipeline: Enhanced System Telemetry Active.")

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)

    def image_callback(self, msg: Image):
        current_time = time.perf_counter()
        self.frame_counter += 1

        callback_interval_ms = 0.0
        callback_rate_hz = 0.0
        if self.last_frame_time is not None:
            time_delta = current_time - self.last_frame_time
            if time_delta > 0:
                callback_interval_ms = time_delta * 1000.0
                callback_rate_hz = 1.0 / time_delta
        self.last_frame_time = current_time

        header_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        header_rate_hz = 0.0
        if self.last_header_time is not None:
            header_delta = header_time - self.last_header_time
            if header_delta > 0:
                header_rate_hz = 1.0 / header_delta
        self.last_header_time = header_time

        cpu_times_start = self.process.cpu_times()
        self.profiler.start()

        self.profiler.start_stage('decode_convert')
        try:
            if msg.encoding == 'rgb8':
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
        self.profiler.stop_stage('decode_convert')

        # 1. Stage 1: Search Region Crop & Telemetry Extraction
        current_roi = None
        input_image = cv_img
        img_h, img_w = cv_img.shape[:2]

        mode_used = self.controller.mode
        alpha_used = self.roi_manager.alpha
        scale_used = self.controller.scale_factor
        subpix_used = False
        self.profiler.start_stage('roi_compute')
        if self.controller.mode == "ROI" and self.last_corners is not None:
            current_roi = self.roi_manager.compute_roi(self.last_corners, cv_img.shape)
        self.profiler.stop_stage('roi_compute')
        if current_roi is not None:
            self.profiler.start_stage('crop')
            input_image = self.roi_manager.crop_image(cv_img, current_roi)
            self.profiler.stop_stage('crop')

        if current_roi is not None:
            rx1, ry1, rx2, ry2 = current_roi
            roi_x_used = int(rx1)
            roi_y_used = int(ry1)
            roi_width_used = int(rx2 - rx1)
            roi_height_used = int(ry2 - ry1)
        else:
            roi_x_used = 0
            roi_y_used = 0
            roi_width_used = int(img_w)
            roi_height_used = int(img_h)
            mode_used = "FULL_IMAGE"

        # 2. Stage 2: Adaptive Processing Scale Resampling
        scale = self.controller.scale_factor
        if scale != 1.0:
            self.profiler.start_stage('resize')
            h, w = input_image.shape[:2]
            scaled_w = max(1, int(w * scale))
            scaled_h = max(1, int(h * scale))
            proc_image = cv2.resize(input_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            self.profiler.stop_stage('resize')
        else:
            proc_image = input_image

        # 3. Perception Detection
        detector_input_height, detector_input_width = proc_image.shape[:2]
        detector_input_pixels = detector_input_width * detector_input_height
        self.profiler.start_stage('detector')
        success, corners, ids = self.detector_module.detect(proc_image)
        self.profiler.stop_stage('detector')
        raw_miss_start = 0
        raw_reacquisition_event = 0
        raw_frames_to_reacquire = 0
        if success:
            if self.raw_miss_count > 0:
                raw_reacquisition_event = 1
                raw_frames_to_reacquire = self.raw_miss_count
            self.raw_miss_count = 0
        else:
            raw_miss_start = int(self.raw_miss_count == 0)
            self.raw_miss_count += 1
        if success and corners is not None:
            corners = list(corners)

        # Transform corners back to full crop resolution if scaled
        if success and scale != 1.0:
            corners = [c / scale for c in corners]

        # Transform corners back to full image space if cropped
        if success and current_roi is not None:
            corners = self.roi_manager.transform_corners_to_full_image(corners, current_roi)

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

        # 4. Stage 2: Adaptive Sub-Pixel Corner Refinement
        if success and self.controller.use_subpix and primary_corners is not None:
            subpix_used = True
            self.profiler.start_stage('subpix')
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            pts = np.squeeze(primary_corners).astype(np.float32)
            refined_pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
            primary_corners = refined_pts.reshape(1, 4, 2)
            corners[0] = primary_corners
            self.profiler.stop_stage('subpix')

        # 5. Continuous Confidence Calculation
        self.profiler.start_stage('confidence')
        confidence = self.confidence_evaluator.compute_confidence(corners, cv_img.shape)
        self.profiler.stop_stage('confidence')

        # 6. Two-Stage Controller Decision
        controller_detection_failure = not success or primary_corners is None
        if not controller_detection_failure:
            controller_pts = np.squeeze(primary_corners)
            controller_detection_failure = controller_pts.ndim != 2 or controller_pts.shape[0] < 4
        adaptive_reset_event = int(
            controller_detection_failure and
            self.controller.miss_count + 1 >= self.controller.MISS_TOLERANCE
        )
        self.profiler.start_stage('controller')
        control_state = self.controller.update(
            detection_success=success, 
            corners=primary_corners, 
            confidence=confidence, 
            latency_ms=self.last_latency_ms
        )
        self.profiler.stop_stage('controller')
        self.roi_manager.alpha = control_state["alpha"]

        # Track Mode Switch for telemetry
        roi_switch = 1 if control_state["mode"] != self.previous_mode else 0
        self.previous_mode = control_state["mode"]

        if success and confidence >= self.controller.min_confidence:
            self.last_corners = primary_corners
        else:
            self.last_corners = None

        # 7. Pose Estimation
        pose_success = False
        rvec, tvec, quat = None, None, None
        self.profiler.start_stage('pose')
        if success and self.camera_matrix is not None:
            pose_success, rvec, tvec, quat = self.pose_estimator.estimate(
                corners, self.camera_matrix, self.dist_coeffs, self.marker_length
            )
        self.profiler.stop_stage('pose')

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

        # 8. Profiling & Telemetry
        latency_ms = self.profiler.stop()
        cpu_times_end = self.process.cpu_times()
        cpu_user_ms = (cpu_times_end.user - cpu_times_start.user) * 1000.0
        cpu_system_ms = (cpu_times_end.system - cpu_times_start.system) * 1000.0
        cpu_total_ms = cpu_user_ms + cpu_system_ms
        process_cpu_util_pct = (100.0 * cpu_total_ms / latency_ms) if latency_ms > 0 else 0.0
        latency_derived_fps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0
        self.last_latency_ms = latency_ms

        lat_msg = Float32(data=latency_ms)
        self.pub_latency.publish(lat_msg)

        status_str = (f"Mode: {control_state['mode']} | Alpha: {control_state['alpha']:.2f} | "
                      f"Scale: {control_state['scale']}x | Subpix: {control_state['subpix']} | "
                      f"Streak: {control_state['streak']} | Latency: {latency_ms:.2f}ms")
        self.pub_status.publish(String(data=status_str))

        # CSV Logging Step
        if ENABLE_CSV_LOG and self.csv_path:
            miss_count = getattr(self.controller, 'miss_count', 0)
            rss_mb = self.process.memory_info().rss / (1024.0 * 1024.0)

            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.frame_counter,
                    f"{ts:.4f}",
                    int(success),
                    int(pose_success),
                    f"{confidence:.3f}",
                    f"{latency_ms:.2f}",
                    f"{latency_derived_fps:.2f}",
                    f"{callback_interval_ms:.2f}",
                    f"{callback_rate_hz:.2f}",
                    f"{header_rate_hz:.2f}",
                    mode_used,
                    f"{alpha_used:.2f}",
                    scale_used,
                    int(subpix_used),
                    control_state['mode'],
                    f"{control_state['alpha']:.2f}",
                    control_state['scale'],
                    int(control_state['subpix']),
                    control_state['streak'],
                    f"{marker_x:.1f}",
                    f"{marker_y:.1f}",
                    f"{marker_size_px:.1f}",
                    raw_miss_start,
                    self.raw_miss_count,
                    raw_reacquisition_event,
                    raw_frames_to_reacquire,
                    adaptive_reset_event,
                    miss_count,
                    f"{marker_area_px:.1f}",
                    f"{bbox_width_px:.1f}",
                    f"{bbox_height_px:.1f}",
                    roi_x_used,
                    roi_y_used,
                    roi_width_used,
                    roi_height_used,
                    detector_input_width,
                    detector_input_height,
                    detector_input_pixels,
                    f"{self.profiler.stage_ms('decode_convert'):.2f}",
                    f"{self.profiler.stage_ms('roi_compute'):.2f}",
                    f"{self.profiler.stage_ms('crop'):.2f}",
                    f"{self.profiler.stage_ms('resize'):.2f}",
                    f"{self.profiler.stage_ms('detector'):.2f}",
                    f"{self.profiler.stage_ms('subpix'):.2f}",
                    f"{self.profiler.stage_ms('confidence'):.2f}",
                    f"{self.profiler.stage_ms('controller'):.2f}",
                    f"{self.profiler.stage_ms('pose'):.2f}",
                    f"{cpu_user_ms:.2f}",
                    f"{cpu_system_ms:.2f}",
                    f"{cpu_total_ms:.2f}",
                    f"{process_cpu_util_pct:.2f}",
                    f"{rss_mb:.2f}",
                    "ADAPTIVE"
                ])

        # 9. Debug Visualization
        debug_img = cv_img.copy()
        
        if current_roi is not None:
            x1, y1, x2, y2 = current_roi
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"ROI ({control_state['scale']}x, subpix={control_state['subpix']})", 
                        (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    node = MainPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

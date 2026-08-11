#!/usr/bin/env python3
"""
Confidence Evaluator Module
Computes a lightweight, continuous visual confidence metric [0.0, 1.0]
for ArUco marker detections in UAV perception pipelines.
"""

import math
import numpy as np


class ConfidenceEvaluator:
    def __init__(self, ref_area: float = 10000.0):
        # Nominal marker area in pixels corresponding to 1.0 scale score (100x100 px)
        self.ref_area = ref_area
        
        # Adjusted Component Weights (Sum = 1.0)
        self.w_scale = 0.25
        self.w_center = 0.35
        self.w_geom = 0.40

    def compute_confidence(self, corners: list, img_shape: tuple = (480, 640, 3)) -> float:
        """
        Computes continuous confidence score C in [0.0, 1.0] from current frame geometry.
        Applies square-root scaling to area to prevent quadratic distance penalties.
        """
        if corners is None or len(corners) == 0:
            return 0.0

        pts = np.squeeze(corners[0])
        if pts.ndim != 2 or pts.shape[0] < 4:
            return 0.0

        h, w = img_shape[:2]
        img_center_x, img_center_y = w / 2.0, h / 2.0
        max_dist = math.sqrt(img_center_x**2 + img_center_y**2)

        # 1. Scale / Length Score (Square-root transform converts area ratio to linear pixel length ratio)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        bbox_width = max(1.0, x_max - x_min)
        bbox_height = max(1.0, y_max - y_min)
        bbox_area = bbox_width * bbox_height
        
        # Softened scale curve: sqrt(area / 10000) == (side_px / 100)
        s_scale = min(1.0, math.sqrt(bbox_area / self.ref_area))

        # 2. Image Center Proximity Score
        marker_center = np.mean(pts, axis=0)
        dist_to_center = math.sqrt((marker_center[0] - img_center_x)**2 + (marker_center[1] - img_center_y)**2)
        s_center = max(0.0, 1.0 - (dist_to_center / max_dist))

        # 3. Geometric Quadrilateral Aspect Score
        x = pts[:, 0]
        y = pts[:, 1]
        quad_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        fill_ratio = quad_area / bbox_area if bbox_area > 0 else 0.0
        s_geom = min(1.0, fill_ratio / 0.8)

        # Weighted Sum
        confidence = (self.w_scale * s_scale) + (self.w_center * s_center) + (self.w_geom * s_geom)
        
        return float(np.clip(confidence, 0.0, 1.0))

    def compute_placeholder_confidence(self, corners: list, img_shape: tuple = (480, 640, 3)) -> float:
        return self.compute_confidence(corners, img_shape)
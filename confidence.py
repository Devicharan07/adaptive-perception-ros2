#!/usr/bin/env python3
"""
Confidence Evaluator Module — P2
Computes a geometric confidence score c_geom in [0.0, 1.0].
Two active components (s_geom removed — saturated at 1.0 on all detected frames):
  c_geom = 0.417 * s_scale + 0.583 * s_center
Used only for subpixel refinement gating. Scale/mode decisions use EWMA in the controller.
"""

import math
import numpy as np


class ConfidenceEvaluator:
    def __init__(self, ref_area: float = 10000.0):
        # Nominal marker area in pixels corresponding to 1.0 scale score (100x100 px)
        self.ref_area = ref_area

        # Two-component weights (sum = 1.0). s_geom removed — empirically saturated.
        self.w_scale  = 0.417
        self.w_center = 0.583

    def compute_confidence(self, corners: list, img_shape: tuple = (480, 640, 3)) -> float:
        """
        Computes geometric confidence score c_geom in [0.0, 1.0].
        c_geom = 0.417 * s_scale + 0.583 * s_center
        s_geom removed: empirically saturated at 1.0 on every detected frame.
        """
        if corners is None or len(corners) == 0:
            return 0.0

        pts = np.squeeze(corners[0])
        if pts.ndim != 2 or pts.shape[0] < 4:
            return 0.0

        h, w = img_shape[:2]
        img_center_x, img_center_y = w / 2.0, h / 2.0
        max_dist = math.sqrt(img_center_x**2 + img_center_y**2)

        # 1. Scale score: sqrt(area/ref_area) — linear in pixel side length
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        bbox_width  = max(1.0, x_max - x_min)
        bbox_height = max(1.0, y_max - y_min)
        bbox_area   = bbox_width * bbox_height
        s_scale     = min(1.0, math.sqrt(bbox_area / self.ref_area))

        # 2. Image-center proximity score
        marker_center  = np.mean(pts, axis=0)
        dist_to_center = math.sqrt((marker_center[0] - img_center_x)**2
                                   + (marker_center[1] - img_center_y)**2)
        s_center = max(0.0, 1.0 - (dist_to_center / max_dist))

        # Weighted geometric confidence (no s_geom)
        c_geom = (self.w_scale * s_scale) + (self.w_center * s_center)

        return float(np.clip(c_geom, 0.0, 1.0))

    def compute_placeholder_confidence(self, corners: list, img_shape: tuple = (480, 640, 3)) -> float:
        return self.compute_confidence(corners, img_shape)
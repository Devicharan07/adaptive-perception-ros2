#!/usr/bin/env python3
"""
Controller Module
Two-Stage Hierarchical Adaptive Controller for UAV Perception
Stage 1: Region Selection (FULL_IMAGE vs ROI) & Dynamic Alpha
Stage 2: Resource Allocation (Processing Scale & Corner Refinement)
FSM Stabilization Layer: Miss Tolerance Hysteresis & Low-Confidence Holding
"""

import math
import numpy as np


class AdaptiveController:
    # --- STAGE 2 CONFIGURATION CONSTANTS ---
    HIGH_CONFIDENCE = 0.85
    TRACKING_CONFIDENCE = 0.70
    SUBPIX_CONFIDENCE = 0.75

    # Streak Thresholds
    DEEP_TRACKING_STREAK = 10

    # FSM Hysteresis Safeguard
    MISS_TOLERANCE = 3  # Allows up to 3 consecutive real detection drops before resetting state

    # Latency Safety Safeguard (ms)
    MAX_LATENCY_SAFEGUARD_MS = 35.0

    def __init__(self, streak_threshold: int = 3, max_displacement: float = 250.0, min_confidence: float = 0.55):
        # Stage 1 Parameters
        self.streak_threshold = streak_threshold
        self.max_displacement = max_displacement
        self.min_confidence = min_confidence
        
        # Tracking State Buffers
        self.streak = 0
        self.miss_count = 0
        self.mode = "FULL_IMAGE"  # "FULL_IMAGE", "ROI"
        self.last_center = None
        self.last_alpha = 2.0
        
        # Stage 1: Dynamic Alpha Constants
        self.alpha_base = 2.0
        self.alpha_min = 1.6
        self.alpha_max = 4.0
        self.k1 = 1.2
        self.k2 = 0.4
        self.k3 = 0.8
        self.s_max = 20.0
        self.ref_marker_size = 100.0  # Nominal pixel dimension

        # Stage 2 Outputs
        self.scale_factor = 1.0       # [1.0, 0.5, 0.25]
        self.use_subpix = False       # [False (NONE), True (SUBPIX)]

    def update(self, detection_success: bool, corners: np.ndarray = None, confidence: float = 1.0, latency_ms: float = 0.0) -> dict:
        """
        Executes Stage 1 & Stage 2 control evaluation.
        Inputs: detection_success, corners, confidence, latency_ms.
        Outputs: Dict containing mode, alpha, scale, subpix, and streak.
        """
        # ==========================================
        # CASE 1: REAL DETECTION FAILURE
        # ==========================================
        if not detection_success or corners is None:
            self.miss_count += 1
            if self.miss_count >= self.MISS_TOLERANCE:
                self.miss_count = 0
                self.streak = 0
                self.last_center = None
                self.mode = "FULL_IMAGE"
                self.scale_factor = 1.0
                self.use_subpix = False
                self.last_alpha = self.alpha_base
                return self._get_control_dict(self.alpha_base)
            else:
                return self._get_control_dict(self.last_alpha)

        pts = np.squeeze(corners)
        if pts.ndim != 2 or pts.shape[0] < 4:
            self.miss_count += 1
            if self.miss_count >= self.MISS_TOLERANCE:
                self.miss_count = 0
                self.streak = 0
                self.last_center = None
                self.mode = "FULL_IMAGE"
                self.scale_factor = 1.0
                self.use_subpix = False
                self.last_alpha = self.alpha_base
                return self._get_control_dict(self.alpha_base)
            else:
                return self._get_control_dict(self.last_alpha)

        # ==========================================
        # CASE 2: LOW CONFIDENCE OBSERVATION
        # ==========================================
        if confidence < self.min_confidence:
            # Update geometric reference to keep delta_s calculation accurate
            self.last_center = np.mean(pts, axis=0)
            
            # Do NOT increment miss_count, do NOT reset streak/mode/alpha/scale.
            # Hold previous valid control state and wait for next frame.
            return self._get_control_dict(self.last_alpha)

        # ==========================================
        # CASE 3: GOOD DETECTION
        # ==========================================
        # Reset miss counter on valid, high-confidence detection
        self.miss_count = 0

        current_center = np.mean(pts, axis=0)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        marker_pixel_size = max(1.0, max(x_max - x_min, y_max - y_min))

        delta_s = 0.0
        if self.last_center is not None:
            delta_s = math.sqrt((current_center[0] - self.last_center[0]) ** 2 + 
                                (current_center[1] - self.last_center[1]) ** 2)

        self.last_center = current_center
        effective_max_disp = max(100.0, marker_pixel_size * 2.5)

        if delta_s > effective_max_disp:
            self.miss_count = 0
            self.streak = 0
            self.last_center = None
            self.mode = "FULL_IMAGE"
            self.scale_factor = 1.0
            self.use_subpix = False
            self.last_alpha = self.alpha_base
            return self._get_control_dict(self.alpha_base)

        self.streak += 1

        # ==========================================
        # STAGE 1: Region Selection & Dynamic Alpha
        # ==========================================
        if self.streak >= self.streak_threshold:
            self.mode = "ROI"
        else:
            self.mode = "FULL_IMAGE"

        scale_boost = max(0.0, (self.ref_marker_size - marker_pixel_size) / self.ref_marker_size)
        norm_delta_s = min(1.0, delta_s / effective_max_disp)
        norm_streak = min(1.0, float(self.streak) / self.s_max)
        
        raw_alpha = (self.alpha_base 
                     + self.k1 * norm_delta_s 
                     - self.k2 * norm_streak 
                     + self.k3 * (1.0 - confidence)
                     + 1.0 * scale_boost)
        
        dynamic_alpha = max(self.alpha_min, min(self.alpha_max, raw_alpha))
        self.last_alpha = dynamic_alpha

        # ==========================================
        # STAGE 2: Processing Scale Selection
        # ==========================================
        if self.mode == "FULL_IMAGE":
            self.scale_factor = 1.0
        elif self.streak >= self.DEEP_TRACKING_STREAK and confidence >= self.HIGH_CONFIDENCE:
            self.scale_factor = 0.25
        elif self.streak >= self.streak_threshold and confidence >= self.TRACKING_CONFIDENCE:
            self.scale_factor = 0.5
        else:
            self.scale_factor = 1.0

        if latency_ms > self.MAX_LATENCY_SAFEGUARD_MS and self.scale_factor == 0.25:
            self.scale_factor = 0.5

        # ==========================================
        # STAGE 2: Corner Refinement Selection
        # ==========================================
        if confidence < self.SUBPIX_CONFIDENCE:
            self.use_subpix = True
        else:
            self.use_subpix = False

        return self._get_control_dict(dynamic_alpha)

    def _get_control_dict(self, alpha: float) -> dict:
        return {
            "mode": self.mode,
            "alpha": alpha,
            "scale": self.scale_factor,
            "subpix": self.use_subpix,
            "streak": self.streak
        }
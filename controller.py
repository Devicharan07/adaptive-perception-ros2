#!/usr/bin/env python3
"""
Controller Module — P2
Explicit two-state FSM: RECOVERING (FULL_IMAGE @ 1.0x) | LOCKED (ROI @ {1.0x, 0.6x, 0.25x})

Key design decisions vs P1:
  - EWMA replaces geometric confidence for scale/mode decisions (r=+0.22 vs r=-0.024)
  - Joint mode+scale state: FULL_IMAGE + reduced scale is structurally impossible
  - Immediate backoff on any miss within LOCKED; dwell required to downscale
  - last_valid_marker_size gates 0.25x (empirical 120px resolution-adequacy cliff)
  - Alpha simplified: 2.0 + 0.5*(1-ewma) — same effective range, honest signal
"""

import math
import numpy as np


class AdaptiveController:
    # ------------------------------------------------------------------
    # Recalibrated thresholds (pooled 19,350-frame dataset, 2-component
    # confidence formula — same behavioral coverage as P1 thresholds)
    # ------------------------------------------------------------------
    HIGH_CONFIDENCE     = 0.75   # ~24% of detected frames above (was 0.85)
    TRACKING_CONFIDENCE = 0.58   # ~98% of detected frames above (was 0.70)
    SUBPIX_CONFIDENCE   = 0.62   # ~87% of detected frames above (was 0.75)
    MIN_CONFIDENCE      = 0.45   # ~100% coverage (was 0.55)

    # P2 FSM parameters
    LOCK_STREAK     = 3      # consecutive successes to enter LOCKED
    MISS_TOLERANCE  = 3      # consecutive misses to exit LOCKED -> RECOVERING
    DWELL_FRAMES    = 5      # frames to hold a scale before downscaling

    # Scale gating (data-derived)
    EWMA_GATE_025      = 0.80   # ewma required to enter 0.25x
    EWMA_GATE_06       = 0.75   # ewma required to hold 0.6x; below -> step up to 1.0x
    MIN_MARKER_FOR_025 = 120.0  # px — empirical resolution-adequacy cliff

    # Latency safeguard
    MAX_LATENCY_SAFEGUARD_MS = 35.0

    def __init__(self, streak_threshold: int = 3, max_displacement: float = 250.0,
                 min_confidence: float = 0.45,
                 enable_dwell: bool = True,
                 enable_asymmetric_miss: bool = True,
                 enable_marker_size_gate: bool = True,
                 enable_ewma: bool = True):
        # Kept for API compatibility with main.py constructor call
        self.streak_threshold = streak_threshold
        self.max_displacement = max_displacement
        self.min_confidence   = self.MIN_CONFIDENCE

        # Ablation control flags
        self.enable_dwell = enable_dwell
        self.enable_asymmetric_miss = enable_asymmetric_miss
        self.enable_marker_size_gate = enable_marker_size_gate
        self.enable_ewma = enable_ewma

        # EWMA temporal reliability signal
        self.ewma       = 0.5    # initialise at neutral
        self.ewma_alpha = 0.15   # 15% update per frame (~6-frame time constant)

        # P2 FSM state
        self.fsm_state  = "RECOVERING"   # "RECOVERING" | "LOCKED"
        self.mode       = "FULL_IMAGE"   # mirrors fsm_state for telemetry compatibility

        # Scale within LOCKED
        self.scale_factor  = 1.0
        self.dwell_counter = 0

        # Tracking state
        self.streak      = 0
        self.miss_count  = 0
        self.last_center = None
        self.last_alpha  = 2.0
        self.last_valid_marker_size = 0.0

        # Subpixel refinement
        self.use_subpix = False

    # ------------------------------------------------------------------
    def update(self, detection_success: bool, corners: np.ndarray = None,
               confidence: float = 1.0, latency_ms: float = 0.0,
               marker_size_px: float = -1.0) -> dict:
        """
        Executes P2 FSM control evaluation.
        Inputs:  detection_success, corners, confidence (c_geom), latency_ms, marker_size_px
        Outputs: dict with mode, alpha, scale, subpix, streak
        """
        # ------------------------------------------------------------------
        # 1. EWMA update is deferred until after displacement validation
        #    (step 3b below) so it reflects tracking validity, not raw
        #    detector output. A displaced/invalid detection must not credit
        #    EWMA as a success.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # 2. Corner geometry validation
        # ------------------------------------------------------------------
        valid_corners     = False
        current_center    = None
        marker_pixel_size = 0.0

        if detection_success and corners is not None:
            pts = np.squeeze(corners)
            if pts.ndim == 2 and pts.shape[0] >= 4:
                valid_corners     = True
                current_center    = np.mean(pts, axis=0)
                x_min, y_min      = np.min(pts, axis=0)
                x_max, y_max      = np.max(pts, axis=0)
                marker_pixel_size = max(1.0, max(x_max - x_min, y_max - y_min))

        detection_valid = detection_success and valid_corners

        # ------------------------------------------------------------------
        # 3. Displacement guard — gross jump treated as miss (LOCKED mode only)
        #    In RECOVERING (FULL_IMAGE search), the target can legitimately
        #    reappear anywhere across the full 640x480 frame.
        # ------------------------------------------------------------------
        if self.fsm_state == "LOCKED":
            if detection_valid and self.last_center is not None:
                delta         = math.sqrt(float(np.sum((current_center - self.last_center) ** 2)))
                effective_max = max(100.0, marker_pixel_size * 2.5)
                if delta > effective_max:
                    detection_valid = False

        # ------------------------------------------------------------------
        # 3b. EWMA update — after full validation (corner + displacement)
        # ------------------------------------------------------------------
        self.ewma = (self.ewma * (1.0 - self.ewma_alpha)
                     + float(detection_valid) * self.ewma_alpha)

        # ------------------------------------------------------------------
        # 4. Update tracking state
        # ------------------------------------------------------------------
        if detection_valid:
            self.miss_count  = 0
            self.streak     += 1
            self.last_center = current_center
            if marker_size_px > 0:
                self.last_valid_marker_size = marker_size_px
        else:
            self.miss_count += 1
            self.streak      = 0
            # If in RECOVERING, clear stale center so it does not ghost-pollute search
            if self.fsm_state == "RECOVERING":
                self.last_center = None

        # ------------------------------------------------------------------
        # 5. FSM state transitions
        # ------------------------------------------------------------------
        if self.fsm_state == "RECOVERING":
            if self.streak >= self.LOCK_STREAK:
                self.fsm_state     = "LOCKED"
                self.scale_factor  = 1.0   # enter LOCKED conservatively at 1.0x
                self.dwell_counter = 0

        elif self.fsm_state == "LOCKED":
            if self.miss_count >= self.MISS_TOLERANCE:
                self.fsm_state     = "RECOVERING"
                self.miss_count    = 0
                self.streak        = 0
                self.scale_factor  = 1.0
                self.dwell_counter = 0
                self.last_center   = None

        # ------------------------------------------------------------------
        # 6. Mode assignment — structural: FULL_IMAGE only in RECOVERING
        # ------------------------------------------------------------------
        if self.fsm_state == "RECOVERING":
            self.mode         = "FULL_IMAGE"
            self.scale_factor = 1.0
        else:
            self.mode = "ROI"

        # ------------------------------------------------------------------
        # 7. Scale selection within LOCKED
        # ------------------------------------------------------------------
        if self.fsm_state == "LOCKED":
            self._update_scale_locked(detection_valid, confidence, latency_ms)

        # ------------------------------------------------------------------
        # 8. Subpixel refinement gate (geometric quality of current frame)
        # ------------------------------------------------------------------
        self.use_subpix = confidence < self.SUBPIX_CONFIDENCE

        # ------------------------------------------------------------------
        # 9. Alpha — simplified, ewma-driven. Range [2.0, 2.5].
        # ------------------------------------------------------------------
        dynamic_alpha   = 2.0 + 0.5 * (1.0 - self.ewma)
        self.last_alpha = dynamic_alpha

        return self._get_control_dict(dynamic_alpha)

    # ------------------------------------------------------------------
    def _update_scale_locked(self, detection_valid: bool,
                             confidence: float, latency_ms: float) -> None:
        """
        Scale ladder within LOCKED (ROI) mode.
        Ladder conservative -> aggressive: 1.0x -> 0.6x -> 0.25x
        Upscale: immediate on any miss.
        Downscale: requires dwell >= DWELL_FRAMES + ewma/marker-size gates.
        """
        self.dwell_counter += 1

        # --- BACKOFF on detection failure ---
        if not detection_valid:
            if self.scale_factor == 0.25:
                # 0.25x operates only on large markers (>=120px).
                # If asymmetric miss is enabled, tolerates 1 isolated miss and steps up on 2.
                # If disabled (ablation), steps up immediately on 1 miss.
                miss_limit = 2 if self.enable_asymmetric_miss else 1
                if self.miss_count >= miss_limit:
                    self.scale_factor = 0.6
                    self.dwell_counter = 0
            elif self.scale_factor == 0.6:
                # 0.6x operates near the decode cliff (45-70px). Immediate backoff to 1.0x.
                self.scale_factor = 1.0
                self.dwell_counter = 0
            # already at 1.0x: stay — MISS_TOLERANCE (3) will trigger RECOVERING
            return

        # --- LATENCY SAFEGUARD ---
        if latency_ms > self.MAX_LATENCY_SAFEGUARD_MS and self.scale_factor == 0.25:
            self.scale_factor  = 0.6
            self.dwell_counter = 0
            return

        # --- EWMA & RESOLUTION GATES ---
        ewma_val   = self.ewma if self.enable_ewma else 1.0
        min_marker = self.MIN_MARKER_FOR_025 if self.enable_marker_size_gate else 0.0

        # --- EWMA-DRIVEN UPSCALE (no dwell required) ---
        if self.scale_factor == 0.25 and (
            ewma_val < self.EWMA_GATE_025
            or self.last_valid_marker_size < min_marker
        ):
            self.scale_factor  = 0.6
            self.dwell_counter = 0

        elif self.scale_factor == 0.6 and ewma_val < self.EWMA_GATE_06:
            self.scale_factor  = 1.0
            self.dwell_counter = 0

        # --- DWELL-GATED DOWNSCALE ---
        dwell_limit = self.DWELL_FRAMES if self.enable_dwell else 0
        if self.dwell_counter >= dwell_limit:
            if self.scale_factor == 1.0 and ewma_val >= self.EWMA_GATE_06:
                self.scale_factor  = 0.6
                self.dwell_counter = 0

            elif (self.scale_factor == 0.6
                  and ewma_val >= self.EWMA_GATE_025
                  and self.last_valid_marker_size >= min_marker):
                self.scale_factor  = 0.25
                self.dwell_counter = 0

    # ------------------------------------------------------------------
    def _get_control_dict(self, alpha: float) -> dict:
        return {
            "mode":   self.mode,
            "alpha":  alpha,
            "scale":  self.scale_factor,
            "subpix": self.use_subpix,
            "streak": self.streak,
        }

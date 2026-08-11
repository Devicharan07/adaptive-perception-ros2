#!/usr/bin/env python3
"""
ROI Module - Adaptive Search Region Management
Handles cropping regions of interest and translating local coordinates back to full image space.
"""

import numpy as np


class ROIManager:
    def __init__(self, alpha: float = 2.0, min_padding_px: int = 20):
        self.alpha = alpha  # Expansion factor
        self.min_padding_px = min_padding_px  # Minimum absolute pixel margin around marker
        self.current_roi = None  # Tuple: (x1, y1, x2, y2)

    def compute_roi(self, corners: np.ndarray, img_shape: tuple) -> tuple:
        """
        Computes bounding ROI based on detected ArUco corners, expansion factor alpha,
        and absolute minimum pixel padding to preserve the ArUco quiet zone.
        """
        if corners is None:
            return None

        # Robust array dimension handling
        pts = np.squeeze(corners)
        if pts.ndim != 2 or pts.shape[0] < 4:
            return None

        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Scale box by alpha, but ensure an absolute minimum pixel padding (quiet zone)
        roi_w = max(width * self.alpha, width + 2 * self.min_padding_px)
        roi_h = max(height * self.alpha, height + 2 * self.min_padding_px)

        img_h, img_w = img_shape[:2]

        x1 = int(max(0, center_x - roi_w / 2.0))
        y1 = int(max(0, center_y - roi_h / 2.0))
        x2 = int(min(img_w, center_x + roi_w / 2.0))
        y2 = int(min(img_h, center_y + roi_h / 2.0))

        # Ensure valid sub-image dimensions
        if (x2 - x1) < 15 or (y2 - y1) < 15:
            return None

        self.current_roi = (x1, y1, x2, y2)
        return self.current_roi

    def crop_image(self, image: np.ndarray, roi: tuple) -> np.ndarray:
        """Crops image given (x1, y1, x2, y2)."""
        if roi is None:
            return image
        x1, y1, x2, y2 = roi
        return image[y1:y2, x1:x2]

    def transform_corners_to_full_image(self, corners: list, roi: tuple) -> list:
        """
        Translates detected corners in cropped ROI frame back to full image coordinate space.
        """
        if corners is None or len(corners) == 0 or roi is None:
            return corners

        x1, y1, _, _ = roi
        transformed_corners = []
        for corner in corners:
            c_copy = corner.copy()
            c_copy[..., 0] += x1  # Add x offset
            c_copy[..., 1] += y1  # Add y offset
            transformed_corners.append(c_copy)

        return transformed_corners
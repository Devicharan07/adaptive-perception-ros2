#!/usr/bin/env python3

import cv2
import numpy as np

class ArUcoDetectorModule:
    def __init__(self, dictionary_name='DICT_ARUCO_ORIGINAL'):
        dict_type = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()

        # Restored conservative parameters to prevent texture false positives
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.maxErroneousBitsInBorderRate = 0.35

        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            self.use_legacy = False
        else:
            self.use_legacy = True

    def detect(self, image: np.ndarray):
        """Detects ArUco markers in the given image."""
        if image is None or image.size == 0:
            return False, [], None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if self.use_legacy:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        else:
            corners, ids, _ = self.detector.detectMarkers(gray)

        success = (ids is not None and len(ids) > 0)
        return success, corners, ids

    def draw(self, image: np.ndarray, corners, ids, camera_matrix=None, dist_coeffs=None, rvec=None, tvec=None, length=0.1):
        """Draws detected markers and 3D coordinate axes."""
        annotated_img = image.copy()
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated_img, corners, ids)
            
            if rvec is not None and tvec is not None and camera_matrix is not None and dist_coeffs is not None:
                try:
                    cv2.drawFrameAxes(
                        annotated_img, camera_matrix, dist_coeffs, rvec, tvec, length
                    )
                except Exception:
                    pass

        annotated_img = np.ascontiguousarray(annotated_img, dtype=np.uint8)
        return annotated_img
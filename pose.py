import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseEstimator:
    def __init__(self):
        pass

    def estimate(self, corners, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, marker_length: float):
        """
        Solves Perspective-n-Point (PnP) for detected 2D marker corners.
        Uses SOLVEPNP_IPPE_SQUARE with SOLVEPNP_ITERATIVE fallback.
        
        Returns:
            success (bool), rvec (3x1 array), tvec (3x1 array), quat (4x1 array)
        """
        if corners is None or len(corners) == 0 or camera_matrix is None:
            return False, None, None, None

        # Define 3D Object Points in Marker Local Frame
        half_l = marker_length / 2.0
        obj_points = np.array([
            [-half_l,  half_l, 0.0],
            [ half_l,  half_l, 0.0],
            [ half_l, -half_l, 0.0],
            [-half_l, -half_l, 0.0]
        ], dtype=np.float32)

        # First detected marker corners (TODO: expand to multi-marker tracking)
        img_points = corners[0][0].astype(np.float32)

        # 1. Try Primary Solver: SOLVEPNP_IPPE_SQUARE
        success, rvec, tvec = cv2.solvePnP(
            obj_points, 
            img_points, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        # 2. Fallback Solver: SOLVEPNP_ITERATIVE
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                obj_points, 
                img_points, 
                camera_matrix, 
                dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            return False, None, None, None

        # Convert rvec to Quaternion [x, y, z, w]
        rmat, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rmat).as_quat()

        return success, rvec, tvec, quat

if __name__ == '__main__':
    # Standalone Unit Test
    estimator = PoseEstimator()
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    D = np.zeros((5, 1), dtype=np.float32)
    dummy_corners = [np.array([[[270, 190], [370, 190], [370, 290], [270, 290]]], dtype=np.float32)]
    
    success, rvec, tvec, quat = estimator.estimate(dummy_corners, K, D, 0.15)
    print(f"[Pose Test] Success: {success}")
    if success:
        print(f"[Pose Test] Translation (x,y,z meters):\n{tvec.ravel()}")
        print(f"[Pose Test] Quaternion (x,y,z,w):\n{quat}")
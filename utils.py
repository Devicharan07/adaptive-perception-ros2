import numpy as np
from sensor_msgs.msg import Image

def format_status_string(detected: bool, confidence: float, latency_ms: float) -> str:
    """Formats perception pipeline telemetry into a single status string."""
    return f"DETECTED:{detected}|CONF:{confidence:.2f}|LATENCY:{latency_ms:.2f}ms"

def cv2_to_imgmsg_manual(cv_img: np.ndarray, frame_id: str = "camera_link", stamp=None) -> Image:
    """
    Directly converts a 3-channel BGR OpenCV uint8 image into a ROS 2 Image message,
    bypassing cv_bridge C++ binding constraints.
    """
    img_msg = Image()
    
    # 1. Set Header Info
    if stamp is not None:
        img_msg.header.stamp = stamp
    img_msg.header.frame_id = frame_id
    
    # 2. Dimensions & Channel Stride
    height, width, channels = cv_img.shape
    img_msg.height = int(height)
    img_msg.width = int(width)
    img_msg.encoding = 'bgr8'
    img_msg.is_bigendian = 0
    img_msg.step = int(width * channels)  # 3 bytes per pixel
    
    # 3. Direct Flattened Byte Buffer
    img_msg.data = cv_img.astype(np.uint8).tobytes()
    
    return img_msg
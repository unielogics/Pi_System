"""
The only Aurora-930-specific file in this service. Wraps the vendor's ROS2 driver
(deptrum-ros-driver-aurora930) topics behind the camera-agnostic CameraAdapter interface.

Topic names/message shapes per the vendor's ROS2 usage docs:
  /aurora/rgb/image_raw     sensor_msgs/Image (bgr8)
  /aurora/depth/image_raw   sensor_msgs/Image (16-bit depth, millimeters)
  /aurora/ir/camera_info    sensor_msgs/CameraInfo (depth/IR intrinsics -- the depth stream is
                             registered to the IR sensor, not the RGB one, per the vendor driver)
  /aurora/rgb/camera_info   sensor_msgs/CameraInfo (RGB camera's OWN intrinsics -- confirmed via
                             live topic inspection that these are DIFFERENT numbers from the IR
                             camera's: depth and RGB are two physically separate lenses on the
                             same module, not one registered sensor)
  /tf_static                tf2_msgs/TFMessage, latched (TRANSIENT_LOCAL) -- contains the rigid
                             depth_camera_link -> rgb_camera_link transform (~10mm baseline,
                             confirmed live), needed to move a 3D point measured in depth-camera
                             space into RGB-camera space before projecting it onto the RGB image.

A future camera gets its own adapter file implementing the same CameraAdapter methods; nothing in
measure.py or api.py changes.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

from measure import Intrinsics

# /tf_static is published once per static transform, retained for any subscriber that joins later
# (confirmed live via `ros2 topic info --verbose`: Durability=TRANSIENT_LOCAL) -- matching QoS is
# required on the subscriber side or rclpy will never deliver the already-published message.
_TF_STATIC_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _quat_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class AuroraAdapter(Node):
    def __init__(self) -> None:
        super().__init__("dimensioner_aurora_adapter")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._depth_frame: Optional[np.ndarray] = None
        self._depth_frame_at: Optional[float] = None
        self._rgb_frame: Optional[np.ndarray] = None
        self._rgb_frame_at: Optional[float] = None
        self._intrinsics: Optional[Intrinsics] = None
        self._rgb_intrinsics: Optional[Intrinsics] = None
        # Rotation + translation (mm) of depth_camera_link -> rgb_camera_link, i.e. p_depth = R @
        # p_rgb + t -- the exact convention tf2 static transforms publish (transform.frame_id is
        # the parent, child_frame_id is the child; translation/rotation carry a point FROM child
        # TO parent). depth_to_rgb_mm() below applies the inverse of this to go the other way.
        self._depth_to_rgb_rotation: Optional[np.ndarray] = None
        self._depth_to_rgb_translation_mm: Optional[np.ndarray] = None

        self.create_subscription(Image, "/aurora/depth/image_raw", self._on_depth, 1)
        self.create_subscription(Image, "/aurora/rgb/image_raw", self._on_rgb, 1)
        self.create_subscription(CameraInfo, "/aurora/ir/camera_info", self._on_camera_info, 1)
        self.create_subscription(CameraInfo, "/aurora/rgb/camera_info", self._on_rgb_camera_info, 1)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf_static, _TF_STATIC_QOS)

    def _on_depth(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self._lock:
            self._depth_frame = frame.astype(np.uint16)
            self._depth_frame_at = time.monotonic()

    def _on_rgb(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self._lock:
            self._rgb_frame = frame
            self._rgb_frame_at = time.monotonic()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        k = msg.k  # row-major 3x3 intrinsic matrix
        with self._lock:
            self._intrinsics = Intrinsics(fx=k[0], fy=k[4], cx=k[2], cy=k[5])

    def _on_rgb_camera_info(self, msg: CameraInfo) -> None:
        k = msg.k
        with self._lock:
            self._rgb_intrinsics = Intrinsics(fx=k[0], fy=k[4], cx=k[2], cy=k[5])

    def _on_tf_static(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            if t.header.frame_id == "depth_camera_link" and t.child_frame_id == "rgb_camera_link":
                q = t.transform.rotation
                translation_mm = np.array(
                    [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
                ) * 1000.0
                with self._lock:
                    self._depth_to_rgb_rotation = _quat_to_rotation_matrix(q.x, q.y, q.z, q.w)
                    self._depth_to_rgb_translation_mm = translation_mm
                break

    def get_depth_frame(self) -> np.ndarray:
        with self._lock:
            if self._depth_frame is None:
                raise RuntimeError("No depth frame received yet from the Aurora 930 driver.")
            return self._depth_frame.copy()

    def get_rgb_frame(self) -> np.ndarray:
        with self._lock:
            if self._rgb_frame is None:
                raise RuntimeError("No RGB frame received yet from the Aurora 930 driver.")
            return self._rgb_frame.copy()

    def get_intrinsics(self) -> Intrinsics:
        with self._lock:
            if self._intrinsics is None:
                raise RuntimeError("No camera_info received yet from the Aurora 930 driver.")
            return self._intrinsics

    def get_rgb_intrinsics(self) -> Intrinsics:
        with self._lock:
            if self._rgb_intrinsics is None:
                raise RuntimeError("No RGB camera_info received yet from the Aurora 930 driver.")
            return self._rgb_intrinsics

    def depth_to_rgb_mm(self, point_mm: tuple) -> tuple:
        with self._lock:
            r = self._depth_to_rgb_rotation
            t = self._depth_to_rgb_translation_mm
        if r is None or t is None:
            raise RuntimeError("No depth_camera_link->rgb_camera_link transform received yet from the Aurora 930 driver.")
        p_depth = np.array(point_mm)
        p_rgb = r.T @ (p_depth - t)  # inverse of p_depth = R @ p_rgb + t (see _on_tf_static)
        return (float(p_rgb[0]), float(p_rgb[1]), float(p_rgb[2]))

    def get_frame_age_sec(self) -> Optional[float]:
        """Age of the OLDER of the two streams (worst case) -- None if neither has ever arrived."""
        with self._lock:
            timestamps = [t for t in (self._depth_frame_at, self._rgb_frame_at) if t is not None]
            if not timestamps:
                return None
            return time.monotonic() - min(timestamps)


_adapter: Optional[AuroraAdapter] = None
_spin_thread: Optional[threading.Thread] = None


def get_aurora_adapter() -> AuroraAdapter:
    """Singleton: starts rclpy + the adapter node + a background spin thread on first call."""
    global _adapter, _spin_thread
    if _adapter is None:
        rclpy.init()
        _adapter = AuroraAdapter()
        _spin_thread = threading.Thread(target=rclpy.spin, args=(_adapter,), daemon=True)
        _spin_thread.start()
    return _adapter

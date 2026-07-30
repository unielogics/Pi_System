"""
CameraAdapter interface. This is the ONLY boundary a new camera model needs to implement --
measure.py and api.py never import a vendor SDK directly, only this interface.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from measure import Intrinsics


class CameraAdapter(Protocol):
    def get_depth_frame(self) -> np.ndarray:
        """Returns an HxW uint16/float array, depth in millimeters. 0 = invalid/no return."""
        ...

    def get_rgb_frame(self) -> np.ndarray:
        """Returns an HxWx3 uint8 array (RGB, not BGR)."""
        ...

    def get_intrinsics(self) -> Intrinsics:
        """Returns the depth camera's intrinsics (fx, fy, cx, cy), matching the depth frame's resolution."""
        ...

    def get_rgb_intrinsics(self) -> Intrinsics:
        """Returns the RGB camera's OWN intrinsics -- DIFFERENT from get_intrinsics() (the depth/IR
        camera's) whenever the two are physically separate lenses, which is common on structured-
        light/stereo depth modules including the Aurora 930 (confirmed live: distinct fx/fy/cx/cy
        on /aurora/ir/camera_info vs /aurora/rgb/camera_info, and a real ~10mm baseline between
        depth_camera_link and rgb_camera_link on /tf_static -- these are two lenses on one PCB, not
        one registered sensor). A future adapter for a camera whose depth and RGB streams truly
        share one optical center may just return the same Intrinsics as get_intrinsics().

        Drawing a 3D point (measured in depth-camera space) onto the RGB image REQUIRES first
        converting it into RGB-camera space via depth_to_rgb_mm(), then projecting with THESE
        intrinsics -- projecting a depth-space point directly with the depth intrinsics (the
        original bug this interface fixes) silently offsets the drawn overlay by however far apart
        the two lenses' optical centers are, worse the closer the item is to the camera.
        """
        ...

    def depth_to_rgb_mm(self, point_mm: tuple[float, float, float]) -> tuple[float, float, float]:
        """Converts a 3D point FROM the depth camera's coordinate frame INTO the RGB camera's
        coordinate frame, both in millimeters. Identity for a camera whose depth and RGB streams
        already share one optical center; a real rigid transform (rotation + translation between
        the two physical lenses) otherwise -- see get_rgb_intrinsics()'s docstring for why this
        distinction is real on the Aurora 930, not a hypothetical."""
        ...

    def get_frame_age_sec(self) -> Optional[float]:
        """Seconds since the older of the two streams last updated. None if nothing has arrived yet.

        Required so api.py can detect a hung/disconnected camera instead of silently serving or
        measuring a frozen frame forever.
        """
        ...

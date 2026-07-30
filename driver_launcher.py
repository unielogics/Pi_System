"""
Launches the vendor's Aurora 930 ROS2 driver node.

Built and verified against the real camera on 2026-07-25. Two runtime quirks the vendor's own
install/setup.bash doesn't handle, both worked around here:

1. LD_LIBRARY_PATH must explicitly include the workspace's installed lib dir (for
   libdeptrum_stream_aurora900.so) AND the RoboStack conda env's lib dir (for libcv_bridge.so) --
   neither is added by sourcing install/setup.bash alone; ros2 launch spawns the compiled node
   as a raw subprocess that doesn't inherit the conda env's linker search path otherwise.

2. This Pi's camera is plugged into a USB 2.0 (480Mbps) hub port, not a USB 3.0 port -- at full
   bandwidth (default fps + point cloud enabled) the camera disconnects after ~10s
   (`bulk_transfer failed`). Point cloud is disabled and RGB/IR fps capped at 5 here as a
   bandwidth-budget workaround. If the camera is later moved to a real USB 3.0 port (the Pi4's
   blue ports, not the internal hub), these caps can likely be lifted -- test with
   `ros2 topic hz /aurora/depth/image_raw` over a few minutes before removing them.
"""
import os
import subprocess
import sys

WORKSPACE_SETUP = os.path.expanduser("~/dimensioner_ws/install/setup.bash")
DRIVER_LIB_DIR = os.path.expanduser("~/dimensioner_ws/install/deptrum-ros-driver-aurora930/lib")
CONDA_ENV_LIB_DIR = os.path.expanduser("~/micromamba/envs/ros2/lib")


def main() -> int:
    if not os.path.exists(WORKSPACE_SETUP):
        print(
            "Vendor driver workspace not found at " + WORKSPACE_SETUP + ". "
            "Build the deptrum-ros-driver-aurora930 package first (see dimensioner/README.md).",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    existing_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{DRIVER_LIB_DIR}:{CONDA_ENV_LIB_DIR}:{existing_ld_path}"

    cmd = (
        f"source {WORKSPACE_SETUP} && exec ros2 launch deptrum-ros-driver-aurora930 "
        f"aurora930_launch.py point_cloud_enable:=false ir_fps:=5 rgb_fps:=5"
    )
    return subprocess.call(["bash", "-c", cmd], env=env)


if __name__ == "__main__":
    raise SystemExit(main())

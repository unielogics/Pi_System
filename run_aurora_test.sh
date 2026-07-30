#!/usr/bin/env bash
export MAMBA_ROOT_PREFIX=/home/franco/micromamba
export LD_LIBRARY_PATH=/home/franco/dimensioner_ws/install/deptrum-ros-driver-aurora930/lib:/home/franco/micromamba/envs/ros2/lib:$LD_LIBRARY_PATH
source /home/franco/dimensioner_ws/install/setup.bash
exec ros2 launch deptrum-ros-driver-aurora930 aurora930_launch.py point_cloud_enable:=false ir_fps:=5 rgb_fps:=5

"""Conversions between Unitree DDS messages and spatial transforms."""

import numpy as np
import spatialmath as sm
from scipy.spatial.transform import Rotation
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_


def pose_stamped_to_se2(message: PoseStamped_) -> sm.SE2:
    position = message.pose.position
    orientation = message.pose.orientation
    quaternion = np.asarray([orientation.w, orientation.x, orientation.y, orientation.z], dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-9:
        raise ValueError("received a pose with a zero-length quaternion")
    yaw = Rotation.from_quat(quaternion / norm, scalar_first=True).as_euler("xyz")[2]
    return sm.SE2(position.x, position.y, yaw)


def pose_stamped_to_matrix(message: PoseStamped_) -> np.ndarray:
    position = message.pose.position
    orientation = message.pose.orientation
    quaternion = np.asarray([orientation.w, orientation.x, orientation.y, orientation.z], dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-9:
        raise ValueError("received a pose with a zero-length quaternion")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm, scalar_first=True).as_matrix()
    transform[:3, 3] = [position.x, position.y, position.z]
    return transform


__all__ = ["pose_stamped_to_matrix", "pose_stamped_to_se2"]

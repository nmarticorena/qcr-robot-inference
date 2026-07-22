"""Gear-Sonic VR targets for grasping and following dolly handles."""

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation, Slerp
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from rs_imle_policy.configs.g1.locomotion import DollyVRConfig
from rs_imle_policy.controllers.g1.gear_sonic_interface import (
    DEFAULT_VR_ORIENTATIONS,
    DEFAULT_VR_POSITIONS,
)
from rs_imle_policy.utils.dds import pose_stamped_to_matrix


BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
DEFAULT_HAND_POSE_URDF = Path(__file__).resolve().parents[4] / "assets/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"


class CurrentHandPoseFK:
    def __init__(self, urdf_path: Path = DEFAULT_HAND_POSE_URDF) -> None:
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.joint_q_indices = tuple(self.model.joints[self.model.getJointId(name)].idx_q for name in BODY_JOINT_NAMES)
        self.hand_frame_ids = (
            self.model.getFrameId("left_wrist_yaw_link"),
            self.model.getFrameId("right_wrist_yaw_link"),
        )

    def vr_poses(self, lowstate: LowState_) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if lowstate is None:
            raise TimeoutError("timed out waiting for rt/lowstate")
        if len(lowstate.motor_state) < len(self.joint_q_indices):
            raise ValueError("rt/lowstate does not contain all body joints")
        body_q = np.asarray([lowstate.motor_state[index].q for index in range(len(self.joint_q_indices))])
        if not np.all(np.isfinite(body_q)):
            raise ValueError("rt/lowstate contains a non-finite joint position")
        q = pin.neutral(self.model)
        q[np.asarray(self.joint_q_indices)] = body_q
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        poses = [self.data.oMf[frame_id] for frame_id in self.hand_frame_ids]
        positions = np.concatenate([pose.translation for pose in poses] + [np.asarray(DEFAULT_VR_POSITIONS[6:9])])
        orientations = np.concatenate(
            [Rotation.from_matrix(pose.rotation).as_quat(scalar_first=True) for pose in poses]
            + [np.asarray(DEFAULT_VR_ORIENTATIONS[8:12])]
        )
        return tuple(positions), tuple(orientations)


def waist_relative_vr_targets(
    dolly_pose: PoseStamped_,
    waist_pose: PoseStamped_,
    config: DollyVRConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    waist_from_dolly = np.linalg.inv(pose_stamped_to_matrix(waist_pose)) @ pose_stamped_to_matrix(dolly_pose)
    poses = []
    for handle in (config.left_handle, config.right_handle):
        dolly_from_handle = np.eye(4)
        dolly_from_handle[:3, 3] = handle
        poses.append(waist_from_dolly @ dolly_from_handle)
    positions = np.concatenate([poses[0][:3, 3], poses[1][:3, 3], DEFAULT_VR_POSITIONS[6:9]])
    orientations = np.concatenate(
        [Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True) for pose in poses]
        + [np.asarray(DEFAULT_VR_ORIENTATIONS[8:12])]
    )
    return tuple(positions), tuple(orientations)


def dolly_center_velocity(
    dolly_pose: PoseStamped_,
    waist_pose: PoseStamped_,
    config: DollyVRConfig,
) -> tuple[float, float, float]:
    waist_from_dolly = np.linalg.inv(pose_stamped_to_matrix(waist_pose)) @ pose_stamped_to_matrix(dolly_pose)
    direction = waist_from_dolly[:2, 3]
    distance = np.linalg.norm(direction)
    vx, vy = (0.0, 0.0) if distance <= 1e-9 else config.tracking_speed * direction / distance
    yaw_error = np.arctan2(waist_from_dolly[1, 0], waist_from_dolly[0, 0])
    omega = np.clip(config.yaw_kp * yaw_error, -config.max_omega, config.max_omega)
    return float(vx), float(vy), float(omega)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def interpolate_vr_poses(
    start: tuple[tuple[float, ...], tuple[float, ...]],
    target: tuple[tuple[float, ...], tuple[float, ...]],
    alpha: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    positions = (1.0 - alpha) * np.asarray(start[0]).reshape(3, 3) + alpha * np.asarray(target[0]).reshape(3, 3)
    orientations = []
    for start_quaternion, target_quaternion in zip(
        np.asarray(start[1]).reshape(3, 4),
        np.asarray(target[1]).reshape(3, 4),
    ):
        rotations = Rotation.from_quat(np.stack([start_quaternion, target_quaternion]), scalar_first=True)
        orientations.append(Slerp([0.0, 1.0], rotations)(alpha).as_quat(scalar_first=True))
    return tuple(positions.reshape(-1)), tuple(np.concatenate(orientations))


__all__ = [
    "CurrentHandPoseFK",
    "dolly_center_velocity",
    "interpolate_vr_poses",
    "smoothstep",
    "waist_relative_vr_targets",
]

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from numpy.typing import NDArray

from rs_imle_policy.utils.transforms import matrix_to_quaternion, rotation_6d_to_matrix


DEFAULT_ROBOT_IP = "172.16.0.2"
DEFAULT_HOME_Q = (0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4)
DEFAULT_RATE_HZ = 10.0
GRIPPER_CLOSE_THRESHOLD = 0.5


@dataclass
class ReplayConfig:
    dataset_path: Path
    episode: str = "0"
    backend: Literal["frankx", "pandapy"] = "frankx"
    trajectory: Literal["current", "action"] = "current"
    robot_ip: str = DEFAULT_ROBOT_IP
    rate_hz: float = DEFAULT_RATE_HZ
    dynamic_rel: float = 0.2
    start: int = 0
    limit: Optional[int] = None
    stride: int = 1
    home: Literal["episode", "default", "none"] = "episode"
    gripper_threshold: float = GRIPPER_CLOSE_THRESHOLD
    release_at_end: bool = True
    rerun: bool = True
    rerun_spawn: bool = True
    axes_length: float = 0.05
    point_radius: float = 0.006
    execute: bool = False
    yes: bool = False


@dataclass
class EpisodeData:
    state_path: Path
    episode_index: int
    episode_name: str
    poses: NDArray[np.float64]
    gripper_actions: Optional[NDArray[np.float64]]
    robot_q: Optional[NDArray[np.float64]]


def resolve_episode_index(dataset: object, episode: str) -> int:
    episode_name_to_index = getattr(dataset, "episode_name_to_index", {})
    candidates = [episode]
    if episode.isdigit():
        candidates.extend([str(int(episode)), episode.zfill(4)])

    for candidate in candidates:
        if candidate in episode_name_to_index:
            return int(episode_name_to_index[candidate])

    if episode.isdigit() and int(episode) in dataset.rlds:
        return int(episode)

    available = ", ".join(getattr(dataset, "episode_names", [])[:10])
    raise KeyError(f"Episode {episode!r} was not found. Available episode folders include: {available}")


def poses_from_action_fields(ep_data: dict[str, object]) -> NDArray[np.float64]:
    positions = np.asarray(ep_data["action_pos"], dtype=float)
    rot_6d = ep_data["action_orien"]
    if isinstance(rot_6d, torch.Tensor):
        rot_6d_tensor = rot_6d.to(dtype=torch.float64)
    else:
        rot_6d_tensor = torch.from_numpy(np.asarray(rot_6d, dtype=float))
    rotations = rotation_6d_to_matrix(rot_6d_tensor).numpy()

    poses = np.repeat(np.eye(4)[None, :, :], len(positions), axis=0)
    poses[:, :3, 3] = positions
    poses[:, :3, :3] = rotations
    return poses


def load_episode(config: ReplayConfig) -> EpisodeData:
    from rs_imle_policy.datasets.single_franka import PandaPolicyDataset

    dataset = PandaPolicyDataset.load_for_replay(config.dataset_path)
    episode_index = resolve_episode_index(dataset, config.episode)
    episode_name = dataset.episode_names[episode_index]
    ep_data = dataset.rlds[episode_index]

    if config.trajectory == "current":
        poses = np.asarray(ep_data["X_BE"], dtype=float).reshape(-1, 4, 4)
    else:
        poses = poses_from_action_fields(ep_data)

    gripper_actions = None
    if "action_gripper" in ep_data:
        gripper_actions = np.asarray(ep_data["action_gripper"], dtype=float).reshape(-1)

    robot_q = None
    if "robot_q" in ep_data:
        robot_q = np.asarray(ep_data["robot_q"], dtype=float).reshape(-1, 7)

    return EpisodeData(
        state_path=config.dataset_path / "episodes" / episode_name / "state.json",
        episode_index=episode_index,
        episode_name=episode_name,
        poses=poses,
        gripper_actions=gripper_actions,
        robot_q=robot_q,
    )


def select_indices(num_frames: int, start: int, limit: Optional[int], stride: int) -> NDArray[np.int64]:
    if start < 0:
        raise ValueError("--start must be non-negative.")
    if stride < 1:
        raise ValueError("--stride must be at least 1.")
    if start >= num_frames:
        raise ValueError(f"--start={start} is outside the episode with {num_frames} frames.")

    indices = np.arange(start, num_frames, stride)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive when provided.")
        indices = indices[:limit]
    if len(indices) == 0:
        raise ValueError("No frames selected for replay.")
    return indices


def quaternions_from_poses(poses: NDArray[np.float64]) -> NDArray[np.float64]:
    quats = matrix_to_quaternion(torch.from_numpy(poses[:, :3, :3])).numpy()
    for idx in range(1, len(quats)):
        if np.dot(quats[idx - 1], quats[idx]) < 0:
            quats[idx] *= -1.0
    return quats


def trajectory_colors(length: int) -> NDArray[np.uint8]:
    if length == 1:
        return np.asarray([[30, 180, 255]], dtype=np.uint8)
    t = np.linspace(0.0, 1.0, length)
    return np.stack(
        [
            255 * (1.0 - t),
            180 * np.ones_like(t),
            255 * t,
        ],
        axis=-1,
    ).astype(np.uint8)


def log_rerun_preview(
    config: ReplayConfig,
    episode: EpisodeData,
    selected_indices: NDArray[np.int64],
    poses: NDArray[np.float64],
) -> None:
    if not config.rerun:
        return

    import rerun as rr

    rr.init(
        "replay_dataset_episode",
        recording_id=f"{episode.episode_name}_{config.trajectory}",
        spawn=config.rerun_spawn,
    )
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    positions = poses[:, :3, 3]
    rr.log(
        "preview/trajectory_points",
        rr.Points3D(
            positions=positions,
            colors=trajectory_colors(len(positions)),
            radii=np.full(len(positions), config.point_radius),
        ),
    )

    for replay_idx, pose in enumerate(poses):
        frame_idx = int(selected_indices[replay_idx])
        rr.log(
            f"preview/poses/{frame_idx:05d}",
            rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
            rr.TransformAxes3D(axis_length=config.axes_length),
        )

    rr.log(
        "preview/start",
        rr.Transform3D(translation=poses[0, :3, 3], mat3x3=poses[0, :3, :3]),
        rr.TransformAxes3D(axis_length=config.axes_length * 1.5),
    )
    rr.log(
        "preview/end",
        rr.Transform3D(translation=poses[-1, :3, 3], mat3x3=poses[-1, :3, :3]),
        rr.TransformAxes3D(axis_length=config.axes_length * 1.5),
    )


def get_home_q(config: ReplayConfig, episode: EpisodeData, selected_indices: NDArray[np.int64]) -> Optional[NDArray]:
    if config.home == "none":
        return None
    if config.home == "episode" and episode.robot_q is not None:
        return episode.robot_q[selected_indices[0]]
    if config.home == "episode":
        print("Episode has no robot_q; using the default Franka home joint configuration.")
    return np.asarray(DEFAULT_HOME_Q, dtype=float)


def sleep_until_period(start_time: float, dt: float) -> None:
    remaining = dt - (time.perf_counter() - start_time)
    if remaining > 0:
        time.sleep(remaining)


def apply_gripper(robot: object, action: Optional[float], threshold: float) -> None:
    if action is None:
        return
    if action > threshold:
        robot.close_gripper()
    else:
        robot.open_gripper()


def confirm_execution(config: ReplayConfig, frame_count: int) -> None:
    if config.yes:
        return
    message = (
        f"Review the Rerun preview, then press Enter to move the robot with {frame_count} "
        f"frames through {config.backend} at up to {config.rate_hz:g} Hz. Ctrl-C aborts."
    )
    input(message)


def replay_with_frankx(
    config: ReplayConfig,
    episode: EpisodeData,
    selected_indices: NDArray[np.int64],
    positions: NDArray[np.float64],
    quaternions: NDArray[np.float64],
    gripper_actions: Optional[NDArray[np.float64]],
) -> None:
    from rs_imle_policy.robots.panda import FrankxRobot
    from rs_imle_policy.configs.panda_configs import SinglePandaConfig

    dt = 1.0 / config.rate_hz
    robot = FrankxRobot(config = SinglePandaConfig(robot_ip = config.robot_ip, dynamic_rel = config.dynamic_rel))
    home_q = get_home_q(config, episode, selected_indices)
    if home_q is not None:
        robot.move_to_start(home_q)

    robot.init_waypoint_motion()
    try:
        for idx, (position, quaternion) in enumerate(zip(positions, quaternions)):
            step_start = time.perf_counter()
            robot.set_next_waypoints(
                np.asarray([position], dtype=float),
                np.asarray([quaternion], dtype=float),
                relative=False,
            )
            action = None if gripper_actions is None else float(gripper_actions[idx])
            apply_gripper(robot, action, config.gripper_threshold)
            sleep_until_period(step_start, dt)
    finally:
        robot.stop_motion(release=config.release_at_end)
        if robot.move_async is not None:
            robot.move_async.join()


def replay_with_pandapy(
    config: ReplayConfig,
    episode: EpisodeData,
    selected_indices: NDArray[np.int64],
    gripper_actions: Optional[NDArray[np.float64]],
) -> None:
    from rs_imle_policy.robots.panda import PandaPyRobot
    from rs_imle_policy.configs.panda_configs import SinglePandaConfig

    if episode.robot_q is None:
        raise ValueError("The pandapy backend replays recorded robot_q joints, but this episode has no robot_q.")

    print("PandaPyRobot has no waypoint replay in panda.py; using blocking robot_q joint replay instead.")
    dt = 1.0 / config.rate_hz
    robot = PandaPyRobot(SinglePandaConfig(robot_ip=config.robot_ip, dynamic_rel=config.dynamic_rel))
    home_q = get_home_q(config, episode, selected_indices)
    if home_q is not None:
        robot.move_to_start(home_q)

    try:
        for idx, q in enumerate(episode.robot_q[selected_indices]):
            step_start = time.perf_counter()
            robot.robot.move_to_joint_position(q)
            action = None if gripper_actions is None else float(gripper_actions[idx])
            apply_gripper(robot, action, config.gripper_threshold)
            sleep_until_period(step_start, dt)
    finally:
        if config.release_at_end:
            robot.open_gripper()


def main(config: ReplayConfig) -> None:
    if config.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive.")

    episode = load_episode(config)
    selected_indices = select_indices(len(episode.poses), config.start, config.limit, config.stride)
    poses = episode.poses[selected_indices]
    positions = poses[:, :3, 3]
    quaternions = quaternions_from_poses(poses)
    gripper_actions = None if episode.gripper_actions is None else episode.gripper_actions[selected_indices]

    print(f"Loaded {len(episode.poses)} raw frames from {episode.state_path}")
    print(f"Using trajectory {config.trajectory!r}; selected {len(selected_indices)} replay frames.")
    print(f"First xyz: {positions[0].round(4).tolist()}  Last xyz: {positions[-1].round(4).tolist()}")

    log_rerun_preview(config, episode, selected_indices, poses)
    if not config.execute:
        print("Preview only. Re-run with --execute to connect to the robot and replay.")
        return

    confirm_execution(config, len(selected_indices))
    if config.backend == "frankx":
        replay_with_frankx(config, episode, selected_indices, positions, quaternions, gripper_actions)
    elif config.backend == "pandapy":
        replay_with_pandapy(config, episode, selected_indices, gripper_actions)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")


if __name__ == "__main__":
    main(tyro.cli(ReplayConfig))

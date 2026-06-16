import os
from dataclasses import replace
from pathlib import Path

import h5py
import pims
import cv2
import numpy as np
import tqdm
import tyro
from InquirerPy import inquirer

from rs_imle_policy.configs.train_config import VisionConfig


def _list_episodes(dataset_path: Path) -> list[str]:
    episodes_path = dataset_path / "episodes"
    return sorted(
        [episode for episode in os.listdir(episodes_path) if not episode.startswith(".")],
        key=int,
    )


def _available_video_serials(dataset_path: Path, episode: str) -> list[str]:
    video_path = dataset_path / "episodes" / episode / "video"
    if not video_path.exists():
        raise RuntimeError(f"Episode video directory does not exist: {video_path}")

    return sorted(video_file.stem for video_file in video_path.glob("*.mp4"))


def _select_cameras(dataset_path: Path, episodes: list[str], vision_config: VisionConfig) -> VisionConfig:
    first_episode = episodes[0]
    available_serials = _available_video_serials(dataset_path, first_episode)
    if not available_serials:
        raise RuntimeError(f"No .mp4 files found in first episode: {first_episode}")

    configured_by_serial = {camera.serial_number: camera for camera in vision_config.cameras_params}
    available_configured_cameras = [
        configured_by_serial[serial] for serial in available_serials if serial in configured_by_serial
    ]

    if not available_configured_cameras:
        configured_serials = ", ".join(configured_by_serial)
        discovered_serials = ", ".join(available_serials)
        raise RuntimeError(
            "No discovered videos match the configured cameras. "
            f"Configured serials: {configured_serials}. Discovered serials in episode {first_episode}: {discovered_serials}."
        )

    unknown_serials = [serial for serial in available_serials if serial not in configured_by_serial]
    if unknown_serials:
        print("Ignoring videos with serials that are not in VisionConfig:")
        for serial in unknown_serials:
            print(f"  - {serial}")
        print()

    choices = [
        {
            "name": f"{camera.name} ({camera.serial_number})",
            "value": camera.name,
            "enabled": camera.name in vision_config.cameras,
        }
        for camera in available_configured_cameras
    ]

    selected_cameras: list[str] = inquirer.checkbox(
        message=f"Which cameras do you want to save? Discovered from episode {first_episode}:",
        choices=choices,
        instruction="Use <space> to select, <enter> to confirm",
        validate=lambda result: len(result) > 0,
        invalid_message="Select at least one camera.",
    ).execute()

    return replace(vision_config, cameras=tuple(selected_cameras))


def convert_to_h5(dataset_path: str, /, vision_config: VisionConfig, select_cameras: bool = True):
    dataset_path_obj = Path(dataset_path)
    episodes = _list_episodes(dataset_path_obj)
    if not episodes:
        raise RuntimeError(f"No episodes found under {dataset_path_obj / 'episodes'}")

    if select_cameras:
        vision_config = _select_cameras(dataset_path_obj, episodes, vision_config)

    dataset_name = dataset_path_obj / "images.h5"
    # Open an HDF5 file in write mode
    with h5py.File(dataset_name, "w") as h5f:
        for episode in tqdm.tqdm(episodes):
            grp = h5f.create_group(f"{episode}")
            for ix, cam_name in enumerate(vision_config.cameras):
                serial = vision_config.cameras_params[ix].serial_number
                video = dataset_path_obj / "episodes" / episode / "video" / f"{serial}.mp4"
                if not video.exists():
                    raise RuntimeError(f"Selected camera {cam_name} ({serial}) is missing video: {video}")

                pims_video = pims.PyAVReaderIndexed(video)
                frames = np.array([cv2.resize(np.array(frame), vision_config.img_shape[::-1]) for frame in pims_video])

                # Assuming typical access of 16 frames at a time.
                chunk_size = (16, *vision_config.img_shape, 3)

                # Store wrist and side video frames in separate datasets with chunking
                grp.create_dataset(cam_name, data=frames, compression="gzip", chunks=chunk_size)
    print(f"Video data saved to {dataset_name}")
    with open(dataset_path_obj / "vision_config.yml", "w") as f:
        f.write(tyro.extras.to_yaml(vision_config))


if __name__ == "__main__":
    tyro.cli(convert_to_h5)

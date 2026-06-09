from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro



@dataclass
class Args:
    dataset: Path
    """Dataset name under data_root, dataset directory, or images.h5 path."""
    save_path: Path | None = None
    """Optional path at which to save the figure."""


def resolve_images_path(dataset: Path) -> Path:
    candidates = [dataset]
    for candidate in candidates:
        images_path = candidate / "images.h5" if candidate.is_dir() else candidate
        if images_path.is_file():
            return images_path

    raise FileNotFoundError(
        f"Could not find images.h5 for {dataset}. Checked: " + ", ".join(str(candidate) for candidate in candidates)
    )


def load_average_first_frames(images_path: Path) -> tuple[dict[str, np.ndarray], int]:
    with h5py.File(images_path, "r") as h5_file:
        episodes = [name for name, item in h5_file.items() if isinstance(item, h5py.Group)]
        if not episodes:
            raise ValueError(f"No episode groups found in {images_path}")

        first_episode = h5_file[episodes[0]]
        cameras = [name for name, item in first_episode.items() if isinstance(item, h5py.Dataset)]
        if not cameras:
            raise ValueError(f"No camera datasets found in episode {episodes[0]}")

        totals: dict[str, np.ndarray] = {}
        for episode in episodes:
            episode_group = h5_file[episode]
            for camera in cameras:
                if camera not in episode_group:
                    raise ValueError(f"Camera {camera!r} is missing from episode {episode}")

                camera_frames = episode_group[camera]
                if len(camera_frames) == 0:
                    raise ValueError(f"Camera {camera!r} has no frames in episode {episode}")

                frame = np.asarray(camera_frames[0], dtype=np.float64)
                if camera in totals and totals[camera].shape != frame.shape:
                    raise ValueError(
                        f"Camera {camera!r} frame shape changed from "
                        f"{totals[camera].shape} to {frame.shape} in episode {episode}"
                    )
                if camera not in totals:
                    totals[camera] = np.zeros_like(frame)
                totals[camera] += frame

    return {camera: total / len(episodes) for camera, total in totals.items()}, len(episodes)


def show_average_first_frames(
    average_frames: dict[str, np.ndarray],
    dataset_name: str,
    episode_count: int,
    save_path: Path | None,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(average_frames),
        figsize=(6 * len(average_frames), 5),
        squeeze=False,
    )

    for axis, (camera, average_frame) in zip(axes[0], average_frames.items()):
        axis.imshow(np.clip(np.rint(average_frame), 0, 255).astype(np.uint8))
        axis.set_title(camera)
        axis.axis("off")

    figure.suptitle(f"Average first frame across {episode_count} episodes: {dataset_name}")
    figure.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved average first-frame figure to {save_path}")

    plt.show()


def main(args: Args) -> None:
    images_path = resolve_images_path(args.dataset)
    average_frames, episode_count = load_average_first_frames(images_path)
    show_average_first_frames(
        average_frames,
        dataset_name=images_path.parent.name,
        episode_count=episode_count,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))

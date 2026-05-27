import json
import pims
import os
import time
from contextlib import suppress
from pathlib import Path

import cv2
import numpy as np
import tyro

from rs_imle_policy.configs.franka.evaluation import EvaluationConfig
from rs_imle_policy.inference import PerceptionSystem
from rs_imle_policy.robots.panda import FrankxRobot
from rs_imle_policy.visualizer.eval_utils import mean_image


FRANKA_HOME = np.deg2rad([0.0, 0.0, 0.0, -90.0, 0.0, 90.0, 45.0])
WINDOW_NAME = "Franka evaluation experiments"
DISPLAY_SIZE = (320, 240)


def _resize(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, DISPLAY_SIZE)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _blank(text: str) -> np.ndarray:
    frame = np.zeros((*DISPLAY_SIZE[::-1], 3), dtype=np.uint8)
    return _label(frame, text)


def _pad_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] == width:
        return frame
    pad = np.zeros((frame.shape[0], width - frame.shape[1], 3), dtype=np.uint8)
    return np.hstack([frame, pad])


def _make_display(
    live_frames: dict[str, np.ndarray],
    snapshot_frames: dict[str, list[np.ndarray]],
    n_experiments: int,
    target_experiments: int,
) -> np.ndarray:
    live_row = np.hstack(
        [_label(_resize(frame), f"{name}: live") for name, frame in live_frames.items()]
    )

    mean_tiles = []
    for name in live_frames:
        if snapshot_frames[name]:
            mean = mean_image([_resize(frame) for frame in snapshot_frames[name]])
            mean_tiles.append(_label(mean, f"{name}: snapshot mean"))
        else:
            mean_tiles.append(_blank(f"{name}: snapshot mean"))
    mean_row = np.hstack(mean_tiles)

    footer = np.zeros((44, live_row.shape[1], 3), dtype=np.uint8)
    status = (
        f"space/s: add experiment  q/esc: quit  "
        f"experiments {n_experiments}/{target_experiments}"
    )
    cv2.putText(
        footer,
        status,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    width = max(live_row.shape[1], mean_row.shape[1], footer.shape[1])
    return np.vstack(
        [
            _pad_to_width(live_row, width),
            _pad_to_width(mean_row, width),
            _pad_to_width(footer, width),
        ]
    )


def _experiment_dir(output_dir: Path, index: int) -> Path:
    return output_dir / f"experiment_{index:03d}"


def _save_experiment(
    output_dir: Path,
    index: int,
    camera_frames: dict[str, np.ndarray],
) -> dict:
    experiment_dir = _experiment_dir(output_dir, index)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    image_paths = {}
    for camera_name, frame in camera_frames.items():
        image_path = experiment_dir / f"{camera_name}.png"
        cv2.imwrite(str(image_path), frame)
        image_paths[camera_name] = str(image_path)

    return {
        "index": index,
        "timestamp": time.time(),
        "home_q": FRANKA_HOME.tolist(),
        "images": image_paths,
    }


def _write_manifest(
    output_dir: Path,
    args: EvaluationConfig,
    experiments: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": args.task_name,
        "home_q": FRANKA_HOME.tolist(),
        "cameras": list(args.vision.cameras),
        "camera_serials": {
            params.name: params.serial_number for params in args.vision.cameras_params
        },
        "experiments": experiments,
    }
    with (output_dir / "experiments.json").open("w") as f:
        json.dump(payload, f, indent=2)

def overlay(
    perception: PerceptionSystem,
    dataset_path: Path
):
    def load_cameras(dataset_path) -> dict[str,np.ndarray]:
        experiment = os.listdir(dataset_path / "episodes")[0]
        video_folder = dataset_path/ "episodes" /experiment / "video"
        frames: dict[str, np.ndarray] = {}

        for video_name in sorted(os.listdir(video_folder)):
            if not video_name.endswith(".mp4"):
                continue

            video_id = Path(video_name).stem
            video_path = video_folder / video_name

            pims_video = pims.PyAVReaderIndexed(str(video_path))

            if len(pims_video) == 0:
                raise ValueError(f"Video has no frames: {video_path}")

            frame = np.asarray(pims_video[0])

            # pims usually returns RGB, OpenCV expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frames[perception.serial_to_name(video_id)] = frame_bgr
        return frames

    camera_frames = load_cameras(dataset_path)
    print(camera_frames)
    print(perception.cams_config.cameras)

    while True:
        images = perception.cams.get() 
        for ix, cam_name in enumerate(perception.cams_config.cameras):
            if cam_name in camera_frames:
                img = cv2.addWeighted(images[ix]["color"], 0.5, camera_frames[cam_name], 0.5, 0, images[ix]["color"])
            else:
                img = images[ix]["color"]
            cv2.imshow(cam_name, img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows() 


def main(args: EvaluationConfig) -> None:
    output_dir = args.output_dir / args.task_name
    robot = FrankxRobot(ip=args.robot_ip)
    perception = PerceptionSystem(args.vision)
    experiments: list[dict] = []
    snapshot_frames = {camera_name: [] for camera_name in args.vision.cameras}
    perception_started = False

    robot.move_to_start(FRANKA_HOME)
    perception.start()
    perception_started = True

    print("Robot is at the fixed Franka home position.")
    print("Press space or s to add an experiment; q or esc exits.")

    if args.dataset_path:
        overlay(perception, args.dataset_path)
            
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while len(experiments) < args.n_experiments:
            frames = perception.cams.get()
            live_frames = {
                camera_name: frames[ix]["color"]
                for ix, camera_name in enumerate(args.vision.cameras)
            }

            cv2.imshow(
                WINDOW_NAME,
                _make_display(
                    live_frames,
                    snapshot_frames,
                    len(experiments),
                    args.n_experiments,
                ),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord(" ")):
                index = len(experiments)
                camera_frames = {name: frame.copy() for name, frame in live_frames.items()}
                experiments.append(_save_experiment(output_dir, index, camera_frames))
                for camera_name, frame in camera_frames.items():
                    snapshot_frames[camera_name].append(frame)
                _write_manifest(output_dir, args, experiments)
                print(f"Added experiment {index}: {output_dir / f'experiment_{index:03d}'}")

        _write_manifest(output_dir, args, experiments)
        print(f"Saved {len(experiments)} experiments to {output_dir}")
    finally:
        if perception_started:
            with suppress(Exception):
                perception.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(EvaluationConfig))

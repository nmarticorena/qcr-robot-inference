"""Small reusable helpers for sequential video replay and synchronization."""

from pathlib import Path

import cv2
import numpy as np


class SequentialVideoReader:
    """Read monotonically increasing frame indices without random seeking."""

    def __init__(self, path: Path, preserve_datatype: bool = False) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if preserve_datatype:
            self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self.index = -1
        self.frame: np.ndarray | None = None

    def read(self, index: int) -> np.ndarray:
        if index < self.index:
            raise ValueError(f"frame indices must be monotonic for {self.path}")
        while self.index < index:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {self.index + 1} from {self.path}")
            self.index += 1
            self.frame = frame
        if self.frame is None:
            raise RuntimeError(f"video has no frame at index {index}: {self.path}")
        return self.frame

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "SequentialVideoReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def resize_to_width(
    frame: np.ndarray,
    image_width: int,
    nearest: bool = False,
) -> np.ndarray:
    if image_width <= 0 or frame.shape[1] == image_width:
        return frame
    image_height = round(frame.shape[0] * image_width / frame.shape[1])
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(frame, (image_width, image_height), interpolation=interpolation)


def nearest_frame_indices(
    state_times: np.ndarray,
    video_times: np.ndarray,
) -> np.ndarray:
    """Match video frames to state samples using relative timestamps."""

    state_times = np.asarray(state_times)
    video_times = np.asarray(video_times)
    if state_times.size == 0 or video_times.size == 0:
        raise ValueError("timestamp arrays must not be empty")
    if np.any(np.diff(video_times) < 0.0):
        raise ValueError("video timestamps must be monotonic")
    state_times = state_times - state_times[0]
    video_times = video_times - video_times[0]
    insertion = np.clip(np.searchsorted(video_times, state_times), 0, len(video_times) - 1)
    previous = np.maximum(insertion - 1, 0)
    use_previous = np.abs(video_times[previous] - state_times) <= np.abs(video_times[insertion] - state_times)
    return np.where(use_previous, previous, insertion)


__all__ = ["SequentialVideoReader", "nearest_frame_indices", "resize_to_width"]

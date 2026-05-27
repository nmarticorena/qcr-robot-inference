import numpy as np


def mean_image(frames: list[np.ndarray]) -> np.ndarray:
    """Compute the mean image from a list of frames."""
    frames = np.stack(frames, axis=0)
    mean_img = np.mean(frames, axis=0).astype(np.uint8)
    return mean_img


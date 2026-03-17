"""Base dataset abstractions for robot policy learning.

This module provides a reusable ``BaseDataset`` class that owns common logic
for loading episodes, computing normalization statistics, indexing temporal
windows, and returning tensors for model training.
"""

import abc
import pathlib
import pickle as pkl
from collections import defaultdict
from typing import Callable, Optional, List
import matplotlib.pyplot as plt
import spatialmath as sm

import h5py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from numpy.typing import NDArray
from torch.utils.data import Dataset

from rs_imle_policy.configs.train_config import VisionConfig


class BaseDataset(Dataset, abc.ABC):
    """
    Base dataset example for robot manipulation demostrations

    This dataset loads and preprocesses demonstration data including robot states,
    actions, and multi-camera observations for training imitation learning policies.

    """

    def __init__(
        self,
        dataset_path: pathlib.Path,
        pred_horizon: int = 1,
        obs_horizon: int = 1,
        action_horizon: int = 1,
        transform: Optional[Callable] = None,
        low_dim_obs_keys: tuple[str, ...] = (),
        action_keys: tuple[str, ...] = (),
        vision_config: VisionConfig = VisionConfig(),
        visualize: bool = False,
    ):
        self.dataset_path = dataset_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.transform = transform
        self.low_dim_obs_keys = low_dim_obs_keys
        self.action_keys = action_keys
        self.vision_config = vision_config
        self.visualize = visualize

        self.transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop((216, 288)),
                transforms.ToTensor(),
            ]
        )

        self.rlds = self.create_rlds_dataset()
        self.stats: dict[str, dict[str, NDArray]] = defaultdict(dict)
        self.compute_normalization_stats()
        if not visualize:
            with open(self.dataset_path / "stats.pkl", "wb") as f:
                pkl.dump(dict(self.stats), f)

        self.indices = self.create_sample_indices(self.rlds, sequence_length=self.pred_horizon)
        self.normalize_rlds()

        self.cached_dataset = h5py.File(self.dataset_path / "images.h5", "r")
        assert self.cached_dataset is not None, "Failed to load cached dataset from HDF5 file."

        if not visualize:
            self.low_dim_obs_shape = self.rlds[0]["state"].shape[1]
            self.img_shape = vision_config.vision_features_dim * len(vision_config.cameras)
            self.obs_shape = self.low_dim_obs_shape + self.img_shape
            self.action_shape = self.rlds[0]["action"].shape[1]

    def get_relative_transform(self, current_pose: List[NDArray], next_pose: List[NDArray]) -> List[sm.SE3]:
        """Compute relative transformation between consecutive poses.

        Args:
            current_pose: List of current pose matrices
            next_pose: List of next pose matrices

        Returns:
            List of relative transformations as SE3 objects
        """
        current_pose_sm = [sm.SE3(pose) for pose in current_pose]
        next_pose_sm = [sm.SE3(pose) for pose in next_pose]

        relative_transform = [current.inv() * nxt for current, nxt in zip(current_pose_sm, next_pose_sm)]
        return relative_transform

    @abc.abstractmethod
    def create_rlds_dataset(self) -> dict[int, dict[str, NDArray]]:
        """Build an RLDS-like dictionary from raw dataset files."""

    def compute_normalization_stats(self):
        """Compute normalization statistics for all data keys."""

        def get_data(key: str):
            data = np.concatenate(
                [np.array(self.rlds[episode][key]) for episode in self.rlds.keys()],
                axis=0,
            )
            self.stats[key] = get_data_stats(data)

        for keys in self.rlds[0].keys():
            get_data(keys)

    def normalize_rlds(self) -> None:
        """Apply normalization to every key in each episode."""
        for episode in self.rlds:
            for key in self.rlds[episode]:
                self.rlds[episode][key] = normalize_data(np.array(self.rlds[episode][key]), self.stats[key])

    def create_sample_indices(self, rlds_dataset: dict, sequence_length: int = 16) -> NDArray:
        """Create valid sample indices for the dataset.

        Args:
            rlds_dataset: Dictionary of episode data
            sequence_length: Length of sequences to sample

        Returns:
            Array of sample indices with shape (N, 3) containing
            [episode_idx, start_idx, end_idx]
        """
        indices = []
        for episode in rlds_dataset.keys():
            key = next(iter(rlds_dataset[episode]))
            episode_length = len(rlds_dataset[episode][key])
            range_idx = episode_length - (sequence_length + 2)
            for idx in range(range_idx):
                buffer_start_idx = idx
                buffer_end_idx = idx + sequence_length
                indices.append([episode, buffer_start_idx, buffer_end_idx])
        return np.array(indices)

    def read_video_frames(self, episode: int, start_frame: int, end_frame: int) -> dict:
        """Read video frames from cached HDF5 file.

        Args:
            episode: Episode index
            start_frame: Start frame index
            end_frame: End frame index (not used, reads obs_horizon frames from start)

        Returns:
            Dictionary mapping camera names to frame arrays
        """
        frames = {}
        video = self.cached_dataset[str(episode)]
        for key in self.vision_config.cameras:
            frame = video[key][start_frame : start_frame + self.obs_horizon]
            frames[key] = np.array([self.transform(f) for f in frame])

        return frames

    def linear_progress(self, length: int) -> NDArray:
        """Generate linear progress values from 0 to 1.

        Args:
            length: Number of progress values to generate

        Returns:
            Array of progress values linearly spaced from 0 to 1
        """
        return np.linspace(0, 1, length)

    def visualize_images_in_row(self, tensor: torch.Tensor):
        """Visualize a batch of images in a single row.

        Args:
            tensor: Image tensor of shape (16, 3, 216, 288)
        """
        # Ensure the input tensor is in the right shape
        assert tensor.shape == (16, 3, 216, 288), "Tensor should have shape (16, 3, 216, 288)"

        # Create a grid of images in a single row
        grid_img = torchvision.utils.make_grid(tensor, nrow=16)  # Arrange 16 images in a single row
        # Convert the tensor to a numpy array for displaying
        plt.figure(figsize=(20, 5))  # Adjust figure size if necessary
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())  # Permute to get (H, W, C) for display
        plt.axis("off")  # Hide axis
        plt.show()

    def sample_sequence(self, episode: int, buffer_start_idx: int, buffer_end_idx: int) -> dict:
        """Sample a sequence from an episode.

        Args:
            episode: Episode index
            buffer_start_idx: Start index of the sequence
            buffer_end_idx: End index of the sequence

        Returns:
            Dictionary containing state, action, and frame data
        """
        frames = self.read_video_frames(episode, buffer_start_idx, buffer_end_idx)

        robot_state = self.rlds[episode]["state"][buffer_start_idx:buffer_end_idx]
        robot_action = self.rlds[episode]["action"][buffer_start_idx:buffer_end_idx]

        seq = {"state": robot_state, "action": robot_action, "frames": frames}
        return seq

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing 'state', 'action', and camera frame tensors
        """
        episode, buffer_start_idx, buffer_end_idx = self.indices[idx]
        seq = self.sample_sequence(episode, buffer_start_idx, buffer_end_idx)

        frames = {f"frame_{key}": seq["frames"][key] for key in self.vision_config.cameras}

        # Convert to tensors
        state = torch.tensor(seq["state"], dtype=torch.float32)
        action = torch.tensor(seq["action"], dtype=torch.float32)

        # discard unused observations
        state = state[: self.obs_horizon]

        return {
            "state": state,
            "action": action,
            **frames,
        }


def get_data_stats(data: NDArray) -> dict:
    """Compute normalization statistics for data.

    Args:
        data: Input data array

    Returns:
        Dictionary with 'min' and 'max' statistics
    """
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data: NDArray, stats: dict) -> NDArray:
    """Normalize data to [-1, 1] range.

    Args:
        data: Input data array
        stats: Dictionary containing 'min' and 'max' statistics

    Returns:
        Normalized data in range [-1, 1]
    """
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])  # Normalize to [0,1]
    ndata = ndata * 2 - 1  # Normalize to [-1,1]
    return ndata


def unnormalize_data(ndata: NDArray, stats: dict) -> NDArray:
    """Unnormalize data from [-1, 1] range to original range.

    Args:
        ndata: Normalized data in range [-1, 1]
        stats: Dictionary containing 'min' and 'max' statistics

    Returns:
        Data in original range
    """
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data

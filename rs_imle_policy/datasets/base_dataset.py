"""Base dataset abstractions for robot policy learning.

This module provides a reusable ``BaseDataset`` class that owns common logic
for loading episodes, computing normalization statistics, indexing temporal
windows, and returning tensors for model training.
"""

import abc
import pathlib
import pickle as pkl
from collections import defaultdict
from typing import Callable, Optional, List, Sequence, Self, Any, overload, TypeAlias, Mapping
import matplotlib.pyplot as plt
import spatialmath as sm

import h5py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from numpy.typing import NDArray
from torch.utils.data import Dataset

from rs_imle_policy.configs.train_config import VisionConfig, ExperimentConfig

Array: TypeAlias = NDArray[np.floating] | torch.Tensor
Stats: TypeAlias = Mapping[str, Array]


RAW_ORIENTATION_PREFIXES = ("orien_",)
RAW_ORIENTATION_SUFFIXES = ("_orien",)


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
        skip_normalization_keys: tuple[str, ...] = (),
        save_normalization_stats: Optional[bool] = None,
        normalization_stats: Optional[dict[str, dict[str, NDArray]]] = None,
        normalize: bool = True,
        load_images: bool = True,
        episode_names: Sequence[str | int] | None = None,
        action_mode: str = "absolute",
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
        self.skip_normalization_keys = ("gt",*skip_normalization_keys)
        self.save_normalization_stats = not visualize if save_normalization_stats is None else save_normalization_stats
        self.normalize = normalize
        self.load_images = load_images
        self.action_mode = action_mode

        self.selected_episode_names = (
            None if episode_names is None else tuple(str(episode) for episode in episode_names)
        )


        self.transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop((216, 288)),
                transforms.ToTensor(),
            ]
        )

        self.rlds = self.create_rlds_dataset()
        if not self.rlds:
            raise ValueError("Dataset contains no episodes after filtering.")

        self.stats: dict[str, dict[str, NDArray]] = defaultdict(dict)
        self.torch_stats: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        if normalization_stats is None:
            self.compute_normalization_stats()
        else:
            self.stats.update(normalization_stats)

        self.torch_stats = {
            key: {stat_key: torch.tensor(stat_value, dtype=torch.float32).to("cuda") for stat_key, stat_value in stat_dict.items()}
            for key, stat_dict in self.stats.items()
        }

        if self.save_normalization_stats:
            with open(self.dataset_path / "stats.pkl", "wb") as f:
                pkl.dump(dict(self.stats), f)

        self.indices = self.create_sample_indices(self.rlds, sequence_length=self.pred_horizon)
        if self.normalize:
            self.normalize_rlds()

        self.cached_dataset = None
        if self.load_images:
            self.cached_dataset = h5py.File(self.dataset_path / "images.h5", "r")
            assert self.cached_dataset is not None, "Failed to load cached dataset from HDF5 file."

        if not visualize:
            first_episode = next(iter(self.rlds))
            self.low_dim_obs_shape = self.rlds[first_episode]["state"].shape[1]
            self.action_shape = self.rlds[first_episode]["action"].shape[-1]

    def get_delta_transform(self, current_pose: List[NDArray], next_pose: List[NDArray]) -> List[sm.SE3]:
        """Compute delta transforms between consecutive poses.

        Args:
            current_pose: List of current pose matrices
            next_pose: List of next pose matrices

        Returns:
            List of delta transformations as SE3 objects
        """
        current_pose_sm = [sm.SE3(pose) for pose in current_pose]
        next_pose_sm = [sm.SE3(pose) for pose in next_pose]

        delta_transform = [current.inv() * nxt for current, nxt in zip(current_pose_sm, next_pose_sm)]
        return delta_transform

    def get_relative_transform(self, current_pose: List[NDArray], next_pose: List[NDArray]) -> List[sm.SE3]:
        """Compatibility alias for the old consecutive-step relative transform."""
        return self.get_delta_transform(current_pose, next_pose)

    def get_relative_horizon_transform(self, poses: List[NDArray]) -> NDArray:
        """Compute horizon targets relative to the anchor observation pose."""
        pose_sm = [sm.SE3(pose) for pose in poses]
        last_idx = len(pose_sm) - 1
        relative_horizon = []
        for sample_idx in range(len(pose_sm)):
            anchor_idx = min(sample_idx + self.obs_horizon - 1, last_idx)
            anchor_inv = pose_sm[anchor_idx].inv()
            sample_transforms = []
            for horizon_idx in range(self.pred_horizon):
                target_idx = min(sample_idx + horizon_idx + 1, last_idx)
                sample_transforms.append((anchor_inv * pose_sm[target_idx]).A)
            relative_horizon.append(sample_transforms)
        return np.asarray(relative_horizon)

    @abc.abstractmethod
    def create_rlds_dataset(self) -> dict[int, dict[str, NDArray]]:
        """Build an RLDS-like dictionary from raw dataset files."""

    @abc.abstractmethod
    def n_action_to_robot_action(self, naction: torch.Tensor, nstate:torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert normalized action to action for the robot"""

    def compute_normalization_stats(self):
        """Compute normalization statistics for all data keys."""

        def get_data(key: str) -> NDArray:
            return np.concatenate(
                [np.array(self.rlds[episode][key]) for episode in self.rlds.keys()],
                axis=0,
            )

        def get_key_stats(key: str, *, flatten_leading: bool = False) -> dict:
            data = get_data(key)
            if flatten_leading and data.ndim > 2:
                data = data.reshape(-1, data.shape[-1])
            if self.skip_normalization(key):
                return get_identity_data_stats(data)
            return get_data_stats(data)

        def get_composite_stats(keys: tuple[str, ...]) -> dict:
            mins = []
            maxes = []
            for key in keys:
                key_stats = get_key_stats(key, flatten_leading=True)
                mins.append(np.asarray(key_stats["min"]).reshape(-1))
                maxes.append(np.asarray(key_stats["max"]).reshape(-1))
            return {
                "min": np.concatenate(mins, axis=0),
                "max": np.concatenate(maxes, axis=0),
            }

        first_episode = next(iter(self.rlds))
        for keys in self.rlds[first_episode].keys():
            if keys == "state" and self.low_dim_obs_keys:
                self.stats[keys] = get_composite_stats(self.low_dim_obs_keys)
            elif keys == "action" and self.action_keys:
                self.stats[keys] = get_composite_stats(self.action_keys)
            elif keys in self.skip_normalization_keys:
                pass
            else:
                self.stats[keys] = get_key_stats(keys)

    def skip_normalization(self, key: str) -> bool:
        """Return True when a key should pass through normalization unchanged."""
        return (
            key in self.skip_normalization_keys
            or key.startswith(RAW_ORIENTATION_PREFIXES)
            or key.endswith(RAW_ORIENTATION_SUFFIXES)
        )

    def normalize_rlds(self) -> None:
        """Apply normalization to every key in each episode."""
        for episode in self.rlds:
            for key in self.rlds[episode]:
                if not self.skip_normalization(key):
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
        if self.cached_dataset is None:
            raise RuntimeError("Image cache was not loaded. Instantiate the dataset with load_images=True.")

        frames = {}
        video = self.cached_dataset[str(episode).zfill(4)]
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

    def build_action_array(self, episode_data: dict[str, NDArray]) -> NDArray:
        """Build the model action array from configured action keys."""
        action_parts = []
        for key in self.action_keys:
            values = np.asarray(episode_data[key])
            if self.action_mode == "relative":
                values = self.expand_to_action_horizon(values)
            action_parts.append(values)
        return np.concatenate(action_parts, axis=-1)

    def expand_to_action_horizon(self, values: NDArray) -> NDArray:
        """Expand per-timestep actions to one horizon action per sample start."""
        if values.ndim == 3:
            return values

        length = values.shape[0]
        horizon_offsets = np.arange(self.pred_horizon)
        sample_offsets = np.arange(length)[:, None]
        indices = np.minimum(sample_offsets + horizon_offsets, length - 1)
        return values[indices]

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
        if self.load_images:
            frames = self.read_video_frames(episode, buffer_start_idx, buffer_end_idx)
        else:
            frames = {"frames": self.rlds[episode]["images"][buffer_start_idx : buffer_end_idx]}

        robot_state = self.rlds[episode]["state"][buffer_start_idx:buffer_end_idx]
        robot_action = self.sample_action_sequence(episode, buffer_start_idx, buffer_end_idx)
        gt = self.rlds[episode]["gt"][buffer_start_idx:buffer_end_idx]

        seq = {"state": robot_state, "action": robot_action, "frames": frames, "gt": gt}
        return seq

    def sample_action_sequence(self, episode: int, buffer_start_idx: int, buffer_end_idx: int) -> NDArray:
        """Return the action tensor for a sampled training window."""
        robot_action = self.rlds[episode]["action"]
        if robot_action.ndim == 3:
            return robot_action[buffer_start_idx]
        return robot_action[buffer_start_idx:buffer_end_idx]

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
            Dictionary containing 'state', 'action', camera frame tensors and "gt" as the gt action poses
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
            "gt": seq["gt"],
            **frames,
        }

    @classmethod
    def from_config(
        cls,
        config:ExperimentConfig,
        *,
        episode_names: tuple[str, ...] | None = None,
        normalization_stats: dict[str, Any] | None = None,
        save_normalization_stats: bool | None = None,
        **kwargs,
    ) -> Self:
        return cls(
            config.dataset_path,
            pred_horizon=config.model.pred_horizon,
            obs_horizon=config.model.obs_horizon,
            action_horizon=config.model.action_horizon,
            low_dim_obs_keys=config.data.lowdim_obs_keys,
            action_keys=config.data.action_keys,
            vision_config=config.data.vision,
            action_mode=config.data.action_mode,
            episode_names=episode_names,
            normalization_stats=normalization_stats,
            save_normalization_stats=save_normalization_stats,
            **kwargs,
        )

    @classmethod
    def train_val(
        cls,
        config:ExperimentConfig,
        *,
        val_episode_count: int = 0,
        **kwargs,
    ) -> tuple[Self, Self | None]:
        val_episode_count = getattr(config.data, "val_episode_count", val_episode_count)
        if val_episode_count <= 0:
            return cls.from_config(config, **kwargs), None
        train_episodes, val_episodes = split_episode_names(
            config.dataset_path,
            val_episode_count=val_episode_count,
        )

        train_dataset = cls.from_config(
            config,
            episode_names=train_episodes,
            **kwargs,
        )

        #TODO: Here will be nicer to get the 
        val_dataset = cls.from_config(
            config,
            episode_names=val_episodes,
            normalization_stats=train_dataset.stats,
            save_normalization_stats=False,
            **kwargs,
        )

        return train_dataset, val_dataset

def split_episode_names(dataset_path:pathlib.Path, val_episode_count:int):
    episode_dir = dataset_path / "episodes"
    episode_names = sorted(
        [p.name for p in episode_dir.iterdir() if p.is_dir()],
        key = int,
    )
    if val_episode_count < 1:
            raise ValueError("val_episode_count must be at least 1.")

    if len(episode_names) <= val_episode_count:
        raise ValueError(
            f"Need more than {val_episode_count} episodes to create a train/val split; "
            f"found {len(episode_names)}."
        )

    return (
        tuple(episode_names[:-val_episode_count]),
        tuple(episode_names[-val_episode_count:]),
    )


def get_data_stats(data: NDArray) -> dict:
    """Compute normalization statistics for data.

    Args:
        data: Input data array

    Returns:
        Dictionary with 'min' and 'max' statistics
    """
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def get_identity_data_stats(data: NDArray) -> dict:
    """Compute stats that make min/max normalization a no-op."""
    feature_shape = np.asarray(data).shape[1:]
    return {
        "min": np.full(feature_shape, -1.0, dtype=np.float32),
        "max": np.full(feature_shape, 1.0, dtype=np.float32),
    }


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

@overload
def unnormalize_data(
    ndata: torch.Tensor,
    stats: Mapping[str, torch.Tensor],
) -> torch.Tensor: ...


@overload
def unnormalize_data(
    ndata: NDArray[np.floating],
    stats: Mapping[str, NDArray[np.floating]],
) -> NDArray[np.floating]: ...


def unnormalize_data(
    ndata: Array,
    stats: Stats,
) -> Array:
    """Unnormalize data from [-1, 1] range to original range."""

    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


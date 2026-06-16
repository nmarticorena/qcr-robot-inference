"""Dataset utilities for robot policy learning.

This module provides dataset classes and data normalization utilities
for training robot manipulation policies from demonstrations.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import roboticstoolbox as rtb

from rs_imle_policy.datasets.base_dataset import BaseDataset
from rs_imle_policy.utils import transforms as transform_utils


class PandaPolicyDataset(BaseDataset):
    """PyTorch dataset for robot manipulation demonstrations.

    This dataset loads and preprocesses demonstration data including robot states,
    actions, and multi-camera observations for training imitation learning policies.

    Attributes:
        dataset_path: Path to the dataset directory
        pred_horizon: Number of future actions to predict
        obs_horizon: Number of historical observations to use
        action_horizon: Number of actions to execute
        vision_config: Configuration for camera setup
        rlds: Dictionary containing episode data
        stats: Normalization statistics for each data key
        indices: Sample indices for dataset access
    """

    def __init__(
        self,
        *args,
        use_next_state: bool = True,
        **kwargs,
    ):
        """Initialize the policy dataset.

        Args:
            dataset_path: Path to dataset directory
            pred_horizon: Number of future actions to predict
            obs_horizon: Number of historical observations
            action_horizon: Number of actions to execute
            transform: Optional image transform
            low_dim_obs_keys: Keys for low-dimensional observations
            action_keys: Keys for action data
            vision_config: Camera configuration
            visualize: If True, skip saving stats (for visualization only)
            use_next_state: If True, use next robot state as action target
        """
        self.use_next_state = use_next_state
        self.robot = rtb.models.Panda()
        super().__init__(*args, **kwargs)

    @classmethod
    def load_for_replay(
        cls,
        dataset_path: str | os.PathLike,
        *,
        use_next_state: bool = True,
    ) -> "PandaPolicyDataset":
        """Load low-dimensional episode data without rewriting stats or images.

        This is intended for inspection/replay tools that need raw poses and
        gripper actions rather than normalized training samples.
        """
        return cls(
            Path(dataset_path),
            pred_horizon=1,
            obs_horizon=1,
            action_horizon=1,
            visualize=True,
            save_normalization_stats=False,
            normalize=False,
            load_images=False,
            use_next_state=use_next_state,
        )

    def create_rlds_dataset(self) -> dict:
        """Load and process demonstration episodes into RLDS format.

        Returns:
            Dictionary mapping episode indices to episode data
        """
        rlds = {}
        episodes = sorted(
            os.listdir(os.path.join(self.dataset_path, "episodes")),
            key=int,
        )
        if self.selected_episode_names is not None:
            requested_episodes = set(self.selected_episode_names)
            missing_episodes = sorted(requested_episodes.difference(episodes), key=int)
            if missing_episodes:
                raise ValueError(
                    "Requested episode folders were not found: "
                    + ", ".join(missing_episodes)
                )
            episodes = [episode for episode in episodes if episode in requested_episodes]

        self.episode_names = episodes
        self.episode_name_to_index = {episode: int(episode) for episode in episodes}

        for episode in episodes:
            episode_index = int(episode)
            episode_path = os.path.join(self.dataset_path, "episodes", episode, "state.json")

            with open(episode_path, "r") as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            df["idx"] = range(len(df))

            X_BE_current = df["robot_X_BE"].tolist()

            if self.use_next_state:
                X_BE_next = X_BE_current[1:]
                X_BE_next.append(X_BE_current[-1])
            else:
                X_BE_next = [(self.robot.fkine(np.array(q))).A for q in df["gello_q"]][1:]
                X_BE_next.append(X_BE_next[-1])

            delta_transform = self.get_delta_transform(X_BE_current, X_BE_next)
            relative_transform = self.get_relative_horizon_transform(X_BE_current)

            gripper_width = df["gripper_width"].tolist()
            gripper_width = np.array(gripper_width).reshape(-1, 1)
            gripper_action = df["gripper_action"].tolist()
            gripper_action = np.array(gripper_action).reshape(-1, 1)

            X_BE_current_pos, X_BE_current_orien = transform_utils.extract_robot_pos_orien(np.array(X_BE_current))
            X_BE_next_pos, X_BE_next_orien = transform_utils.extract_robot_pos_orien(np.array(X_BE_next))
            delta_pos, delta_orien = transform_utils.extract_robot_pos_orien(np.array(delta_transform))
            relative_pos, relative_orien = transform_utils.extract_robot_pos_orien(np.array(relative_transform))

            progress = self.linear_progress(len(df)).reshape(-1, 1)

            rlds[episode_index] = {
                "robot_pos": X_BE_current_pos,
                "robot_orien": X_BE_current_orien,
                "gripper_state": gripper_width,
                "action_pos": X_BE_next_pos,
                "action_orien": X_BE_next_orien,
                "delta_pos": delta_pos,
                "delta_orien": delta_orien,
                "relative_pos": relative_pos if self.action_mode == "relative" else delta_pos,
                "relative_orien": relative_orien if self.action_mode == "relative" else delta_orien,
                "action_gripper": gripper_action,
                "X_BE": X_BE_current,
                "gello_q": df["gello_q"].tolist(),
                "robot_q": df["robot_q"].tolist(),
                "progress": progress,
            }

            if len(self.low_dim_obs_keys) != 0:
                state = np.concatenate([rlds[episode_index][key] for key in self.low_dim_obs_keys], axis=-1)
                action = self.build_action_array(rlds[episode_index])

                rlds[episode_index]["state"] = state
                rlds[episode_index]["action"] = action
        return rlds


if __name__ == "__main__":
    import time

    dataset = PandaPolicyDataset("data/t_block_1", pred_horizon=16, obs_horizon=2, action_horizon=8)

    idx = 0
    while True:
        start_time = time.time()
        dataset.__getitem__(idx)
        print(f"Time taken: {time.time() - start_time}")
        idx += 1

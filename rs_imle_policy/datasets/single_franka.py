"""Dataset utilities for robot policy learning.

This module provides dataset classes and data normalization utilities
for training robot manipulation policies from demonstrations.
"""

import json
import os
from typing import List

import numpy as np
import pandas as pd
import roboticstoolbox as rtb
import spatialmath as sm
from numpy.typing import NDArray

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

        for episode_index, episode in enumerate(episodes):
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

            relative_transform = self.get_relative_transform(X_BE_current, X_BE_next)

            gripper_width = df["gripper_width"].tolist()
            gripper_width = np.array(gripper_width).reshape(-1, 1)
            gripper_action = df["gripper_action"].tolist()
            gripper_action = np.array(gripper_action).reshape(-1, 1)

            X_BE_current_pos, X_BE_current_orien = transform_utils.extract_robot_pos_orien(np.array(X_BE_current))
            X_BE_next_pos, X_BE_next_orien = transform_utils.extract_robot_pos_orien(np.array(X_BE_next))
            relative_pos, relative_orien = transform_utils.extract_robot_pos_orien(np.array(relative_transform))

            progress = self.linear_progress(len(df)).reshape(-1, 1)

            rlds[episode_index] = {
                "robot_pos": X_BE_current_pos,
                "robot_orien": X_BE_current_orien,
                "gripper_state": gripper_width,
                "action_pos": X_BE_next_pos,
                "action_orien": X_BE_next_orien,
                "relative_pos": relative_pos,
                "relative_orien": relative_orien,
                "action_gripper": gripper_action,
                "X_BE": X_BE_current,
                "gello_q": df["gello_q"].tolist(),
                "robot_q": df["robot_q"].tolist(),
                "progress": progress,
            }

            if len(self.low_dim_obs_keys) != 0:
                state = np.concatenate([rlds[episode_index][key] for key in self.low_dim_obs_keys], axis=-1)
                action = np.concatenate([rlds[episode_index][key] for key in self.action_keys], axis=-1)

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

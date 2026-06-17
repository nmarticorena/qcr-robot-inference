"""Dataset utilities for robot policy learning.

This module provides dataset classes and data normalization utilities
for training robot manipulation policies from demonstrations.
"""
from rs_imle_policy.datasets import unnormalize_data
from numpy.typing import NDArray

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import roboticstoolbox as rtb
import torch

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
            episode_names: Optional subset of episode folder names to load
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
                "gt" : np.array(X_BE_next),
            }

            if len(self.low_dim_obs_keys) != 0:
                state = np.concatenate([rlds[episode_index][key] for key in self.low_dim_obs_keys], axis=-1)
                action = self.build_action_array(rlds[episode_index])

                rlds[episode_index]["state"] = state
                rlds[episode_index]["action"] = action
        return rlds

    def n_action_to_robot_action(self, naction: NDArray, nstate: NDArray) -> dict[str, NDArray]:
        actions = unnormalize_data(naction, stats = self.stats["action"])
        states = unnormalize_data(nstate, stats = self.stats["state"])
        robot_action = {}
        robot_action["progress"] = actions[...,-1:]
        robot_action["gripper"] = actions[...,-1:]
        if self.action_mode in ("delta", "relative"):
            batch_pos = []
            batch_rot = []
            for batch_idx in range(actions.shape[0]):
                pos = states[batch_idx, -1, :3]
                rot = transform_utils.rotation_6d_to_matrix(states[batch_idx, -1, 3:9]).numpy()
                trans = actions[batch_idx, :, :3]
                rot_6d = actions[batch_idx, :, 3:9]
                rot_mats = transform_utils.rotation_6d_to_matrix(torch.from_numpy(rot_6d)).numpy()
                if self.action_mode == "delta":
                    trans, rot_mats = self.delta_action_to_absolute(trans, rot_mats, pos, rot)
                else:
                    trans, rot_mats = self.relative_action_to_absolute(trans, rot_mats, pos, rot)
                batch_pos.append(trans)
                batch_rot.append(rot_mats)

            robot_action["pos"] = np.stack(batch_pos, axis=0)
            robot_action["rot"] = np.stack(batch_rot, axis=0)
        else:
            robot_action["pos"] = actions[..., :3]
            robot_action["rot"] = transform_utils.rotation_6d_to_matrix(torch.from_numpy(actions[...,3:9])).numpy()

        return robot_action

    def delta_action_to_absolute(
        self,
        trans: NDArray,
        rots: NDArray,
        p0: NDArray,
        r0: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Transform consecutive delta actions to absolute actions.
        Args:
            trans (NDArray): Nx3 array of translations
            rots (NDArray): Nx3x3 array of rotation matrices
        Returns:
            tuple[NDArray, NDArray]: absolute translations and rotations
        """
        n_actions = len(trans)
        current_rot = r0
        current_pos = p0
        translations = np.empty_like(trans)
        rotations = np.empty_like(rots)
        for i in range(n_actions):
            rel_rot = rots[i]
            rel_trans = trans[i]
            rotations[i] = current_rot @ rel_rot
            translations[i] = current_pos + current_rot @ rel_trans
            current_rot = rotations[i]
            current_pos = translations[i]
        return translations, rotations

    def relative_action_to_absolute(
        self,
        trans: NDArray,
        rots: NDArray,
        p0: NDArray,
        r0: NDArray ,
    ) -> tuple[NDArray, NDArray]:
        """
        Transform anchor-relative actions to absolute actions.

        Every action is expressed with respect to the same anchor pose.
        """
        current_rot = r0 
        current_pos = p0
        rotations = current_rot @ rots
        translations = current_pos + np.einsum("ij,nj->ni", current_rot, trans)
        return translations, rotations

    transform_action_to_absolute = delta_action_to_absolute


        


if __name__ == "__main__":
    import time
    from rs_imle_policy.configs.franka.experiments_configs import AbsoluteActionsConfig 

    dataset = PandaPolicyDataset(dataset_path = Path("./data/red_fruit_in_black_pot"), 
                                 pred_horizon=16, 
                                 obs_horizon=2, 
                                 action_horizon=8,
                                 low_dim_obs_keys = AbsoluteActionsConfig().lowdim_obs_keys,
                                 action_keys = AbsoluteActionsConfig().action_keys,
                                 )

    idx = 0
    while True:
        start_time = time.time()
        dataset.__getitem__(idx)
        print(f"Time taken: {time.time() - start_time}")
        breakpoint()
        idx += 1

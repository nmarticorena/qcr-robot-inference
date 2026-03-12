"""Dataset utilities for robot policy learning.

This module provides dataset classes and data normalization utilities
for training robot manipulation policies from demonstrations.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import roboticstoolbox as rtb

from rs_imle_policy.datasets.base_dataset import BaseDataset
from rs_imle_policy.utils import transforms as transform_utils


class G1ArmsDataset(BaseDataset):
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
        urdf_path: str | None = None,
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
            urdf_path: Optional path to the G1 URDF file
        """
        self.use_next_state = use_next_state
        repo_root = Path(__file__).resolve().parents[2]
        resolved_urdf_path = (
            Path(urdf_path) if urdf_path else repo_root / "g1_no_hands.urdf"
        )
        self.robot = rtb.ERobot.URDF(str(resolved_urdf_path))
        super().__init__(*args, **kwargs)

    def create_rlds_dataset(self) -> dict:
        """Load and process demonstration episodes into RLDS format.

        Returns:
            Dictionary mapping episode indices to episode data
        """
        rlds = {}
        episodes = sorted(
            os.listdir(os.path.join(self.dataset_path, "episodes")),
            key=lambda x: int(x.split("_")[1]),
        )

        for episode_index, episode in enumerate(episodes):
            episode_path = os.path.join(
                self.dataset_path, "episodes", episode, "data.json"
            )

            with open(episode_path, "r") as f:
                data = json.load(f)["data"]

            df = pd.json_normalize(data)

            robot_state = df["states.body.qpos"].tolist()
            left_hand_X_BE_current = [
                self.robot.fkine(q, start="pelvis", end="left_hand_palm_link").A
                for q in robot_state
            ]
            right_hand_X_BE_current = [
                self.robot.fkine(q, start="pelvis", end="right_hand_palm_link").A
                for q in robot_state
            ]

            # to check thath the fkine was working
            # import rerun as rr
            #
            # from motion_tools.robot_gui import ReRunRobot
            #
            # rec = rr.RecordingStream("g1_arms_dataset")
            # rec.spawn()
            # g1 = ReRunRobot.g1(rec, target_frame = "pelvis")
            #
            # for left, right, q in zip(left_hand_X_BE_current, right_hand_X_BE_current, robot_state):
            #     left_t, left_r = left.t, left.R
            #     rec.log("left_hand",rr.Transform3D(
            #                     translation= left_t ,
            #                     mat3x3=left_r,
            #                     parent_frame="pelvis"),
            #             rr.TransformAxes3D(axis_length=0.1)
            #            )
            #     right_t, right_r = right.t, right.R
            #     rec.log("right_hand",rr.Transform3D(
            #                     translation= right_t ,
            #                     mat3x3=right_r,
            #                     parent_frame="pelvis"),
            #             rr.TransformAxes3D(axis_length=0.1)
            #            )
            #     g1.log(q)

            if self.use_next_state:
                left_hand_X_BE_next = left_hand_X_BE_current[1:]
                left_hand_X_BE_next.append(left_hand_X_BE_current[-1])
                right_hand_X_BE_next = right_hand_X_BE_current[1:]
                right_hand_X_BE_next.append(right_hand_X_BE_current[-1])
            else:
                raise (
                    NotImplementedError(
                        "Using current state as action target is not implemented yet"
                    )
                )
                # X_BE_next = [(self.robot.fkine(np.array(q))).A for q in df["gello_q"]][
                # 1:
                # ]
                # X_BE_next.append(X_BE_next[-1])

            relative_transform_left = self.get_relative_transform(
                left_hand_X_BE_current, left_hand_X_BE_next
            )
            relative_transform_right = self.get_relative_transform(
                right_hand_X_BE_current, right_hand_X_BE_next
            )

            left_hand_state = df["states.left_ee.qpos"].tolist()
            right_hand_state = df["states.right_ee.qpos"].tolist()

            left_hand_action = df["actions.left_ee.qpos"].tolist()
            right_hand_action = df["actions.right_ee.qpos"].tolist()

            left_hand_current_pos, left_hand_current_orien = (
                transform_utils.extract_robot_pos_orien(
                    np.array(left_hand_X_BE_current)
                )
            )
            left_hand_next_pos, left_hand_next_orien = (
                transform_utils.extract_robot_pos_orien(np.array(left_hand_X_BE_next))
            )
            left_hand_relative_pos, left_hand_relative_orien = (
                transform_utils.extract_robot_pos_orien(
                    np.array(relative_transform_left)
                )
            )

            right_hand_current_pos, right_hand_current_orien = (
                transform_utils.extract_robot_pos_orien(
                    np.array(right_hand_X_BE_current)
                )
            )
            right_hand_next_pos, right_hand_next_orien = (
                transform_utils.extract_robot_pos_orien(np.array(right_hand_X_BE_next))
            )
            right_hand_relative_pos, right_hand_relative_orien = (
                transform_utils.extract_robot_pos_orien(
                    np.array(relative_transform_right)
                )
            )

            progress = self.linear_progress(len(df)).reshape(-1, 1)

            rlds[episode_index] = {
                "left_robot_pos": left_hand_current_pos,
                "left_robot_orien": left_hand_current_orien,
                "left_action_pos": left_hand_next_pos,
                "left_action_orien": left_hand_next_orien,
                "left_relative_pos": left_hand_relative_pos,
                "left_relative_orien": left_hand_relative_orien,
                "right_robot_pos": right_hand_current_pos,
                "right_robot_orien": right_hand_current_orien,
                "right_action_pos": right_hand_next_pos,
                "right_action_orien": right_hand_next_orien,
                "right_relative_pos": right_hand_relative_pos,
                "right_relative_orien": right_hand_relative_orien,
                "left_hand_state": left_hand_state,
                "left_hand_action": left_hand_action,
                "right_hand_state": right_hand_state,
                "right_hand_action": right_hand_action,
                "progress": progress,
            }

            if len(self.low_dim_obs_keys) != 0:
                state = np.concatenate(
                    [rlds[episode_index][key] for key in self.low_dim_obs_keys], axis=-1
                )
                action = np.concatenate(
                    [rlds[episode_index][key] for key in self.action_keys], axis=-1
                )

                rlds[episode_index]["state"] = state
                rlds[episode_index]["action"] = action
        return rlds


if __name__ == "__main__":
    import time
    from pathlib import Path
    from rs_imle_policy.configs.train_config import G1VisionConfig
    import matplotlib.pyplot as plt

    dataset = G1ArmsDataset(
        Path("data/unitree/xr_teleoperate"),
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        low_dim_obs_keys=[
            "left_robot_pos",
            "left_robot_orien",
            "right_robot_pos",
            "right_robot_orien",
        ],
        action_keys=[
            "left_action_pos",
            "left_action_orien",
            "right_action_pos",
            "right_action_orien",
            "progress",
        ],
        vision_config=G1VisionConfig(),
    )

    n_data = len(dataset)

    times = np.empty(1000)
    for i in range(1000):
        start_time = time.time()
        dataset.__getitem__(i % n_data)
        print(f"Time taken: {time.time() - start_time}")
        times[i] = time.time() - start_time

    plt.hist(times, bins=50)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Data Loading Times")
    plt.savefig("data_loading_times_histogram.png")
    plt.show()

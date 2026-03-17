from rs_imle_policy.visualizer.rtb import RobotViz
import numpy as np
from rs_imle_policy.dataset import PolicyDataset, unnormalize_data
from rs_imle_policy.configs.train_config import LoaderConfig, VisionConfig
import rs_imle_policy.utilities as utils
import cv2
import spatialgeometry as sg

import tyro

args = tyro.cli(LoaderConfig)

with open(args.dataset_path / "vision_config.yml", "r") as f:
    vision_config = tyro.extras.from_yaml(VisionConfig, f)

dataset = PolicyDataset(
    args.path,
    vision_config=vision_config,
    visualize=True,
)
rlds = dataset.rlds
env = RobotViz()

robot_pos = sg.Axes(0.05)
env.env.add(robot_pos)

action_pos = sg.Axes(0.05)
env.env.add(action_pos)

arrow = sg.Arrow(0.1, color=(1, 0, 0))
env.env.add(arrow)
arrow.attach_to(robot_pos)

action_pose_2 = sg.Axes(0.025, color=(0, 1, 0))
env.env.add(action_pose_2)

for episode in rlds:
    ep_data = rlds[episode]
    ep_data_video = dataset.cached_dataset[str(episode)]

    for idx in range(len(ep_data["robot_pos"])):
        if idx % 2 != 0:
            continue

        gello_q = unnormalize_data(ep_data["gello_q"][idx], dataset.stats["gello_q"])
        robot_q = unnormalize_data(ep_data["robot_q"][idx], dataset.stats["robot_q"])

        X_BE_gello = env.robot.fkine(np.array(gello_q), "panda_link8")

        # create a pose at 0,0,0
        orien = unnormalize_data(ep_data["action_orien"][idx], dataset.stats["action_orien"])
        t = unnormalize_data(ep_data["action_pos"][idx], dataset.stats["action_pos"])

        pose = utils.pos_rot_to_se3(t, orien)
        action_pos.T = pose

        orien = unnormalize_data(ep_data["robot_orien"][idx], dataset.stats["robot_orien"])
        t = unnormalize_data(ep_data["robot_pos"][idx], dataset.stats["robot_pos"])
        r_pos = utils.pos_rot_to_se3(t, orien)
        robot_pos.T = r_pos

        arrow_length = unnormalize_data(ep_data["relative_pos"][idx], dataset.stats["relative_pos"])
        arrow_dir = unnormalize_data(ep_data["relative_orien"][idx], dataset.stats["relative_orien"])

        pose_2 = utils.pos_rot_to_se3(arrow_length, arrow_dir)
        action_pose_2.T = r_pos * pose_2

        env.step(robot_q, gello_q)

        # show image
        frames = []
        for camera in vision_config.cameras:
            frame = ep_data_video[camera][idx]
            frames.append(frame)
        merged = np.hstack(frames)

        # show one combined window
        cv2.imshow("viewer", merged)
        cv2.waitKey(1)

import time
import roboticstoolbox as rtb
from rs_imle_policy.dataset import PolicyDataset, unnormalize_data
from rs_imle_policy.configs.train_config import VisionConfig
from rs_imle_policy.configs.default_configs import ExperimentConfigChoice
from rs_imle_policy.visualizer import rerun_tools
import rs_imle_policy.utils.transforms as transform_utils

import rerun as rr
import tyro

args = tyro.cli(ExperimentConfigChoice)

with open(args.dataset_path / "vision_config.yml", "r") as f:
    vision_config = tyro.extras.from_yaml(VisionConfig, f)

dataset = PolicyDataset(
    args.dataset_path,
    vision_config=vision_config,
    low_dim_obs_keys=args.data.lowdim_obs_keys,
    action_keys=args.data.action_keys,
    use_next_state=args.data.use_next_state,
)
rlds = dataset.rlds

rr.init("replay_demonstration_rerun", recording_id="test", spawn=True)
gui = rerun_tools.ReRunRobot(rtb.models.Panda(), "leader")
gui_follower = rerun_tools.ReRunRobot(rtb.models.Panda(), "follower")


for episode in rlds:
    ep_data = rlds[episode]
    ep_data_video = dataset.cached_dataset[str(episode)]

    for idx in range(len(ep_data["state"])):
        gello_q = unnormalize_data(ep_data["gello_q"][idx], dataset.stats["gello_q"])
        robot_q = unnormalize_data(ep_data["robot_q"][idx], dataset.stats["robot_q"])

        current_pos = unnormalize_data(ep_data["robot_pos"][idx], dataset.stats["robot_pos"])
        current_orien = unnormalize_data(
            ep_data["robot_orien"][idx], dataset.stats["robot_orien"]
        )  # 6-D representation
        current_orien = transform_utils.rotation_6d_to_quat(current_orien).numpy()
        gui.log_pose(current_pos, current_orien, name="debug/robot_current_pose")

        action_pos = unnormalize_data(ep_data["action_pos"][idx], dataset.stats["action_pos"])
        action_orien = unnormalize_data(ep_data["action_orien"][idx], dataset.stats["action_orien"])
        action_orien = transform_utils.rotation_6d_to_quat(action_orien).numpy()
        gui.log_pose(action_pos, action_orien, name="debug/action_pose")

        relative_pos = unnormalize_data(ep_data["relative_pos"][idx], dataset.stats["relative_pos"])
        relative_orien = unnormalize_data(ep_data["relative_orien"][idx], dataset.stats["relative_orien"])
        relative_orien = transform_utils.rotation_6d_to_quat(relative_orien).numpy()
        gui.log_pose(relative_pos, relative_orien, name="debug/robot_current_pose/relative_pose")

        progress = unnormalize_data(ep_data["progress"][idx], dataset.stats["progress"])

        gui.log_robot_state(gello_q)
        gui_follower.log_robot_state(robot_q)
        for camera in vision_config.cameras:
            rr.log(camera, rr.Image(ep_data_video[camera][idx]).compress(jpeg_quality=80))

        rr.log("action/progress", rr.Scalars(progress))

        time.sleep(0.1)  # 10 hz is the frequency we loged the data

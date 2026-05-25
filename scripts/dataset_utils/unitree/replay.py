import rerun as rr
import cv2
import tyro
import tqdm

from motion_tools.robot_gui import ReRunRobot
from rs_imle_policy.datasets.g1_arms import G1ArmsDataset
from rs_imle_policy.configs.train_config import G1VisionConfig
from rs_imle_policy.datasets.base_dataset import unnormalize_data
from rs_imle_policy.configs.default_configs import ExperimentConfigChoice
import rs_imle_policy.utils.transforms as transform_utils
from rs_imle_policy.inspire import (
    INSPIRE_FTP_LEFT_JOINT_MAP,
    INSPIRE_FTP_RIGHT_JOINT_MAP,
    hand_state_to_urdf_map,
    unnormalize_inspire_angles,
)

args = tyro.cli(ExperimentConfigChoice)

with open(args.dataset_path / "vision_config.yml", "r") as f:
    vision_config = tyro.extras.from_yaml(G1VisionConfig, f)

dataset = G1ArmsDataset(
    args.dataset_path,
    vision_config=vision_config,
    low_dim_obs_keys=args.data.lowdim_obs_keys,
    action_keys=args.data.action_keys,
    use_next_state=args.data.use_next_state,
)
rlds = dataset.rlds

rec = rr.RecordingStream("replay_demonstration_rerun", recording_id="test")
# rec.spawn()

robot = ReRunRobot.g1(rec, target_frame="pelvis")
left_hand = ReRunRobot.left_ftp_hand(rec, target_frame="pelvis")
right_hand = ReRunRobot.right_ftp_hand(rec, target_frame="pelvis")


for episode in tqdm.tqdm(rlds):
    ep_data = rlds[episode]
    ep_data_video = dataset.cached_dataset[str(episode)]

    for idx in tqdm.tqdm(range(len(ep_data["state"])), leave=False):
        for key in "left", "right":
            pos, orien = ep_data[f"{key}_robot_pos"][idx], ep_data[f"{key}_robot_orien"][idx]
            pos, orien = (
                unnormalize_data(pos, dataset.stats[f"{key}_robot_pos"]),
                unnormalize_data(orien, dataset.stats[f"{key}_robot_orien"]),
            )
            pose = transform_utils.pos_rot_to_se3(pos, orien)
            robot.log_se3_transform(f"{key}_arm", pose[0])

        left_hand_state = unnormalize_data(ep_data["left_hand_state"][idx], dataset.stats["left_hand_state"])
        right_hand_state = unnormalize_data(ep_data["right_hand_state"][idx], dataset.stats["right_hand_state"])
        robot_state = unnormalize_data(ep_data["robot_state"][idx], dataset.stats["robot_state"])
        robot.log(robot_state)
        robot.rec.log("state/left_hand_state/unnormalized", rr.Scalars(left_hand_state))
        robot.rec.log("state/right_hand_state/unnormalized", rr.Scalars(right_hand_state))

        left_hand_state = unnormalize_inspire_angles(left_hand_state)
        right_hand_state = unnormalize_inspire_angles(right_hand_state)

        left_hand_state = hand_state_to_urdf_map(left_hand_state, INSPIRE_FTP_LEFT_JOINT_MAP)
        right_hand_state = hand_state_to_urdf_map(right_hand_state, INSPIRE_FTP_RIGHT_JOINT_MAP)

        left_hand.log_from_dict(left_hand_state)
        right_hand.log_from_dict(right_hand_state)

        for camera in vision_config.cameras:
            image = ep_data_video[camera][idx]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rec.log("cameras/{}".format(camera), rr.Image(image).compress(jpeg_quality=40))


rec.save(args.dataset_path / "data.rrd")

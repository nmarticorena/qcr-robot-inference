import collections
import os
from collections import defaultdict
import time
from typing import Tuple

import torch
import cv2
import rerun as rr
import numpy as np
import reactivex as rx
from reactivex.scheduler import NewThreadScheduler
from reactivex import operators as ops
import spatialmath as sm

from teleimager.image_client import ImageClient
from motion_tools.robot_gui import ReRunRobot

from rs_imle_policy.configs.train_config import (
    Diffusion,
    ExperimentConfig,
    RSIMLE,
)

import rs_imle_policy.utils.transforms as transforms_utils
from rs_imle_policy.policy import Policy
from rs_imle_policy.datasets.base_dataset import normalize_data, unnormalize_data
from rs_imle_policy.robots.g1 import G1RobotInterface

# Constants
DEFAULT_SEED = 42
DEFAULT_VIDEO_FPS = 10
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_REFRESH_RATE_HZ = 10
GRIPPER_CLOSE_THRESHOLD = 0.5
PROGRESS_COMPLETE_THRESHOLD = 0.95
OBSERVATION_WAIT_TIME_MS = 1
INFERENCE_TARGET_DT_MULTIPLIER = 4


class PerceptionSystem:
    """Manages camera perception for robot control.

    This class handles initialization and control of multiple RealSense cameras
    used for visual perception in robot inference tasks.

    Attributes:
        cams: MultiRealsense camera manager
        cams_config: Vision configuration parameters
    """

    def __init__(self, host, port):
        self.cams = ImageClient(host=host, request_port=port, request_bgr=True)

    def start(self):
        """Start the camera system and configure camera settings."""

    def stop(self):
        """Stop the camera system."""
        # TODO: Need to implement an stop method
        pass

    def get(self):
        """Get the latest frame"""
        bgr = self.cams.get_head_frame().bgr
        assert bgr is not None, "Failed to get head frame from camera"  # Type narrowing
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return [{"color": bgr, "depth": None, "rgb": rgb}]


class G1ArmsInferenceController:
    def __init__(
        self,
        config: ExperimentConfig,
        eval_name: str,
        timeout: int,
        dry_run: bool = False,
        simulation: bool = True,
    ):
        self.infer_idx = 0
        self.config = config
        self.eval_name = eval_name
        self.timeout = timeout
        self.dry_run = dry_run
        self.simulation = simulation

        self.rec = rr.RecordingStream(f"g1_arms_inference_{eval_name}")
        self.rec.spawn()
        self.rerun_gui = ReRunRobot.g1(self.rec, target_frame="pelvis")
        self.rerun_left_hand = ReRunRobot.left_ftp_hand(self.rec)
        self.rerun_right_hand = ReRunRobot.right_ftp_hand(self.rec)

        self.rerun_ik = ReRunRobot.g1_debug(self.rec, target_frame="pelvis")

        self.rerun_ik.apply_color([0, 1, 0, 0.5])
        self.all_frames = defaultdict(list)

        self.perception_system = PerceptionSystem(os.environ["G1_IP"], 60001)  # TODO: Need to get this from config
        self.robot = G1RobotInterface(simulation)
        self.setup_diffusion_policy()

    def run_experiments(self, episodes: int):
        """Run multiple evaluation episodes.

        Args:
            episodes: Number of episodes to run
        """
        for i in range(episodes):
            self.idx = i
            print(f"Starting episode {i + 1}/{episodes}")
            self.done = False
            if not self.simulation:
                input("Press Enter to start the next episode...")
            else:
                self.robot.reset_sim()
            self.obs_deque.clear()
            self.inference_loop()
            print(f"Finished episode {i + 1}/{episodes}")

    def get_observation(self):
        state = self.robot.get_state()
        images = self.perception_system.get()
        frames = {}
        for ix, cam_name in enumerate(self.config.data.vision.cameras):
            frames[cam_name] = images[ix]["color"]
            self.all_frames[cam_name].append(images[ix]["color"])
            self.rerun_gui.rec.log("cameras/{}".format(cam_name), rr.Image(frames[cam_name]).compress(80))

        self.rerun_gui.log(state.q)
        low_level_state = state.build_low_level_state()

        return {"state": low_level_state, **frames}

    def setup_diffusion_policy(self):
        """Initialize the policy model and observation buffer."""
        torch.cuda.empty_cache()
        self.policy = Policy(self.config, training=False)

        self.obs_horizon = self.config.model.obs_horizon
        self.obs_deque = collections.deque(maxlen=self.config.model.obs_horizon)

        if isinstance(self.config.model, RSIMLE):
            self.prev_traj = torch.randn(
                (1, self.config.model.pred_horizon, self.config.action_shape),
                device=self.policy.device,
            )

    def process_inference_vision(self, obs_deque):
        """Process visual observations through encoders.

        Args:
            obs_deque: Deque of observations containing state and camera images

        Returns:
            Tensor: Processed observation features ready for policy inference
        """
        cams = self.config.data.vision.cameras
        device = self.policy.device
        dtype = self.policy.precision

        agent_pos_np = np.stack([x["state"] for x in obs_deque])
        nagent_pos_np = normalize_data(agent_pos_np, stats=self.policy.stats["state"])
        nagent_pos = torch.from_numpy(nagent_pos_np).to(device, dtype=dtype)

        if isinstance(self.config.model, Diffusion):
            encoders = self.policy.ema_nets
        elif isinstance(self.config.model, RSIMLE):
            encoders = self.policy.nets
        else:
            raise NotImplementedError("Model not supported for inference.")

        image_features = []
        with torch.no_grad():
            for cam_name in cams:
                image = np.stack([x[cam_name] for x in obs_deque])
                input_image = torch.stack([self.policy.transform(img) for img in image])
                feat = encoders[f"vision_encoder_{cam_name}"](input_image.to(device, dtype))
                image_features.append(feat)

        obs_features = torch.cat(image_features + [nagent_pos], dim=-1)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        return obs_cond

    @torch.no_grad()
    def infer_action(self, obs_deque):
        self.infer_idx += 1
        obs_cond = self.process_inference_vision(obs_deque)
        # TODO: Assuming IMLE
        noise = torch.randn(
            (1, self.config.model.pred_horizon, self.config.action_shape),
            device=self.config.model.device,
        )
        # clip noise
        noise = torch.clamp(noise, -1, 1)
        naction = self.policy.nets["generator"](noise, global_cond=obs_cond)
        naction = naction.detach().to("cpu").numpy()[0]

        # unnormalize action
        action_pos = unnormalize_data(naction, stats=self.policy.stats["action"])

        # only take action_horizon number of actions
        start = self.config.model.obs_horizon - 1
        end = start + self.config.model.action_horizon
        action = action_pos[start:end]

        return {"action": action}

    @torch.no_grad()
    def convert_actions(self, action) -> Tuple[list[sm.SE3], list[sm.SE3], np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the raw action output from the policy into translation and rotation commands for the robot.
        Args:
            action:[n,19] Raw action output from the policy, expected to contain position and orientation components.
        """
        l_t = action[:, 0:3]
        l_r = action[:, 3:9]
        r_t = action[:, 9:12]
        r_r = action[:, 12:18]
        l_hand = action[:, 18:24]
        r_hand = action[:, 24:30]

        progress = action[:, 30:]

        left_poses = transforms_utils.pos_rot_to_se3(torch.from_numpy(l_t), torch.from_numpy(l_r))
        right_poses = transforms_utils.pos_rot_to_se3(torch.from_numpy(r_t), torch.from_numpy(r_r))

        return left_poses, right_poses, l_hand, r_hand, progress

    @torch.no_grad()  # Might be unncessary
    def inference_loop(self):
        """Main loop for running inference and controlling the robot."""
        obs_stream = (  # noqa: F841
            rx.interval(0.1, scheduler=NewThreadScheduler())
            .pipe(ops.map(lambda _: self.get_observation()))
            .subscribe(lambda x: self.obs_deque.append(x))
        )
        start_time = time.time()

        time.sleep(0.5)

        while not self.done:
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(OBSERVATION_WAIT_TIME_MS / 1000.0)
                print("Waiting for observation")

            infer_start_time = time.perf_counter()
            obs = self.obs_deque.copy()
            out = self.infer_action(obs)
            action = out["action"]

            print("elapsed time: ", time.time() - start_time)

            X_WL, X_WR, left_hand, right_hand, progress = self.convert_actions(action)

            n_actions = int(len(action) / 2)
            X_WL = X_WL[:n_actions]
            X_WR = X_WR[:n_actions]
            progress = progress[:n_actions]
            left_hand_actions = left_hand[:n_actions]
            right_hand_actions = right_hand[:n_actions]

            self.rerun_gui.rec.log("/plots/progress", rr.Scalars(progress[-1]))

            for pose_index, (x_wl, x_wr) in enumerate(zip(X_WL, X_WR)):
                self.rerun_gui.log_se3_transform(f"left_ee/{pose_index}", x_wl)
                self.rerun_gui.log_se3_transform(f"right_ee/{pose_index}", x_wr)

            for actions_index in range(n_actions):
                x_wl = X_WL[actions_index] @ sm.SE3(0.05, 0, 0)
                x_wr = X_WR[actions_index] @ sm.SE3(0.05, 0, 0)

                self.robot.set_ee_targets(X_WL[actions_index].A, X_WR[actions_index].A)
                for _ in range(1):
                    q_sol = self.robot.step_servo()
                    full_q_sol = np.zeros(29)
                    full_q_sol[15:29] = q_sol

                    self.rerun_ik.log(full_q_sol)

                # Send hand command in sync with arm step
                left_hand = left_hand_actions[actions_index]  # [0,1] normalized
                right_hand = right_hand_actions[actions_index]  # [0,1] normalized
                self.robot.hand_controller.policy_to_hand_command(left_hand, right_hand)
                time.sleep(INFERENCE_TARGET_DT_MULTIPLIER * 0.05)

            if progress[0] >= PROGRESS_COMPLETE_THRESHOLD:
                self.done = True
            elapsed_time = time.perf_counter() - infer_start_time
            rr.log("/debug/inference_time", rr.Scalars(elapsed_time))
            # remaining_time = target_dt - elapsed_time
            # if remaining_time > 0:
            #     time.sleep(remaining_time)

            if (time.time() - start_time) > self.timeout:
                print("Timeout reached, ending inference.")
                # obs_stream.dispose()
                self.done = True

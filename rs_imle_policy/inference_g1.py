import collections
from collections import defaultdict
import time
from dataclasses import dataclass

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

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

import rs_imle_policy.utils.transforms as transforms_utils
from rs_imle_policy.policy import Policy
from rs_imle_policy.unitree import G1_29_ArmController
from rs_imle_policy.g1_arm_ik import G1ReducedPinkIK
from rs_imle_policy.datasets.base_dataset import normalize_data, unnormalize_data

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
        return [{"color": rgb, "depth": None}]


class G1ArmsInferenceController:
    def __init__(
        self,
        config: ExperimentConfig,
        eval_name: str,
        timeout: int,
        dry_run: bool = False,
    ):
        self.infer_idx = 0
        self.config = config
        self.eval_name = eval_name
        self.timeout = timeout
        self.dry_run = dry_run

        self.rec = rr.RecordingStream(f"g1_arms_inference_{eval_name}")
        self.rec.spawn()
        self.rerun_gui = ReRunRobot.g1(self.rec, target_frame="pelvis")
        self.all_frames = defaultdict(list)

        self.perception_system = PerceptionSystem(
            "vlu-isaacsim.qut.edu.au", 60001
        )  # TODO: Need to get this from config
        self.robot = G1RobotInterface()
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
            input("Press Enter to start the next episode...")
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
            self.rerun_gui.rec.log(
                "cameras/{}".format(cam_name), rr.Image(frames[cam_name]).compress(80)
            )

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
                feat = encoders[f"vision_encoder_{cam_name}"](
                    input_image.to(device, dtype)
                )
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
    def convert_actions(self, action):
        """
        Convert the raw action output from the policy into translation and rotation commands for the robot.
        Args:
            action:[n,19] Raw action output from the policy, expected to contain position and orientation components.
        """
        l_t = action[:, 0:3]
        l_r = action[:, 3:9]
        r_t = action[:, 9:12]
        r_r = action[:, 12:18]
        progress = action[:, 18:]

        left_poses = transforms_utils.pos_rot_to_se3(
            torch.from_numpy(l_t), torch.from_numpy(l_r)
        )
        right_poses = transforms_utils.pos_rot_to_se3(
            torch.from_numpy(r_t), torch.from_numpy(r_r)
        )

        return left_poses, right_poses, progress

    def inference_loop(self):
        """Main loop for running inference and controlling the robot."""
        obs_stream = (  # noqa: F841
            rx.interval(0.1, scheduler=NewThreadScheduler())
            .pipe(ops.map(lambda _: self.get_observation()))
            .subscribe(lambda x: self.obs_deque.append(x))
        )
        start_time = time.time()

        time.sleep(0.5)

        # target_dt = 1.0 / DEFAULT_REFRESH_RATE_HZ * INFERENCE_TARGET_DT_MULTIPLIER

        while not self.done:
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(OBSERVATION_WAIT_TIME_MS / 1000.0)
                print("Waiting for observation")

            # infer_start_time = time.perf_counter()
            obs = self.obs_deque.copy()
            breakpoint()
            out = self.infer_action(obs)
            action = out["action"]

            print("elapsed time: ", time.time() - start_time)

            X_WL, X_WR, progress = self.convert_actions(action)

            for pose_index, (x_wl, x_wr) in enumerate(zip(X_WL, X_WR)):
                self.rerun_gui.log_se3_transform(f"left_ee/{pose_index}", x_wl)
                self.rerun_gui.log_se3_transform(f"right_ee/{pose_index}", x_wr)

            print(progress)

            """

            r = transform_utils.rotation_6d_to_matrix(action[:, 3:9])

            self.log_poses(
                n_trans, r.numpy(), relative=self.config.data.action_relative
            )
            progress = action[:, -1:]
            rr.log("/action/gripper", rr.Scalars(action[0, -2].tolist()))
            rr.log("/action/progress", rr.Scalars(action[0, -1].tolist()))

            action_horizon_len = int(len(action) / 2)
            relative = self.config.data.action_relative
            self.robot.set_next_waypoints(
                n_trans[0:action_horizon_len],
                n_quads[0:action_horizon_len],
                relative=relative,
            )
            for i in range(0, int(len(action) / 2)):
                time.sleep(1 / DEFAULT_REFRESH_RATE_HZ)
                if action[i][-2] > GRIPPER_CLOSE_THRESHOLD:
                    self.robot.close_gripper()
                else:
                    self.robot.open_gripper()

            if progress[0] >= PROGRESS_COMPLETE_THRESHOLD:
                self.robot.stop_motion()
                obs_stream.dispose()
                self.record_videos()
                self.done = True

            elapsed_time = time.perf_counter() - infer_start_time
            rr.log("/debug/inference_time", rr.Scalars(elapsed_time))
            remaining_time = target_dt - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)

            if (time.time() - start_time) > self.timeout:
                print("Timeout reached, ending inference.")
                self.robot.stop_motion()
                obs_stream.dispose()
                self.record_videos()
                self.done = True

            """


@dataclass
class RobotState:
    q: np.ndarray
    "Joint positions for the entire robot"
    q_arm: np.ndarray
    "Joint positions for the arms only"
    X_WL: sm.SE3
    "End-effector pose for the left arm in world frame"
    X_WR: sm.SE3
    "End-effector pose for the right arm in world frame"
    left_arm_state: np.ndarray
    "Concatenated position and orientation (6D) for the left arm end-effector"
    right_arm_state: np.ndarray
    "Concatenated position and orientation (6D) for the right arm end-effector"

    def build_low_level_state(self) -> np.ndarray:
        """Build a low-dimensional state representation for policy input."""
        return np.concatenate([self.left_arm_state, self.right_arm_state])


class G1RobotInterface:
    def __init__(self):
        ChannelFactoryInitialize(id=1)  # dds domain id
        self.controller = G1_29_ArmController(motion_mode=False, simulation_mode=True)
        q0 = self.controller.get_current_dual_arm_q()
        self.ik_solver = G1ReducedPinkIK(
            urdf_path="assets/g1.urdf",
            mesh_dirs=["assets/"],
            srdf_path="assets/g1.srdf",
            visualize=False,
            spawn_visualizer=False,
            enable_self_collision=True,
            q0=q0,
        )
        targets = self.ik_solver.get_targets_from_configuration()
        self.ik_solver.set_targets(targets.left, targets.right)

    def get_state(self) -> RobotState:
        """
        Get the current state of the robot, including joint positions and end-effector poses.
        Return:
            RobotState: A dataclass containing the robot's joint positions, end-effector poses, and low-level state representation.
        """
        q = self.controller.get_current_motor_q()
        q_arm = self.controller.get_current_dual_arm_q()

        poses = self.ik_solver.get_ee_poses(q_arm)
        X_WL = poses.left.homogeneous
        X_WR = poses.right.homogeneous

        pos_l, rot_l = transforms_utils.extract_robot_pos_orien(X_WL)
        pos_r, rot_r = transforms_utils.extract_robot_pos_orien(X_WR)

        # PENDING GET HANDS
        return RobotState(
            q=q,
            q_arm=q_arm,
            X_WL=X_WL,
            X_WR=X_WR,
            left_arm_state=np.concatenate([pos_l, rot_l]),
            right_arm_state=np.concatenate([pos_r, rot_r]),
        )

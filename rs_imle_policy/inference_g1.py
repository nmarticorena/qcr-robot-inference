import collections
import os
from collections import defaultdict

import time
from typing import Tuple, Optional

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

# TODO: Need to combine to make it less nasty
from rs_imle_policy.inspire import (
    INSPIRE_FTP_LEFT_JOINT_MAP,
    INSPIRE_FTP_RIGHT_JOINT_MAP,
    hand_state_to_urdf_map,
    unnormalize_inspire_angles,
)

import rs_imle_policy.utils.transforms as transforms_utils
from rs_imle_policy.policy import Policy
from rs_imle_policy.datasets.base_dataset import normalize_data, unnormalize_data
from rs_imle_policy.robots.g1 import G1RobotInterface
from rs_imle_policy.utils import viz as viz_utils

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

ACTION_KEY_DIMS = {
    "left_action_pos": 3,
    "left_action_orien": 6,
    "right_action_pos": 3,
    "right_action_orien": 6,
    "left_relative_pos": 3,
    "left_relative_orien": 6,
    "right_relative_pos": 3,
    "right_relative_orien": 6,
    "left_hand_action": 6,
    "right_hand_action": 6,
    "progress": 1,
}


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
        rec: rr.RecordingStream,
        eval_name: str,
        timeout: int,
        dry_run: bool = False,
        simulation: bool = True,
        home: Optional[np.ndarray] = None,
    ):
        self.infer_idx = 0
        self.config = config
        self.eval_name = eval_name
        self.timeout = timeout
        self.dry_run = dry_run
        self.simulation = simulation
        self.rec = rec
        self.latest_state = None

        # self.rec = rr.RecordingStream(f"g1_arms_inference_{eval_name}")
        # self.rec.spawn()
        self.rerun_gui = ReRunRobot.g1(self.rec, target_frame="pelvis")
        self.rerun_left_hand = ReRunRobot.left_ftp_hand(self.rec)
        self.rerun_right_hand = ReRunRobot.right_ftp_hand(self.rec)

        self.rerun_ik = ReRunRobot.g1_debug(self.rec, target_frame="pelvis")

        self.rerun_ik.apply_color([0, 1, 0, 0.5])
        self.all_frames = defaultdict(list)

        self.perception_system = PerceptionSystem(os.environ["G1_IP"], 60000)  # TODO: Need to get this from config
        self.robot = G1RobotInterface(simulation, home=home)
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
            self.robot.open_hands()
            self.robot.move_to_start()
            if not self.simulation:
                input("Press Enter to start the next episode...")
            else:
                self.robot.reset_sim()
            self.obs_deque.clear()
            self.inference_loop()
            print(f"Finished episode {i + 1}/{episodes}")

    def get_observation(self):
        state = self.robot.get_state()
        self.latest_state = state
        images = self.perception_system.get()
        frames = {}
        for ix, cam_name in enumerate(self.config.data.vision.cameras):
            frames[cam_name] = images[ix]["color"]
            self.all_frames[cam_name].append(images[ix]["color"])
            self.rerun_gui.rec.log("cameras/{}".format(cam_name), rr.Image(images[ix]["rgb"]).compress(80))

        self.rerun_gui.log(state.q)

        # TODO: Clean this mess
        left_hand_state, right_hand_state = state.left_hand_state, state.right_hand_state
        left_hand_state = unnormalize_inspire_angles(left_hand_state)
        right_hand_state = unnormalize_inspire_angles(right_hand_state)

        left_hand_state = hand_state_to_urdf_map(left_hand_state, INSPIRE_FTP_LEFT_JOINT_MAP)
        right_hand_state = hand_state_to_urdf_map(right_hand_state, INSPIRE_FTP_RIGHT_JOINT_MAP)

        self.rerun_left_hand.log_from_dict(left_hand_state)
        self.rerun_right_hand.log_from_dict(right_hand_state)


        low_level_state = state.build_low_level_state(self.config.data.lowdim_obs_keys)

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
                print("getting_image_feature", cam_name)

        obs_features = torch.cat(image_features + [nagent_pos], dim=-1)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        return obs_cond

    @torch.no_grad()
    def infer_action(self, obs_deque):
        self.infer_idx += 1
        obs_cond = self.process_inference_vision(obs_deque)

        if isinstance(self.config.model, Diffusion):
            naction = self._infer_diffusion(obs_cond)
        elif isinstance(self.config.model, RSIMLE):
            naction = self._infer_rsimle(obs_cond)
        else:
            raise NotImplementedError(f"Model type {type(self.config.model)} not supported for inference.")

        naction = naction.detach().cpu().numpy()[0]
        action_pos = unnormalize_data(naction, stats=self.policy.stats["action"])

        start = self.config.model.obs_horizon - 1
        end = start + self.config.model.action_horizon
        return {"action": action_pos[start:end]}

    def _infer_rsimle(self, obs_cond):
        assert isinstance(self.config.model, RSIMLE), "Model must be RSIMLE for this inference method"  # type narrowing
        if self.config.model.traj_consistency:
            print("using traj consistency")
            return self._infer_rsimle_with_consistency(obs_cond)

        noise = torch.clamp(
            torch.randn((1, self.config.model.pred_horizon, self.config.action_shape), device=self.policy.device),
            -1,
            1,
        )
        return self.policy.nets["generator"](noise, global_cond=obs_cond)

    def _infer_rsimle_with_consistency(self, obs_cond):
        assert isinstance(self.config.model, RSIMLE), "Model must be RSIMLE for this inference method"  # type narrowing
        noise = torch.randn(
            (32, self.config.model.pred_horizon, self.config.action_shape),
            device=self.policy.device,
        )
        batched_naction = self.policy.nets["generator"](noise, global_cond=obs_cond)

        prev_traj_end = self.prev_traj[:, 8:].reshape(1, -1)
        gen_traj_start = batched_naction[:, :8, :].reshape(32, -1)
        distances = torch.cdist(gen_traj_start, prev_traj_end)
        min_idx = distances.argmin(dim=0)

        self._debug_log_sampled_trajectories(batched_naction, distances)
        print("sampling_ id ", self.infer_idx)

        if self.infer_idx % self.config.model.periodic_length == 0:
            print("resampling")
            index = np.random.randint(0, 32)
            self.prev_traj = batched_naction[index].unsqueeze(0)
        else:
            self.prev_traj = batched_naction[min_idx]

        return batched_naction[min_idx]

    def _debug_log_sampled_trajectories(self, batched_naction, distances):
        action_debug = unnormalize_data(batched_naction.cpu().numpy(), stats=self.policy.stats["action"])
        positions = self._extract_debug_positions(action_debug)
        if positions.size == 0:
            return
        colors = distances.repeat_interleave(self.config.model.pred_horizon, 0)
        colors = colors.repeat_interleave(positions.shape[0] // colors.shape[0], 0)
        self.rec.log(
            "/debug/sampled_trajectories",
            rr.Transform3D(parent_frame="pelvis"),
            rr.Points3D(
                positions=positions,
                colors=viz_utils.colormap(colors.cpu().numpy(), colors.min().item(), colors.max().item()),
                radii=0.005,
            ),
        )

    def _infer_diffusion(self, obs_cond):
        naction = torch.randn(
            (1, self.config.model.pred_horizon, self.config.action_shape),
            device=self.policy.device,
            dtype=self.policy.precision,
        )

        assert self.policy.noise_scheduler is not None
        assert isinstance(self.config.model, Diffusion), (
            "Model must be Diffusion for this inference method"
        )  # type narrowing
        self.policy.noise_scheduler.set_timesteps(self.config.model.num_diffusion_iters)

        for k in self.policy.noise_scheduler.timesteps:
            noise_pred = self.policy.ema_nets["noise_pred_net"](sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.policy.noise_scheduler.step(
                model_output=noise_pred, timestep=int(k), sample=naction
            ).prev_sample

            self._debug_log_denoising(naction)

        return naction

    def _debug_log_denoising(self, naction):
        action_debug = unnormalize_data(naction[0].detach().cpu().numpy(), stats=self.policy.stats["action"])
        positions = self._extract_debug_positions(action_debug[None, ...])
        if positions.size == 0:
            return

        colors = np.tile(np.array([[0, 0.6, 1.0, 0.8]]), (positions.shape[0], 1))
        self.rec.log(
            "/debug/denoising_positions",
            rr.Transform3D(parent_frame="pelvis"),
            rr.Points3D(
                positions=positions,
                colors=colors,
                radii=0.004,
            ),
        )

    def _split_action_parts(self, action: np.ndarray) -> dict[str, np.ndarray]:
        parts = {}
        start = 0
        for key in self.config.data.action_keys:
            if key not in ACTION_KEY_DIMS:
                supported_keys = ", ".join(sorted(ACTION_KEY_DIMS))
                raise KeyError(f"Unsupported action key '{key}'. Supported keys: {supported_keys}")
            end = start + ACTION_KEY_DIMS[key]
            parts[key] = action[:, start:end]
            start = end

        if start != action.shape[1]:
            raise ValueError(
                f"Decoded {start} action dimensions from keys {self.config.data.action_keys}, "
                f"but model output has {action.shape[1]} dimensions."
            )
        return parts

    def _extract_debug_positions(self, action: np.ndarray) -> np.ndarray:
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor with shape (batch, horizon, dim), got {action.shape}.")

        action_parts = self._split_action_parts(action.reshape(-1, action.shape[-1]))
        position_keys = [
            key for key in self.config.data.action_keys if key.endswith("_action_pos") or key.endswith("_relative_pos")
        ]
        if not position_keys:
            return np.empty((0, 3), dtype=action.dtype)

        positions = [action_parts[key] for key in position_keys]
        return np.concatenate(positions, axis=0)

    def _repeat_pose(self, pose_matrix: np.ndarray, n_actions: int) -> list[sm.SE3]:
        return [sm.SE3(pose_matrix.copy()) for _ in range(n_actions)]

    def _resolve_arm_poses(
        self, parts: dict[str, np.ndarray], side: str, current_pose: np.ndarray, n_actions: int
    ) -> list[sm.SE3]:
        abs_pos_key = f"{side}_action_pos"
        abs_orien_key = f"{side}_action_orien"
        rel_pos_key = f"{side}_relative_pos"
        rel_orien_key = f"{side}_relative_orien"

        has_abs = abs_pos_key in parts or abs_orien_key in parts
        has_rel = rel_pos_key in parts or rel_orien_key in parts

        if has_abs and (abs_pos_key not in parts or abs_orien_key not in parts):
            raise KeyError(f"Both {abs_pos_key} and {abs_orien_key} are required together.")
        if has_rel and (rel_pos_key not in parts or rel_orien_key not in parts):
            raise KeyError(f"Both {rel_pos_key} and {rel_orien_key} are required together.")
        if has_abs and has_rel:
            raise KeyError(f"Action keys for {side} arm cannot mix absolute and relative targets.")

        if has_abs:
            return transforms_utils.pos_rot_to_se3(
                torch.from_numpy(parts[abs_pos_key]),
                torch.from_numpy(parts[abs_orien_key]),
            )

        if has_rel:
            relative_poses = transforms_utils.pos_rot_to_se3(
                torch.from_numpy(parts[rel_pos_key]),
                torch.from_numpy(parts[rel_orien_key]),
            )
            current = sm.SE3(current_pose.copy())
            return [current * rel_pose for rel_pose in relative_poses]

        return self._repeat_pose(current_pose, n_actions)

    @torch.no_grad()
    def convert_actions(self, action) -> Tuple[list[sm.SE3], list[sm.SE3], np.ndarray, np.ndarray, np.ndarray]:
        """Convert policy outputs into arm poses and hand commands using configured action keys."""
        n_actions = int(len(action))
        current_state = self.latest_state if self.latest_state is not None else self.robot.get_state()
        parts = self._split_action_parts(action)

        left_poses = self._resolve_arm_poses(parts, "left", current_state.X_WL, n_actions)
        right_poses = self._resolve_arm_poses(parts, "right", current_state.X_WR, n_actions)

        if "left_hand_action" in parts:
            left_hand = parts["left_hand_action"]
        else:
            left_hand = np.repeat(current_state.left_hand_state[None, :], n_actions, axis=0)

        if "right_hand_action" in parts:
            right_hand = parts["right_hand_action"]
        else:
            right_hand = np.repeat(current_state.right_hand_state[None, :], n_actions, axis=0)

        progress = parts.get("progress", np.zeros((n_actions, 1), dtype=action.dtype))

        return left_poses, right_poses, left_hand, right_hand, progress

    @torch.no_grad()  # Might be unncessary
    def inference_loop(self):
        """Main loop for running inference and controlling the robot."""
        obs_stream = (  # noqa: F841
            rx.interval(0.1, scheduler=NewThreadScheduler())
            .pipe(ops.map(lambda _: self.get_observation()))
            .subscribe(lambda x: self.obs_deque.append(x))
        )
        start_time = time.time()

        while not self.done:
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(OBSERVATION_WAIT_TIME_MS / 1000.0)

            infer_start_time = time.perf_counter()
            obs = self.obs_deque.copy()
            out = self.infer_action(obs)
            action = out["action"]


            X_WL, X_WR, left_hand, right_hand, progress = self.convert_actions(action)

            n_actions = int(len(action))
            X_WL = X_WL[:n_actions]
            X_WR = X_WR[:n_actions]
            progress = progress[:n_actions]
            left_hand_actions = left_hand[:n_actions]
            right_hand_actions = right_hand[:n_actions]

            self.rerun_gui.rec.log("/state/progress", rr.Scalars(progress[-1]))
            self.rerun_gui.rec.log("/state/elapsed_time", rr.Scalars((time.time()-start_time)/self.timeout))
            self.rerun_gui.rec.log("/state/max_time", rr.Scalars(1))

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
                self.rerun_gui.rec.log("state/left_hand_action", rr.Scalars(left_hand))
                time.sleep(0.05)
                # self.rerun_gui.rec.log("state/rigth_hand_action", rr.Scalars(right_hand))
                # time.sleep(INFERENCE_TARGET_DT_MULTIPLIER * 0.05)

            if progress[0] >= PROGRESS_COMPLETE_THRESHOLD:
                self.done = True
            elapsed_time = time.perf_counter() - infer_start_time
            self.rec.log("/debug/inference_time", rr.Scalars(elapsed_time))
            # remaining_time = target_dt - elapsed_time
            # if remaining_time > 0:
            #     time.sleep(remaining_time)

            if (time.time() - start_time) > self.timeout:
                print("Timeout reached, ending inference.")
                # obs_stream.dispose()
                self.done = True

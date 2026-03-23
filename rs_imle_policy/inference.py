import collections
import os
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import rerun as rr
import roboticstoolbox as rtb
import spatialmath as sm
import torch
from numpy.typing import NDArray
import reactivex as rx
from reactivex import operators as ops
from reactivex.scheduler import NewThreadScheduler

import rs_imle_policy.utils.transforms as transform_utils
import rs_imle_policy.utils.viz as viz_utils
from rs_imle_policy.configs.train_config import (
    Diffusion,
    ExperimentConfig,
    RSIMLE,
    VisionConfig,
)
from rs_imle_policy.datasets.base_dataset import normalize_data, unnormalize_data
from rs_imle_policy.policy import Policy
from rs_imle_policy.realsense.multi_realsense import MultiRealsense
from rs_imle_policy.robots import FrankxRobot, PandaPyRobot
from rs_imle_policy.visualizer.rerun_tools import ReRunRobot

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

    def __init__(self, vision_config: VisionConfig):
        serial_numbers = [cam.serial_number for cam in vision_config.cameras_params]
        self.cams = MultiRealsense(
            serial_numbers=serial_numbers,
            resolution=vision_config.cameras_params[0].resolution,
            record_fps=vision_config.cameras_params[0].frame_rate,
            depth_resolution=vision_config.cameras_params[0].depth_resolution,
            enable_depth=vision_config.cameras_params[0].depth_enabled,
        )
        self.cams_config = vision_config

    def start(self):
        """Start the camera system and configure camera settings."""
        for cam_params in self.cams_config.cameras_params:
            self.cams.cameras[cam_params.serial_number].set_exposure(exposure=cam_params.exposure, gain=cam_params.gain)
        self.cams.start()

    def stop(self):
        """Stop the camera system."""
        self.cams.stop()


class RobotInferenceController:
    """Controller for robot inference with visual policy.

    This class manages the complete inference pipeline including perception,
    policy inference, and robot control for executing learned manipulation tasks.

    Attributes:
        config: Experiment configuration
        eval_name: Name of the evaluation run
        timeout: Maximum time (seconds) for an episode
        dry_run: Flag for dry run mode
        robot: Frankx robot controller
        perception_system: Camera perception system
        policy: Trained policy model
        obs_deque: Observation history buffer
        gui: Rerun visualization interface
    """

    def __init__(
        self,
        config: ExperimentConfig,
        eval_name: str,
        timeout: int,
        dry_run: bool = False,
    ):
        self.infer_idx = 0
        self.last_called_obs = time.time()
        self.seed(DEFAULT_SEED)
        self.config = config
        self.config.training = False
        if dry_run:
            self.robot = PandaPyRobot(dry_run=dry_run)
        else:
            self.robot = FrankxRobot(dry_run=dry_run)
        self.robot.move_to_start(np.deg2rad([-90, 0, 0, -90, 0, 90, 45]))
        self.timeout = timeout

        rtb_panda = rtb.models.Panda()
        self.gui = ReRunRobot(rtb_panda, "panda")

        self.perception_system = PerceptionSystem(config.data.vision)
        self.perception_system.start()
        self.setup_diffusion_policy()

        self.eval_name = eval_name
        self.all_frames = defaultdict(list)
        self.done = False
        self.idx = 0

        # create folder to save images
        os.makedirs(f"saved_evaluation_media/{self.eval_name}", exist_ok=True)

    def seed(self, seed: int):
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_diffusion_policy(self):
        """Initialize the policy model and observation buffer."""
        torch.cuda.empty_cache()
        self.policy = Policy(self.config)

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

    def get_observation(self):
        """Capture current robot state and camera frames.

        Returns:
            dict: Dictionary containing robot state and camera frames
        """
        state = self.robot.get_state()
        robot_state = self.robot.get_robot_state()
        images = self.perception_system.cams.get()

        self.gui.log_robot_state(robot_state.q)

        frames = {}
        for ix, cam_name in enumerate(self.config.data.vision.cameras):
            frames[cam_name] = images[ix]["color"]
            self.all_frames[cam_name].append(images[ix]["color"])
            self.gui.log_frame(images[ix]["color"], cam_name)

        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.record_videos()

        return {"state": state, **frames}

    def record_videos(self):
        """Save recorded camera frames as video files."""
        for cam_name in self.config.data.vision.cameras:
            save_path = f"saved_evaluation_media/{self.eval_name}/{self.idx}_{cam_name}.mp4"
            out = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                DEFAULT_VIDEO_FPS,
                (DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT),
            )
            for frame in self.all_frames[cam_name]:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(rgb_frame)
            out.release()

    def infer_action(self, obs_deque):
        """Infer action from observations using the policy model.

        Args:
            obs_deque: Deque of recent observations

        Returns:
            dict: Dictionary containing action sequence
        """
        self.infer_idx += 1

        obs_cond = self.process_inference_vision(obs_deque)

        with torch.no_grad():
            if isinstance(self.config.model, Diffusion):
                # Initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (1, self.config.model.pred_horizon, self.config.action_shape),
                    device=self.policy.device,
                    dtype=self.policy.precision,
                )
                naction = noisy_action
                # Initialize scheduler
                assert self.policy.noise_scheduler is not None
                self.policy.noise_scheduler.set_timesteps(self.config.model.num_diffusion_iters)

                for k in self.policy.noise_scheduler.timesteps:
                    # Predict noise
                    noise_pred = self.policy.ema_nets["noise_pred_net"](
                        sample=naction, timestep=k, global_cond=obs_cond
                    )

                    # Inverse diffusion step (remove noise)
                    naction = self.policy.noise_scheduler.step(
                        model_output=noise_pred, timestep=int(k), sample=naction
                    ).prev_sample

                    debug_denoising = unnormalize_data(naction[0].cpu().numpy(), stats=self.policy.stats["action"])
                    trans = debug_denoising[:, :3]
                    rot_6d = debug_denoising[:, 3:9]
                    rot_mat3x3 = transform_utils.rotation_6d_to_matrix(torch.from_numpy(rot_6d)).numpy()
                    for i in range(trans.shape[0]):
                        rr.log(
                            f"/debug/denoising_step/poses_{i}",
                            rr.Transform3D(
                                translation=trans[i],
                                mat3x3=rot_mat3x3[i],
                            ),
                            rr.TransformAxes3D(axis_length=0.1),
                        )

            elif isinstance(self.config.model, RSIMLE):
                if self.config.model.traj_consistency:
                    noise = torch.randn(
                        (32, self.config.model.pred_horizon, self.config.action_shape),
                        device=self.policy.device,
                    )
                    batched_naction = self.policy.nets["generator"](noise, global_cond=obs_cond)
                    prev_traj_end = self.prev_traj[:, 8:].reshape(1, -1)
                    gen_traj_start = batched_naction[:, :8, :].reshape(32, -1)

                    # Pick the generated trajectory that has its start closest to the end of the prev traj
                    distances = torch.cdist(gen_traj_start, prev_traj_end)
                    min_idx = distances.argmin(dim=0)
                    action_debug = unnormalize_data(
                        batched_naction.cpu().numpy(),
                        stats=self.policy.stats["action"],
                    )
                    naction = batched_naction[min_idx]

                    action_debug_pos = action_debug[:, :, :3].reshape(-1, 3)
                    colors = distances.repeat_interleave(self.config.model.pred_horizon, 0)
                    rr.log(
                        "/debug/sampled_trajectories",
                        rr.Points3D(
                            positions=action_debug_pos,
                            colors=viz_utils.colormap(
                                colors.cpu().numpy(),
                                colors.min().item(),
                                colors.max().item(),
                            ),
                            radii=0.005,
                        ),
                    )

                    if self.infer_idx % self.config.model.periodic_length == 0:
                        index = np.random.uniform(0, 32, size=1)[0].astype(int)
                        self.prev_traj = batched_naction[index, :, :].unsqueeze(0)

                    else:
                        self.prev_traj = naction

                else:
                    noise = torch.randn(
                        (1, self.config.model.pred_horizon, self.config.action_shape),
                        device=self.config.model.device,
                    )
                    # clip noise
                    noise = torch.clamp(noise, -1, 1)
                    naction = self.policy.nets["generator"](noise, global_cond=obs_cond)
            else:
                raise NotImplementedError("Model not supported for inference.")

        naction = naction.detach().to("cpu").numpy()[0]

        # unnormalize action
        action_pos = unnormalize_data(naction, stats=self.policy.stats["action"])

        # only take action_horizon number of actions
        start = self.config.model.obs_horizon - 1
        end = start + self.config.model.action_horizon
        action = action_pos[start:end]

        return {"action": action}

    def log_poses(self, trans: NDArray, rots: NDArray, relative: bool = False):
        """
        Log action poses to rerun
        Args:
            trans (NDArray): Nx3 array of translations
            rots (NDArray): Nx3x3 array of rotation matrices
            relative (bool): whether the poses are relative to each other
        """
        n_actions = len(trans)
        if relative:
            p0, r0 = self.robot.pos, self.robot.rot
            trans, rots = self.transform_action_to_absolute(trans, rots, p0, r0)
        for i in range(n_actions):
            rr.log(
                f"/action/pose_{i}/transform",
                rr.Transform3D(translation=trans[i], mat3x3=rots[i]),
                rr.TransformAxes3D(axis_length=0.1),
            )

    @staticmethod
    def transform_action_to_absolute(
        trans: NDArray,
        rots: NDArray,
        p0: Optional[NDArray] = None,
        r0: Optional[NDArray] = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Transform relative actions to absolute actions
        Args:
            trans (NDArray): Nx3 array of translations
            rots (NDArray): Nx3x3 array of rotation matrices
        Returns:
            tuple[NDArray, NDArray]: absolute translations and rotations
        """
        n_actions = len(trans)
        current_rot = np.eye(3) if r0 is None else r0
        current_pos = np.zeros(3) if p0 is None else p0
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

    def run_experiments(self, episodes: int):
        """Run multiple evaluation episodes.

        Args:
            episodes: Number of episodes to run
        """
        for i in range(episodes):
            self.idx = i
            print(f"Starting episode {i + 1}/{episodes}")
            self.done = False
            time.sleep(0.1)
            self.robot.move_to_start(np.deg2rad([-90, 0, 0, -90, 0, 90, 45]))

            input("Press Enter to start the next episode...")
            self.obs_deque.clear()
            self.inference_loop()
            self.all_frames = defaultdict(list)
            print(f"Finished episode {i + 1}/{episodes}")

    def inference_loop(self):
        """Main inference loop for executing robot policy."""
        self.robot.init_waypoint_motion()

        obs_stream = (
            rx.interval(0.1, scheduler=NewThreadScheduler())
            .pipe(ops.map(lambda _: self.get_observation()))
            .subscribe(lambda x: self.obs_deque.append(x))
        )

        start_time = time.time()

        time.sleep(0.5)

        all_actions = np.zeros((0, self.config.action_shape))

        target_dt = 1.0 / DEFAULT_REFRESH_RATE_HZ * INFERENCE_TARGET_DT_MULTIPLIER

        while not self.done:
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(OBSERVATION_WAIT_TIME_MS / 1000.0)
                print("Waiting for observation")

            infer_start_time = time.perf_counter()
            obs = self.obs_deque.copy()
            out = self.infer_action(obs)
            action = out["action"]
            all_actions = np.concatenate([all_actions, action], axis=0)

            print("elapsed time: ", time.time() - start_time)

            n_trans, n_quads, _ = self.convert_actions(action)

            r = transform_utils.rotation_6d_to_matrix(action[:, 3:9])

            self.log_poses(n_trans, r.numpy(), relative=self.config.data.action_relative)
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

        assert self.robot.move_async is not None
        self.robot.move_async.join()

    def convert_actions(self, action: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, list[sm.SE3]]:
        """Convert action array to different pose representations.

        Args:
            action: Action array with position, rotation, and gripper state

        Returns:
            Tuple of translations, quaternions, and SE3 poses
        """
        t = action[:, :3]
        rot = action[:, 3:-2]

        poses = transform_utils.pos_rot_to_se3(torch.from_numpy(t), rot)
        trans = t.tolist()
        quads = transform_utils.rotation_6d_to_quat(torch.from_numpy(rot)).numpy()

        return trans, quads, poses

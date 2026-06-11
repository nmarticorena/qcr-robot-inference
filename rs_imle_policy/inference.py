import collections
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import reactivex as rx
import rerun as rr
import roboticstoolbox as rtb
import spatialmath as sm
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from numpy.typing import NDArray
from reactivex import operators as ops
from reactivex.scheduler import NewThreadScheduler

import rs_imle_policy.utils.transforms as transform_utils
import rs_imle_policy.utils.viz as viz_utils
from rs_imle_policy.configs.eval_config import SinglePandaEvaluationConfig
from rs_imle_policy.configs.train_config import (
    Diffusion,
    ExperimentConfig,
    FlowMatching,
    RSIMLE,
    VisionConfig,
)
from rs_imle_policy.datasets.base_dataset import normalize_data, unnormalize_data
from rs_imle_policy.policy import Policy
from rs_imle_policy.realsense.multi_realsense import MultiRealsense
from rs_imle_policy.robots import FrankxRobot, PandaPyRobot
from rs_imle_policy.visualizer.rerun_tools import ReRunRobot

# Constants
DEFAULT_VIDEO_FPS = 10
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
OBSERVATION_WAIT_TIME_MS = 100
REFERENCE_DISPLAY_SIZE = (320, 240)
REFERENCE_OVERLAY_ALPHA = 0.5
EVALUATION_WINDOW_NAME = "Evaluation setup"
# DEFAULT_HOME_Q = np.deg2rad([-90, 0, 0, -90, 0, 90, 45])
DEFAULT_HOME_Q = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]


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
        self.serial_numbers = serial_numbers
        self.serial_to_name_map = {cam.serial_number: cam.name for cam in vision_config.cameras_params}

    def start(self):
        """Start the camera system and configure camera settings."""
        for cam_params in self.cams_config.cameras_params:
            self.cams.cameras[cam_params.serial_number].set_exposure(exposure=cam_params.exposure, gain=cam_params.gain)
        self.cams.start()

    def stop(self):
        """Stop the camera system."""
        self.cams.stop()

    def serial_to_name(self, serial: str):
        """Convert camera serial number to camera name.

        Args:
            serial: Serial number of the camera

        Returns:
            str: Name of the camera
        """
        return self.serial_to_name_map.get(serial, serial)
        



class RobotInferenceController:
    """Controller for robot inference with visual policy.

    This class manages the complete inference pipeline including perception,
    policy inference, and robot control for executing learned manipulation tasks.

    Attributes:
        config: Experiment configuration
        media_dir: Directory for evaluation results and recordings
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
        eval_details: SinglePandaEvaluationConfig,
        media_dir: Optional[Path] = None,
        home_q: Optional[NDArray] = None,
        folder: Optional[Path] = None,
    ):
        if isinstance(config.model, RSIMLE):
            config.model.traj_consistency = eval_details.traj_consistency
        self.infer_idx = 0
        self.folder = folder if folder is not None else eval_details.path
        self.media_dir = (
            Path("saved_evaluation_media") / config.task_name / eval_details.run_name
            if media_dir is None
            else media_dir
        )
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.last_called_obs = time.time()
        self.seed(eval_details.seed)
        self.config = config

        if eval_details.dry_run:
            self.robot = PandaPyRobot(eval_details.robot_config, dry_run=True)
        else:
            self.robot = FrankxRobot(eval_details.robot_config, dry_run=False)
        self.home_q = DEFAULT_HOME_Q if home_q is None else np.asarray(home_q, dtype=float)
        self.robot.move_to_start(self.home_q)
        self.timeout = eval_details.timeout

        rtb_panda = rtb.models.Panda()
        self.gui = ReRunRobot(rtb_panda, "panda")

        self.perception_system = PerceptionSystem(config.data.vision)
        self.perception_system.start()
        self.setup_diffusion_policy()

        self.video_output_dir = self.media_dir
        self.video_filename_prefix = ""
        self.all_frames = defaultdict(list)
        self.done = False
        self.idx = 0

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
        self.policy = Policy(self.config, training = False, folder=self.folder)

        self.obs_horizon = self.config.model.obs_horizon
        self.obs_deque = collections.deque(maxlen=self.config.model.obs_horizon)

        if isinstance(self.config.model, RSIMLE):
            self.prev_traj: Optional[torch.Tensor] = None
            self.prev_traj_pos: Optional[NDArray] = None
            self.prev_traj_rot: Optional[NDArray] = None

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

        encoders = self.policy.nets

        image_features = []
        with torch.no_grad():
            for cam_name in cams:
                image = np.stack([x[cam_name] for x in obs_deque])
                input_image = torch.stack([self.policy.transform(img) for img in image])
                encoder = encoders[f"vision_encoder_{cam_name}"]
                feat = encoder(input_image.to(device, dtype))
                self._log_spatialsoftmax_keypoints(cam_name, image[-1], encoder)
                image_features.append(feat)

        obs_features = torch.cat(image_features + [nagent_pos], dim=-1)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        return obs_cond

    def _log_spatialsoftmax_keypoints(self, cam_name: str, image: np.ndarray, encoder: torch.nn.Module) -> None:
        avgpool = getattr(encoder, "avgpool", None)
        kps = getattr(avgpool, "kps", None)
        if kps is None:
            return
        if isinstance(kps, tuple):
            kps = kps[0]

        kps = kps[-1].detach().float().cpu().numpy()
        h, w = image.shape[:2]
        crop_h, crop_w = self.config.data.vision.center_crop
        resized_h, resized_w = self.config.data.vision.img_shape
        offset_x = (resized_w - crop_w) / 2.0
        offset_y = (resized_h - crop_h) / 2.0

        xy = np.empty_like(kps, dtype=np.float32)
        xy[:, 0] = (offset_x + (kps[:, 0] + 1.0) * 0.5 * (crop_w - 1)) / (resized_w - 1) * (w - 1)
        xy[:, 1] = (offset_y + (kps[:, 1] + 1.0) * 0.5 * (crop_h - 1)) / (resized_h - 1) * (h - 1)
        rr.log(f"{self.gui.name}/{cam_name}_inference/spatialsoftmax_kps", rr.Points2D(xy, radii=3))
        self.gui.log_frame(image, cam_name + "_inference", quality = 80)

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
            save_path = self.video_output_dir / f"{self.video_filename_prefix}{cam_name}.mp4"
            out = cv2.VideoWriter(
                str(save_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                DEFAULT_VIDEO_FPS,
                (DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT),
            )
            for frame in self.all_frames[cam_name]:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(rgb_frame)
            out.release()

    def _set_video_output(self, run_id: int, attempt_index: Optional[int] = None) -> int:
        if attempt_index is None:
            attempt_index = self._next_attempt_index(run_id)

        self.video_output_dir = self._experiment_media_dir(run_id) / f"{attempt_index:04d}"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.video_filename_prefix = ""
        return attempt_index

    def _experiment_media_dir(self, experiment_index: int) -> Path:
        return self.media_dir / f"episode_{experiment_index}"

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
                assert isinstance(self.policy.noise_scheduler, DDPMScheduler)
                self.policy.noise_scheduler.set_timesteps(self.config.model.num_diffusion_iters)

                for k in self.policy.noise_scheduler.timesteps:
                    # Predict noise
                    noise_pred = self.policy.nets["noise_pred_net"](
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
                    if self.config.data.action_relative:
                        p0, r0 = self.robot.pos, self.robot.rot
                        trans, rot_mat3x3 = self.transform_action_to_absolute(trans, rot_mat3x3, p0, r0)

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
                    current_pos = self.robot.pos.copy()
                    current_rot = self.robot.rot.copy()
                    split_idx = self.config.model.pred_horizon // 2

                    comparison_actions = self.actions_for_consistency(
                        batched_naction,
                        current_pos,
                        current_rot,
                    )

                    if self.prev_traj is None or self.prev_traj_pos is None or self.prev_traj_rot is None:
                        min_idx = np.random.randint(0, batched_naction.shape[0])
                        distances = torch.zeros((batched_naction.shape[0], 1), device=self.policy.device)
                    else:
                        prev_comparison_actions = self.actions_for_consistency(
                            self.prev_traj,
                            self.prev_traj_pos,
                            self.prev_traj_rot,
                        )
                        prev_traj_end = prev_comparison_actions[:, split_idx:].reshape(1, -1)
                        gen_traj_start = comparison_actions[:, :split_idx].reshape(batched_naction.shape[0], -1)

                        # Pick the generated trajectory that has its start closest to the end of the prev traj.
                        distances = torch.cdist(gen_traj_start, prev_traj_end)
                        min_idx = int(distances.argmin().item())

                    naction = batched_naction[min_idx : min_idx + 1]
                    action_debug_pos = comparison_actions[:, :, :3].reshape(-1, 3).cpu().numpy()
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
                        index = np.random.randint(0, batched_naction.shape[0])
                        self.set_prev_traj(
                            batched_naction[index : index + 1],
                            current_pos,
                            current_rot,
                        )
                    else:
                        self.set_prev_traj(naction, current_pos, current_rot)

                else:
                    noise = torch.randn(
                        (1, self.config.model.pred_horizon, self.config.action_shape),
                        device=self.config.model.device,
                    )
                    # clip noise
                    noise = torch.clamp(noise, -1, 1)
                    naction = self.policy.nets["generator"](noise, global_cond=obs_cond)
            elif isinstance(self.config.model, FlowMatching):
                noisy_action = torch.randn((1, self.config.model.pred_horizon, self.config.action_shape), device=self.config.model.device) #, dtype=self.precision)
                naction = noisy_action

                ts = torch.linspace(0.0, 1.0, self.config.model.num_flow_iters+1, device=self.config.model.device)[:-1]
                dt = 1.0 / self.config.model.num_flow_iters
                for t in ts:
                    timestep = (t * self.config.model.timestep_integer_scaler).long()

                    # predict noise
                    pred = self.policy.nets['noise_pred_net'](
                        sample=naction,
                        timestep=timestep,
                        global_cond=obs_cond
                    )
                    naction = naction + pred * dt
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

    def set_prev_traj(self, traj: torch.Tensor, pos: NDArray, rot: NDArray) -> None:
        self.prev_traj = traj.detach().clone()
        self.prev_traj_pos = np.array(pos, copy=True)
        self.prev_traj_rot = np.array(rot, copy=True)

    def actions_for_consistency(
        self,
        nactions: torch.Tensor,
        pos: NDArray,
        rot: NDArray,
    ) -> torch.Tensor:
        actions = unnormalize_data(nactions.detach().cpu().numpy(), stats=self.policy.stats["action"])

        if self.config.data.action_relative:
            absolute_actions = actions.copy()
            for batch_idx in range(actions.shape[0]):
                trans = actions[batch_idx, :, :3]
                rot_6d = actions[batch_idx, :, 3:9]
                rot_mats = transform_utils.rotation_6d_to_matrix(torch.from_numpy(rot_6d)).numpy()
                trans, rot_mats = self.transform_action_to_absolute(trans, rot_mats, pos, rot)
                absolute_actions[batch_idx, :, :3] = trans
                absolute_actions[batch_idx, :, 3:9] = transform_utils.matrix_to_rotation_6d(rot_mats).numpy()
            actions = absolute_actions

        return torch.from_numpy(actions).to(self.policy.device, dtype=self.policy.precision)

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

    def run_experiments(self, episodes: int, initial_id: int = 0):
        """Run multiple evaluation episodes.

        Args:
            episodes: Exclusive max episode id to run
            initial_id: Episode id to use for the first run
        """
        if initial_id < 0:
            raise ValueError(f"Initial id must be non-negative, got {initial_id}.")
        if episodes <= initial_id:
            raise ValueError(
                f"Episodes must be greater than initial id when used as the max episode, "
                f"got episodes={episodes} and initial_id={initial_id}."
            )

        episode_ids = range(initial_id, episodes)
        total_episodes = episodes - initial_id
        for i, episode_id in enumerate(episode_ids):
            self.idx = episode_id
            attempt_index = self._set_video_output(episode_id)
            print(f"Starting episode {i + 1}/{total_episodes} (id {self.idx})")
            self.done = False
            time.sleep(0.1)
            self.robot.move_to_start(self.home_q)

            input("Press Enter to start the next episode...")
            self.obs_deque.clear()
            episode_log = self.inference_loop()
            self._save_attempt_result(
                {
                    "episode": episode_id,
                    "episode_id": episode_id,
                    "attempt_index": attempt_index,
                    "timestamp": time.time(),
                    "media_relative_path": self.video_output_dir.relative_to(self.media_dir).as_posix(),
                    **episode_log,
                }
            )
            self.all_frames = defaultdict(list)
            print(f"Finished episode {i + 1}/{total_episodes} (id {self.idx})")

    @staticmethod
    def _resize_reference_frame(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, REFERENCE_DISPLAY_SIZE)

    @staticmethod
    def _label_reference_frame(frame: np.ndarray, text: str) -> np.ndarray:
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            out,
            text,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    @staticmethod
    def _blank_reference_frame(text: str) -> np.ndarray:
        frame = np.zeros((*REFERENCE_DISPLAY_SIZE[::-1], 3), dtype=np.uint8)
        return RobotInferenceController._label_reference_frame(frame, text)

    @staticmethod
    def _overlay_reference_frame(
        live_frame: Optional[np.ndarray],
        reference_frame: Optional[np.ndarray],
        camera_name: str,
    ) -> np.ndarray:
        if live_frame is None:
            return RobotInferenceController._blank_reference_frame(f"{camera_name}: live missing")

        live = RobotInferenceController._resize_reference_frame(live_frame)
        if reference_frame is None:
            return RobotInferenceController._label_reference_frame(live, f"{camera_name}: live")

        reference = cv2.resize(reference_frame, (live.shape[1], live.shape[0]))
        overlay = cv2.addWeighted(
            live,
            1.0 - REFERENCE_OVERLAY_ALPHA,
            reference,
            REFERENCE_OVERLAY_ALPHA,
            0,
        )
        return RobotInferenceController._label_reference_frame(overlay, f"{camera_name}: overlay")

    @staticmethod
    def _pad_reference_width(frame: np.ndarray, width: int) -> np.ndarray:
        if frame.shape[1] == width:
            return frame
        pad = np.zeros((frame.shape[0], width - frame.shape[1], 3), dtype=np.uint8)
        return np.hstack([frame, pad])

    def _load_reference_frames(self, experiment: dict) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for camera_name, image_path in experiment.get("images", {}).items():
            frame = cv2.imread(str(Path(image_path)))
            if frame is None:
                raise FileNotFoundError(f"Could not read evaluation image: {image_path}")
            frames[camera_name] = frame
        return frames

    def _make_reference_display(
        self,
        live_frames: dict[str, np.ndarray],
        reference_frames: dict[str, np.ndarray],
        run_label: str,
    ) -> np.ndarray:
        overlay_tiles = []
        for camera_name in self.config.data.vision.cameras:
            live_frame = live_frames.get(camera_name)
            ref_frame = reference_frames.get(camera_name)
            overlay_tiles.append(self._overlay_reference_frame(live_frame, ref_frame, camera_name))

        overlay_row = np.hstack(overlay_tiles)

        footer = np.zeros((44, overlay_row.shape[1], 3), dtype=np.uint8)
        text = f"Align scene to target. enter/space: start  q/esc: abort  {run_label}"
        cv2.putText(
            footer,
            text,
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        width = max(overlay_row.shape[1], footer.shape[1])
        return np.vstack(
            [
                self._pad_reference_width(overlay_row, width),
                self._pad_reference_width(footer, width),
            ]
        )

    def wait_for_experiment_setup(self, experiment: dict, run_label: str) -> None:
        reference_frames = self._load_reference_frames(experiment)
        cv2.namedWindow(EVALUATION_WINDOW_NAME, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(EVALUATION_WINDOW_NAME, 1280, 720)
        while True:
            images = self.perception_system.cams.get()
            live_frames = {
                camera_name: images[ix]["color"]
                for ix, camera_name in enumerate(self.config.data.vision.cameras)
            }
            cv2.imshow(
                EVALUATION_WINDOW_NAME,
                self._make_reference_display(
                    live_frames,
                    reference_frames,
                    run_label,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (13, ord(" "), ord("s")):
                break
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Evaluation aborted by user.")
        cv2.destroyWindow(EVALUATION_WINDOW_NAME)

    @staticmethod
    def _prompt_experiment_success(run_label: str) -> bool:
        while True:
            response = input(f"Did {run_label} succeed? [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                return True
            if response in ("n", "no"):
                return False
            print("Please enter y or n.")

    def _save_attempt_result(self, result: dict) -> None:
        with (self.video_output_dir / "result.json").open("w") as f:
            json.dump(result, f, indent=2)

    def _run_evaluation_attempt(
        self,
        experiment: dict,
        run_label: str,
        episode_index: int,
        attempt_index: Optional[int] = None,
    ) -> dict:
        self.idx = self._evaluation_experiment_index(experiment, episode_index)
        attempt_index = self._set_video_output(self.idx, attempt_index)
        print(f"Starting {run_label}")
        self.done = False
        time.sleep(0.1)
        self.robot.move_to_start(self.home_q)
        self.wait_for_experiment_setup(experiment, run_label)
        self.obs_deque.clear()

        try:
            episode_log = self.inference_loop()
            succeeded = self._prompt_experiment_success(run_label)
            result = {
                "episode": episode_index,
                "episode_id": self.idx,
                "experiment_index": self.idx,
                "attempt_index": attempt_index,
                "success": succeeded,
                "timestamp": time.time(),
                "media_relative_path": self.video_output_dir.relative_to(self.media_dir).as_posix(),
                **episode_log,
            }
            print(f"Finished {run_label}")
            return result
        finally:
            self.all_frames = defaultdict(list)

    def _next_attempt_index(self, experiment_index: int) -> int:
        attempt_indices = []
        experiment_dir = self._experiment_media_dir(experiment_index)
        if experiment_dir.exists():
            attempt_indices = [
                int(attempt_dir.name)
                for attempt_dir in experiment_dir.iterdir()
                if attempt_dir.is_dir() and attempt_dir.name.isdigit()
            ]
        return max(attempt_indices, default=-1) + 1

    def _completed_attempt_indices(self, experiment_index: int) -> set[int]:
        experiment_dir = self._experiment_media_dir(experiment_index)
        if not experiment_dir.exists():
            return set()
        return {
            int(attempt_dir.name)
            for attempt_dir in experiment_dir.iterdir()
            if attempt_dir.is_dir()
            and attempt_dir.name.isdigit()
            and (attempt_dir / "result.json").exists()
        }

    @staticmethod
    def _evaluation_experiment_index(experiment: dict, fallback_index: int) -> int:
        return int(experiment.get("index", fallback_index))

    @staticmethod
    def _serialize_action_outputs(actions: np.ndarray) -> dict:
        if actions.size == 0:
            return {
                "action_count": 0,
                "action_poses": [],
                "gripper_actions": [],
                "progress_outputs": [],
            }

        positions = actions[:, :3]
        rotations_6d = actions[:, 3:9]
        quaternions = transform_utils.rotation_6d_to_quat(torch.from_numpy(rotations_6d)).numpy()
        action_poses = [
            {
                "position": positions[i].tolist(),
                "rotation_6d": rotations_6d[i].tolist(),
                "quaternion": quaternions[i].tolist(),
            }
            for i in range(actions.shape[0])
        ]
        return {
            "action_count": int(actions.shape[0]),
            "action_poses": action_poses,
            "gripper_actions": actions[:, -2].tolist(),
            "progress_outputs": actions[:, -1].tolist(),
        }

    def run_evaluation_experiments(self, experiments: list[dict]):
        """Run evaluation episodes guided by a saved experiment manifest."""
        episodes = len(experiments)
        for i, experiment in enumerate(experiments):
            experiment_index = self._evaluation_experiment_index(experiment, i)
            result = self._run_evaluation_attempt(
                experiment,
                f"episode {i + 1}/{episodes} (experiment id {experiment_index})",
                episode_index=i,
            )
            self._save_attempt_result(result)

    def run_evaluation_until_n_samples(self, experiment: dict, n_samples: int) -> None:
        """Repeat one saved evaluation experiment until n_samples are collected."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}.")

        experiment_index = self._evaluation_experiment_index(experiment, 0)
        completed_attempt_indices = self._completed_attempt_indices(experiment_index)

        while len(completed_attempt_indices) < n_samples:
            attempt_index = self._next_attempt_index(experiment_index)
            sample_number = len(completed_attempt_indices) + 1
            result = self._run_evaluation_attempt(
                experiment,
                (
                    f"experiment id {experiment_index}, sample {sample_number}/{n_samples} "
                    f"(attempt {attempt_index + 1})"
                ),
                episode_index=attempt_index,
                attempt_index=attempt_index,
            )
            self._save_attempt_result(result)
            completed_attempt_indices.add(attempt_index)

    def inference_loop(self):
        """Main inference loop for executing robot policy."""
        self.robot.init_waypoint_motion()

        obs_stream = (
            rx.interval(0.1, scheduler=NewThreadScheduler())
            .pipe(ops.map(lambda _: self.get_observation()))
            .subscribe(lambda x: self.obs_deque.append(x))
        )

        start_time = time.time()
        start_perf = time.perf_counter()

        all_actions = np.zeros((0, self.config.action_shape))

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
            
            action_horizon_len = int(len(action))
            relative = self.config.data.action_relative
            waypoints = self.robot.get_next_waypoints(
                n_trans[0:action_horizon_len],
                n_quads[0:action_horizon_len],
                relative=relative,
            )
            for i in range(int(len(action))):
                if action[i][-2] > self.robot.config.gripper_close_th:
                    self.robot.close_gripper()
                else:
                    self.robot.open_gripper()

                rr.log("/action/gripper", rr.Scalars(action[i, -2].tolist()))
                rr.log("/action/progress", rr.Scalars(action[i, -1].tolist()))


                time.sleep(1 / self.robot.config.action_hz)
                self.robot.motion.set_next_waypoints([waypoints[i]])
                print("progress : ", progress[i])
            
                if progress[i] >= self.robot.config.progress_complete_th:
                    self.robot.stop_motion()
                    obs_stream.dispose()
                    self.record_videos()
                    self.done = True

            elapsed_time = time.perf_counter() - infer_start_time
            rr.log("/debug/inference_time", rr.Scalars(elapsed_time))

            if (time.time() - start_time) > self.timeout:
                print("Timeout reached, ending inference.")
                self.robot.stop_motion()
                obs_stream.dispose()
                self.record_videos()
                self.done = True

        assert self.robot.move_async is not None
        self.robot.move_async.join()
        return {
            "duration_seconds": time.perf_counter() - start_perf,
            **self._serialize_action_outputs(all_actions),
        }

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

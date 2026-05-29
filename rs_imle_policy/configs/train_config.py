from dataclasses import dataclass, field
import pathlib
from typing import Literal, Optional


@dataclass
class CameraConfig:
    """Realsense camera configuration"""

    name: str
    serial_number: str
    exposure: int
    gain: int
    resolution: tuple[int, int] = (640, 480)
    depth_resolution: tuple[int, int] = (640, 480)
    frame_rate: int = 10
    rgb_enabled: bool = True
    depth_enabled: bool = False


@dataclass
class WristCamera(CameraConfig):
    name: str = "wrist"
    serial_number: str = "123622270136"
    exposure: int = 5000
    gain: int = 60


@dataclass
class TopCamera(CameraConfig):
    name: str = "top"
    serial_number: str = "035122250388"
    exposure: int = 100
    gain: int = 60


@dataclass
class SideCamera(CameraConfig):
    name: str = "side"
    serial_number: str = "035122250692"
    exposure: int = 100
    gain: int = 60


@dataclass
class SideCamera2(CameraConfig):
    name: str = "side_2"
    serial_number: str = "036422070913"
    exposure: int = 100
    gain: int = 60


default_cameras = {
    "wrist": WristCamera(),
    "top": TopCamera(),
    "side": SideCamera(),
    "side_2": SideCamera2(),
}


@dataclass
class VisionConfig:
    """Vision feature configuration"""

    vision_features_dim: int = 512
    cameras: tuple[str, ...] = ("wrist", "side", "top")
    img_shape: tuple[int, int] = (240, 320)

    def __post_init__(self):
        self.cameras_params: list[CameraConfig] = [default_cameras[cam] for cam in self.cameras]


@dataclass
class G1VisionConfig(VisionConfig):
    """Vision configuration for G1 dataset"""

    cameras: tuple[str, ...] = ("color_0",)

    def __post_init__(self):
        return


@dataclass
class DataConfig:
    """
    Data / environment configuration.

    Owns:
    - dataset path and task name
    - mapping from logical names to obs/action keys
    - vision / cameras
    """

    dataset_path: pathlib.Path = pathlib.Path(".")
    task_name: str = "default"

    # Low-dimensional observation keys
    lowdim_obs_keys: tuple[str, ...] = (
        "robot_pos",
        "robot_orien",
        "gripper_state",
    )

    # Action keys
    action_keys: tuple[str, ...] = (
        "action_pos",
        "action_orien",
        "action_gripper",
        "progress",
    )

    # Whether actions are relative to current pose
    action_relative: bool = False

    use_next_state: bool = True
    # Whether to use the next_state or the leader position

    # Vision configuration
    vision: VisionConfig = field(default_factory=VisionConfig)

    def get_name(self) -> str:
        string = ""
        string += "ra" if self.action_relative else "aa"
        string += "_next" if self.use_next_state else "_leader"
        string += "_" + "_".join(self.vision.cameras) 
        return string


@dataclass
class G1ArmsDataConfig(DataConfig):
    """Data configuration for G1 arms dataset"""

    lowdim_obs_keys: tuple[str, ...] = (
        "left_robot_pos",
        "left_robot_orien",
        "right_robot_pos",
        "right_robot_orien",
        "left_hand_state",
        "right_hand_state",
    )

    action_keys: tuple[str, ...] = (
        "left_action_pos",
        "left_action_orien",
        "right_action_pos",
        "right_action_orien",
        "left_hand_action",
        "right_hand_action",
        "progress",
    )

    vision: VisionConfig = field(default_factory=G1VisionConfig)


@dataclass
class G1LeftArmDataConfig(DataConfig):
    """Data configuration for G1 left arm dataset"""

    lowdim_obs_keys: tuple[str, ...] = (
        "left_robot_pos",
        "left_robot_orien",
        "left_hand_state",
    )

    action_keys: tuple[str, ...] = (
        "left_action_pos",
        "left_action_orien",
        "left_hand_action",
        "progress",
    )

    vision: VisionConfig = field(default_factory=G1VisionConfig)


@dataclass
class G1RightArmDataConfig(DataConfig):
    """Data configuration for G1 right arm dataset"""

    lowdim_obs_keys: tuple[str, ...] = (
        "right_robot_pos",
        "right_robot_orien",
        "right_hand_state",
    )

    action_keys: tuple[str, ...] = (
        "right_action_pos",
        "right_action_orien",
        "right_hand_action",
        "progress",
    )

    vision: VisionConfig = field(default_factory=G1VisionConfig)


@dataclass
class OptimConfig:
    """Optimization and training parameters"""

    lr: float = 1e-4
    weight_decay: float = 1e-6
    num_epochs: int = 1200
    batch_size: int = 64
    num_workers: int = 11
    lr_scheduler_profile: str = "cosine"
    num_warmup_steps: int = 500
    eval_interval: int = 10
    num_eval_episodes: int = 1
    save_period: int = 10


@dataclass
class BaseModel:
    """Base model with common attributes between methods"""

    device: Literal["cuda", "cpu"] = "cuda"

    pred_horizon: int = 16
    action_horizon: int = 8
    obs_horizon: int = 2


@dataclass
class RSIMLE(BaseModel):
    """RS-IMLE model configuration"""

    name: str = "rs_imle"
    n_samples_per_condition: int = 10
    epsilon: float = 0.1
    traj_consistency: bool = False
    periodic_length: int = 5  # C steps for a new trajectory to be selected eq(6)


@dataclass
class Diffusion(BaseModel):
    """Diffusion model configuration"""

    name: str = "diffusion"
    num_diffusion_iters: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"


@dataclass
class ExperimentConfig:
    """Main training configuration"""

    exp_name: str
    dataset_path: pathlib.Path
    model: Diffusion | RSIMLE
    task_name: str = "default"
    debug: bool = False

    # Sub-configurations
    training_params: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)

    epoch: Optional[int] = None  # For loading checkpoints
    action_shape: int = 0  # Solved during training
    obs_shape: int = 0  # Solved during training

    def process_name(self) -> str:
        name = ""
        name += self.model.name + "_"
        name += self.data.get_name() + "_"
        # name += self.task_name + "_"
        name += self.exp_name
        return name


@dataclass
class LoaderConfig:
    """Configuration for data loading"""

    path: pathlib.Path
    epoch: Optional[int] = None
    timeout: int = 60  # Timeout for experiment in seconds
    episodes: int = 10  # total number of episodes to run
    exp_name: Optional[str] = None
    dry_run: bool = False
    traj_consistency: bool = False # Only valid for RS-IMLE


if __name__ == "__main__":
    import tyro

    args = tyro.cli(ExperimentConfig)
    config = tyro.extras.to_yaml(args)
    with open("example.yml", "w") as outfile:
        outfile.write(config)

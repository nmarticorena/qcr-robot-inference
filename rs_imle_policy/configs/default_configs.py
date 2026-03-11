from rs_imle_policy.configs.train_config import (
    ExperimentConfig,
    RSIMLE,
    Diffusion,
    DataConfig,
    G1ArmsDataConfig,
)
from dataclasses import dataclass, field


@dataclass
class AbsoluteActionsConfig(DataConfig):
    """Configuration for absolute action space"""

    action_keys: tuple[str, ...] = (
        "action_pos",
        "action_orien",
        "action_gripper",
        "progress",
    )
    action_relative: bool = False


@dataclass
class RelativeActionsConfig(DataConfig):
    """Configuration for relative action space"""

    action_keys: tuple[str, ...] = (
        "relative_pos",
        "relative_orien",
        "action_gripper",
        "progress",
    )
    action_relative: bool = True


# RS-IMLE Configurations
@dataclass
class PickPlaceRSMLEConfig(ExperimentConfig):
    """Pick and place task with RS-IMLE using absolute actions"""

    model: RSIMLE | Diffusion = field(default_factory=RSIMLE)
    data: DataConfig = field(default_factory=AbsoluteActionsConfig)


@dataclass
class PickPlaceRSMLERelativeConfig(ExperimentConfig):
    """Pick and place task with RS-IMLE using relative actions"""

    model: RSIMLE | Diffusion = field(default_factory=RSIMLE)
    data: DataConfig = field(default_factory=RelativeActionsConfig)


# Diffusion Configurations
@dataclass
class PickPlaceDiffusionConfig(ExperimentConfig):
    """Pick and place task with Diffusion using absolute actions"""

    model: RSIMLE | Diffusion = field(default_factory=Diffusion)
    data: DataConfig = field(default_factory=AbsoluteActionsConfig)


@dataclass
class PickPlaceDiffusionRelativeConfig(ExperimentConfig):
    """Pick and place task with Diffusion using relative actions"""

    model: RSIMLE | Diffusion = field(default_factory=Diffusion)
    data: DataConfig = field(default_factory=RelativeActionsConfig)


@dataclass
class G1ArmsRSIMLEConfig(ExperimentConfig):
    """G1 arms dataset with RS-IMLE"""

    model: RSIMLE | Diffusion = field(default_factory=RSIMLE)
    data: DataConfig = field(default_factory=G1ArmsDataConfig)


@dataclass
class G1ArmsDiffusionConfig(ExperimentConfig):
    """G1 arms dataset with Diffusion"""

    model: RSIMLE | Diffusion = field(default_factory=Diffusion)
    data: DataConfig = field(default_factory=G1ArmsDataConfig)


ExperimentConfigChoice = (
    PickPlaceRSMLEConfig
    | PickPlaceRSMLERelativeConfig
    | PickPlaceDiffusionConfig
    | PickPlaceDiffusionRelativeConfig
    | G1ArmsRSIMLEConfig
    | G1ArmsDiffusionConfig
)

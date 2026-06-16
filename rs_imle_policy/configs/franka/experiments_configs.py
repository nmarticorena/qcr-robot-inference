from rs_imle_policy.configs.train_config import (
    ExperimentConfig,
    RSIMLE,
    Diffusion,
    FlowMatching,
    DataConfig,
)
from dataclasses import dataclass, field
from typing import Literal


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
    action_mode: Literal["absolute"] = "absolute"


@dataclass
class DeltaActionsConfig(DataConfig):
    """Configuration for consecutive-step delta action space"""

    action_keys: tuple[str, ...] = (
        "delta_pos",
        "delta_orien",
        "action_gripper",
        "progress",
    )
    action_relative: bool = True
    action_mode: Literal["delta"] = "delta"


@dataclass
class RelativeActionsConfig(DataConfig):
    """Configuration for anchor-relative action space"""

    action_keys: tuple[str, ...] = (
        "relative_pos",
        "relative_orien",
        "action_gripper",
        "progress",
    )
    action_relative: bool = True
    action_mode: Literal["relative"] = "relative"

# RS-IMLE Configurations
@dataclass
class PickPlaceRSMLEConfig(ExperimentConfig):
    """Pick and place task with RS-IMLE using absolute actions"""

    model: RSIMLE = field(default_factory=RSIMLE)
    data: DataConfig = field(default_factory=AbsoluteActionsConfig)


@dataclass
class PickPlaceRSIMLEDeltaConfig(ExperimentConfig):
    """Pick and place task with RS-IMLE using delta actions"""

    model: RSIMLE = field(default_factory=RSIMLE)
    data: DataConfig = field(default_factory=DeltaActionsConfig)


@dataclass
class PickPlaceRSMLERelativeConfig(ExperimentConfig):
    """Pick and place task with RS-IMLE using relative actions"""

    model: RSIMLE = field(default_factory=RSIMLE)
    data: DataConfig = field(default_factory=RelativeActionsConfig)


# Diffusion Configurations
@dataclass
class PickPlaceDiffusionConfig(ExperimentConfig):
    """Pick and place task with Diffusion using absolute actions"""

    model: Diffusion = field(default_factory=Diffusion)
    data: DataConfig = field(default_factory=AbsoluteActionsConfig)


@dataclass
class PickPlaceDiffusionDeltaConfig(ExperimentConfig):
    """Pick and place task with Diffusion using delta actions"""

    model: Diffusion = field(default_factory=Diffusion)
    data: DataConfig = field(default_factory=DeltaActionsConfig)


@dataclass
class PickPlaceDiffusionRelativeConfig(ExperimentConfig):
    """Pick and place task with Diffusion using relative actions"""

    model: Diffusion = field(default_factory=Diffusion)
    data: DataConfig = field(default_factory=RelativeActionsConfig)

@dataclass
class PickPlaceFlowMatchingConfig(ExperimentConfig):
    """Pick and place task with Diffusion using absolute actions"""

    model: FlowMatching = field(default_factory=FlowMatching)
    data: DataConfig = field(default_factory=AbsoluteActionsConfig)


@dataclass
class PickPlaceFlowMatchingDeltaConfig(ExperimentConfig):
    """Pick and place task with Flow Matching using delta actions"""

    model: FlowMatching = field(default_factory=FlowMatching)
    data: DataConfig = field(default_factory=DeltaActionsConfig)


@dataclass
class PickPlaceFlowMatchingRelativeConfig(ExperimentConfig):
    """Pick and place task with Diffusion using relative actions"""

    model: FlowMatching = field(default_factory=FlowMatching)
    data: DataConfig = field(default_factory=RelativeActionsConfig)

ExperimentConfigChoice = (
    PickPlaceRSMLEConfig
    | PickPlaceRSIMLEDeltaConfig
    | PickPlaceRSMLERelativeConfig
    | PickPlaceDiffusionConfig
    | PickPlaceDiffusionDeltaConfig
    | PickPlaceDiffusionRelativeConfig
    | PickPlaceFlowMatchingConfig
    | PickPlaceFlowMatchingDeltaConfig
    | PickPlaceFlowMatchingRelativeConfig
)

from rs_imle_policy.configs.train_config import (
    ExperimentConfig,
    RSIMLE,
    Diffusion,
    DataConfig,
    G1ArmsDataConfig,
    G1LeftArmDataConfig,
    G1RightArmDataConfig,
)
from dataclasses import dataclass, field


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


@dataclass
class G1LeftArmRSIMLEConfig(ExperimentConfig):
    """G1 left arm dataset with RS-IMLE"""

    model: RSIMLE | Diffusion = field(default_factory=RSIMLE)
    data: DataConfig = field(default_factory=G1LeftArmDataConfig)


@dataclass
class G1LeftArmDiffusionConfig(ExperimentConfig):
    """G1 left arm dataset with Diffusion"""

    model: RSIMLE | Diffusion = field(default_factory=Diffusion)
    data: DataConfig = field(default_factory=G1LeftArmDataConfig)


@dataclass
class G1RightArmRSIMLEConfig(ExperimentConfig):
    """G1 right arm dataset with RS-IMLE"""

    model: RSIMLE | Diffusion = field(default_factory=RSIMLE)
    data: DataConfig = field(default_factory=G1RightArmDataConfig)


@dataclass
class G1RightArmDiffusionConfig(ExperimentConfig):
    """G1 right arm dataset with Diffusion"""

    model: RSIMLE | Diffusion = field(default_factory=Diffusion)
    data: DataConfig = field(default_factory=G1RightArmDataConfig)


ExperimentConfigChoice = (
    G1ArmsRSIMLEConfig
    | G1ArmsDiffusionConfig
    | G1LeftArmRSIMLEConfig
    | G1LeftArmDiffusionConfig
    | G1RightArmRSIMLEConfig
    | G1RightArmDiffusionConfig
)

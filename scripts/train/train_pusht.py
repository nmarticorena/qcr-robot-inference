from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, TypeAlias

import tyro

from rs_imle_policy.configs.train_config import (
    Diffusion,
    ExperimentConfig,
    FlowMatching,
    OptimConfig,
    PushTDataConfig,
    RSIMLE,
)
from rs_imle_policy.datasets import PushTDataset
from rs_imle_policy.envs.push_t import PushTImageEnv
from rs_imle_policy.train import run_training


@dataclass
class PushTRSIMLEConfig(ExperimentConfig):
    """PushT dataset with RS-IMLE"""

    task_name: str = "pusht"
    model: RSIMLE = field(default_factory=lambda: RSIMLE(action_horizon=2))
    data: PushTDataConfig = field(default_factory=PushTDataConfig)
    training_params: OptimConfig = field(
        default_factory=lambda: OptimConfig(eval_interval=1000, num_eval_episodes=10)
    )


@dataclass
class PushTDiffusionConfig(ExperimentConfig):
    """PushT dataset with Diffusion"""

    task_name: str = "pusht"
    model: Diffusion = field(default_factory=lambda: Diffusion(action_horizon=2))
    data: PushTDataConfig = field(default_factory=PushTDataConfig)
    training_params: OptimConfig = field(
        default_factory=lambda: OptimConfig(eval_interval=1000, num_eval_episodes=10)
    )


@dataclass
class PushTFlowMatchingConfig(ExperimentConfig):
    """PushT dataset with Flow Matching"""

    task_name: str = "pusht"
    model: FlowMatching = field(default_factory=lambda: FlowMatching(action_horizon=2))
    data: PushTDataConfig = field(default_factory=PushTDataConfig)
    training_params: OptimConfig = field(
        default_factory=lambda: OptimConfig(eval_interval=1000, num_eval_episodes=10)
    )


PushTExperimentConfigChoice: TypeAlias = (
    Annotated[PushTRSIMLEConfig, tyro.conf.subcommand(name="rsimle")]
    | Annotated[PushTDiffusionConfig, tyro.conf.subcommand(name="diffusion")]
    | Annotated[PushTFlowMatchingConfig, tyro.conf.subcommand(name="flow-matching")]
)


def build_dataset(config: ExperimentConfig) -> PushTDataset:
    return PushTDataset(
        config.dataset_path,
        pred_horizon=config.model.pred_horizon,
        obs_horizon=config.model.obs_horizon,
        action_horizon=config.model.action_horizon,
        low_dim_obs_keys=config.data.lowdim_obs_keys,
        action_keys=config.data.action_keys,
        action_mode=config.data.action_mode,
    )


def main(config: PushTExperimentConfigChoice) -> None:
    dataset = build_dataset(config)
    env = PushTImageEnv(render_size=config.data.vision.img_shape[0])
    run_training(config, dataset, env=env)


if __name__ == "__main__":
    main(tyro.cli(PushTExperimentConfigChoice))

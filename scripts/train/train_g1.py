from __future__ import annotations

from typing import Annotated, TypeAlias

import tyro

from rs_imle_policy.configs.g1.experiments_configs import (
    G1ArmsDiffusionConfig,
    G1ArmsRSIMLEConfig,
    G1LeftArmDiffusionConfig,
    G1LeftArmRSIMLEConfig,
    G1RightArmDiffusionConfig,
    G1RightArmRSIMLEConfig,
)
from rs_imle_policy.configs.train_config import ExperimentConfig
from rs_imle_policy.datasets import G1ArmsDataset
from rs_imle_policy.train import run_training

G1ExperimentConfigChoice: TypeAlias = (
    Annotated[G1ArmsRSIMLEConfig, tyro.conf.subcommand(name="g1-arms-rsimle")]
    | Annotated[G1ArmsDiffusionConfig, tyro.conf.subcommand(name="g1-arms-diffusion")]
    | Annotated[G1LeftArmRSIMLEConfig, tyro.conf.subcommand(name="g1-left-arm-rsimle")]
    | Annotated[G1LeftArmDiffusionConfig, tyro.conf.subcommand(name="g1-left-arm-diffusion")]
    | Annotated[G1RightArmRSIMLEConfig, tyro.conf.subcommand(name="g1-right-arm-rsimle")]
    | Annotated[G1RightArmDiffusionConfig, tyro.conf.subcommand(name="g1-right-arm-diffusion")]
)


def build_dataset(config: ExperimentConfig) -> G1ArmsDataset:
    return G1ArmsDataset(
        config.dataset_path,
        pred_horizon=config.model.pred_horizon,
        obs_horizon=config.model.obs_horizon,
        action_horizon=config.model.action_horizon,
        low_dim_obs_keys=config.data.lowdim_obs_keys,
        action_keys=config.data.action_keys,
        vision_config=config.data.vision,
        use_next_state=config.data.use_next_state,
        action_mode=config.data.action_mode,
    )


def main(config: G1ExperimentConfigChoice) -> None:
    dataset = build_dataset(config)
    run_training(config, dataset)


if __name__ == "__main__":
    main(tyro.cli(G1ExperimentConfigChoice))

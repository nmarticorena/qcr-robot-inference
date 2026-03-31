from __future__ import annotations

from typing import Annotated, TypeAlias

import torch
import tyro
import wandb

from rs_imle_policy.configs.default_configs import (
    G1ArmsDiffusionConfig,
    G1ArmsRSIMLEConfig,
    G1LeftArmDiffusionConfig,
    G1LeftArmRSIMLEConfig,
    G1RightArmDiffusionConfig,
    G1RightArmRSIMLEConfig,
)
from rs_imle_policy.configs.train_config import ExperimentConfig
from rs_imle_policy.datasets import G1ArmsDataset
from rs_imle_policy.policy import Policy
from rs_imle_policy.train_real_world import train

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
    )


def build_dataloader(config: ExperimentConfig, dataset: G1ArmsDataset) -> torch.utils.data.DataLoader:
    num_workers = 0 if config.debug else config.training_params.num_workers
    persistent_workers = num_workers > 0

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training_params.batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )


def main(config: G1ExperimentConfigChoice) -> None:
    wandb.init(project=config.task_name)
    wandb.run.name = f"{config.exp_name}_{config.model.name}"

    dataset = build_dataset(config)
    dataloader = build_dataloader(config, dataset)

    policy = Policy(config=config, dataset=dataset)
    train(
        config,
        policy.nets,
        dataloader,
        policy.noise_scheduler,
        policy.optimizer,
        policy.lr_scheduler,
        policy.ema,
    )

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(G1ExperimentConfigChoice))

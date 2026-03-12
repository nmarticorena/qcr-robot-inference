import tyro
import torch
import wandb
from rs_imle_policy.datasets import G1ArmsDataset
from rs_imle_policy.configs.default_configs import G1ArmsRSIMLEConfig
from rs_imle_policy.configs.train_config import G1VisionConfig
from pathlib import Path
from rs_imle_policy.policy import Policy
from rs_imle_policy.train_real_world import train


def main(dataset_path: str, task_name: str, exp_name: str, debug: bool = False):
    wandb.init(project=task_name)

    # change wandb name
    wandb.run.name = f"{exp_name}"
    dataset = G1ArmsDataset(
        Path(dataset_path),
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        low_dim_obs_keys=[
            "left_robot_pos",
            "left_robot_orien",
            "right_robot_pos",
            "right_robot_orien",
        ],
        action_keys=[
            "left_action_pos",
            "left_action_orien",
            "right_action_pos",
            "right_action_orien",
            "progress",
        ],
        vision_config=G1VisionConfig(),
        use_next_state=True,
    )

    if debug:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            persistent_workers=False,
        )
        print("debug")
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=11,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )
        print("no debug")

    config = G1ArmsRSIMLEConfig(
        exp_name=exp_name,
        dataset_path=Path(dataset_path),
        task_name=task_name,
        debug=debug,
    )
    policy = Policy(config=config, dataset=dataset)
    nets = policy.nets

    train(
        config,
        nets,
        dataloader,
        policy.noise_scheduler,
        policy.optimizer,
        policy.lr_scheduler,
        policy.ema,
    )

    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)

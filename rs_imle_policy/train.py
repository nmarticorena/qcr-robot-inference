from rs_imle_policy.datasets import BaseDataset
import torch.nn as nn
import copy
import os
import shutil
import time
from collections import deque
from dataclasses import asdict
from typing import TypeVar, Optional

import numpy as np
import torch
import tyro
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from rs_imle_policy.policy import Policy
from rs_imle_policy.loss import rs_imle_loss
from rs_imle_policy.configs.train_config import ExperimentConfig, Diffusion, RSIMLE, FlowMatching
from rs_imle_policy.datasets.base_dataset import normalize_data, unnormalize_data
from rs_imle_policy.vision_encoders.resnet import keypoint_spread_metrics


DatasetT = TypeVar("DatasetT", bound=Dataset)


def process_image(images, vision_encoder, device):
    B, T, C, H, W = images.shape
    images = images.flatten(end_dim=1).to(device)
    image_features = vision_encoder(images)
    image_features = image_features.reshape(B, T, -1)
    return image_features


def log_keypoint_metrics(nets, cams_names, train_step: int):
    metrics = {}
    for cam in cams_names:
        encoder = nets[f"vision_encoder_{cam}"]
        avgpool = getattr(encoder, "avgpool", None)
        kps = getattr(avgpool, "kps", None)
        if kps is None:
            continue

        cam_metrics = keypoint_spread_metrics(kps)
        metrics.update({f"{cam}/{name}": value for name, value in cam_metrics.items()})

    if metrics:
        wandb.log(
            {name: value.item() for name, value in metrics.items()},
            step=train_step,
        )


@torch.no_grad()
def sample_normalized_actions(
    args: ExperimentConfig,
    nets,
    noise_scheduler,
    obs_cond: torch.Tensor,
    batch_size: int = 1,
) -> torch.Tensor:
    device = args.model.device
    action_shape = args.action_shape
    sample_shape = (batch_size, args.model.pred_horizon, action_shape)

    if isinstance(args.model, Diffusion):
        if noise_scheduler is None:
            raise ValueError("Diffusion evaluation requires a noise scheduler.")
        naction = torch.randn(sample_shape, device=device)
        noise_scheduler.set_timesteps(args.model.num_diffusion_iters)

        for timestep in noise_scheduler.timesteps:
            noise_pred = nets["noise_pred_net"](
                sample=naction,
                timestep=timestep,
                global_cond=obs_cond,
            )
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=int(timestep),
                sample=naction,
            ).prev_sample
        return naction

    if isinstance(args.model, RSIMLE):
        noise = torch.clamp(torch.randn(sample_shape, device=device), -1, 1)
        return nets["generator"](noise, global_cond=obs_cond)

    if isinstance(args.model, FlowMatching):
        naction = torch.randn(sample_shape, device=device)
        ts = torch.linspace(0.0, 1.0, args.model.num_flow_iters + 1, device=device)[:-1]
        dt = 1.0 / args.model.num_flow_iters
        for t in ts:
            timestep = (t * args.model.timestep_integer_scaler).long()
            pred = nets["noise_pred_net"](
                sample=naction,
                timestep=timestep,
                global_cond=obs_cond,
            )
            naction = naction + pred * dt
        return naction

    raise NotImplementedError(f"Model type {type(args.model)} is not supported for PushT evaluation.")


def pusht_obs_cond(args: ExperimentConfig, nets, obs_history, stats: dict) -> torch.Tensor:
    device = args.model.device
    cams_names = args.data.vision.cameras
    image_features = []

    for cam in cams_names:
        if cam != "frames":
            raise ValueError(f"PushT evaluation expects camera name 'frames', got {cam!r}.")
        images = np.stack([obs["image"] for obs in obs_history], axis=0)
        images = torch.as_tensor(images, dtype=torch.float32, device=device).unsqueeze(0)
        image_features.append(process_image(images, nets[f"vision_encoder_{cam}"], device))

    agent_pos = np.stack([obs["agent_pos"] for obs in obs_history], axis=0)
    nagent = normalize_data(agent_pos, stats["state"])
    nagent = torch.as_tensor(nagent, dtype=torch.float32, device=device).unsqueeze(0)
    obs_features = torch.cat([*image_features, nagent], dim=-1)
    return obs_features.flatten(start_dim=1)

@torch.no_grad()
def compute_val_rollout_pos_error(
    args: ExperimentConfig,
    nets,
    noise_scheduler,
    batch:dict,
    dataset: BaseDataset,
    *,
    train_step: int | None = None,
) -> torch.Tensor:
    device = args.model.device
    obs_horizon = args.model.obs_horizon
    cams_names = args.data.vision.cameras

    nagent = batch["state"][:, :obs_horizon].to(device)
    naction = batch["action"].to(device)

    images = [batch[f"frame_{cam}"][:, :obs_horizon].to(device) for cam in cams_names]
    image_features = [
        process_image(img, nets[f"vision_encoder_{cam}"], device) for img, cam in zip(images, cams_names)
    ]

    obs_features = torch.cat([*image_features, nagent], dim=-1)
    obs_cond = obs_features.flatten(start_dim=1)
    noise = torch.randn(naction.shape, device=device)

    if isinstance(args.model, Diffusion):
        noise_actions = noise
        for k in noise_scheduler.timesteps:
            noise_pred = nets["noise_pred_net"](sample = noise_actions, timestep=k, global_cond=obs_cond)
            noise_actions = noise_scheduler.step(model_output = noise_pred, timestep = int(k), sample = noise_actions).prev_sample
    elif isinstance(args.model, RSIMLE):
        noise_actions = nets["generator"](noise, global_cond = obs_cond)

    elif isinstance(args.model, FlowMatching):
        noise_actions = noise
        ts = torch.linspace(0.0, 1.0, args.model.num_flow_iters+1, device = args.model.device)[:-1]
        dt = 1.0 / args.model.num_flow_iters
        for t in ts:
            timestep = (t * args.model.timestep_integer_scaler).long()

            # predict noise
            pred = nets['noise_pred_net'](
                sample=noise_actions,
                timestep=timestep,
                global_cond=obs_cond
            )
            noise_actions = noise_actions + pred * dt


    pred_actions = noise_actions.detach().to("cpu").numpy() # [batch_size, action_horizon, action_dim]
    pred_robot_actions = dataset.n_action_to_robot_action(pred_actions, batch["state"])
    pred_robot_pos  = torch.from_numpy(pred_robot_actions["pos"]) # [batch_size, action_horizon, 3]
    gt_pos = batch["gt"][:,:,:3,-1]

    per_timestep_l2 =torch.linalg.norm(pred_robot_pos - gt_pos, dim = -1)   # [batch_size, action_horizon]

    return per_timestep_l2



def compute_batch_loss(
    args: ExperimentConfig,
    nets,
    noise_scheduler,
    batch: dict,
    *,
    train_step: int | None = None,
    log_keypoint_metrics_enabled: bool = False,
) -> torch.Tensor:
    device = args.model.device
    obs_horizon = args.model.obs_horizon
    cams_names = args.data.vision.cameras

    nagent = batch["state"][:, :obs_horizon].to(device)
    naction = batch["action"].to(device)
    batch_size = naction.shape[0]

    images = [batch[f"frame_{cam}"][:, :obs_horizon].to(device) for cam in cams_names]
    image_features = [
        process_image(img, nets[f"vision_encoder_{cam}"], device) for img, cam in zip(images, cams_names)
    ]
    if log_keypoint_metrics_enabled and train_step is not None:
        log_keypoint_metrics(nets, cams_names, train_step)

    obs_features = torch.cat([*image_features, nagent], dim=-1)
    obs_cond = obs_features.flatten(start_dim=1)

    if isinstance(args.model, Diffusion):
        noise = torch.randn(naction.shape, device=device)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

        noise_pred = nets["noise_pred_net"](noisy_actions, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(noise_pred, noise)

    if isinstance(args.model, RSIMLE):
        noise = torch.randn(
            batch_size * args.model.n_samples_per_condition,
            *naction.shape[1:],
            device=device,
        )
        repeated_obs_cond = obs_cond.repeat_interleave(args.model.n_samples_per_condition, dim=0)

        fake_actions = nets["generator"](noise, global_cond=repeated_obs_cond)
        fake_actions = fake_actions.reshape(batch_size, args.model.n_samples_per_condition, *naction.shape[1:])

        return rs_imle_loss(
            naction,
            fake_actions,
            args.model.epsilon,
            train_step=train_step,
        )

    if isinstance(args.model, FlowMatching):
        noise = torch.randn(naction.shape, device=device)
        t = torch.rand(batch_size, device=device)
        t_shaped = t.reshape(-1, *([1] * (noise.dim() - 1)))
        xt = t_shaped * naction + (1 - t_shaped) * noise
        vector = naction - noise
        timesteps = (t * args.model.timestep_integer_scaler).long()
        pred = nets["noise_pred_net"](xt, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(pred, vector)

    raise NotImplementedError


def rollout_error_stats(errors: torch.Tensor) -> dict[str, torch.Tensor]:
    n = errors.shape[0]
    mean = errors.mean(dim=0)
    std = errors.std(dim=0, unbiased=n > 1)
    sem = std / n**0.5
    ci95 = 1.96 * sem

    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - ci95,
        "ci_high": mean + ci95,
        "min": errors.min(dim=0).values,
        "max": errors.max(dim=0).values,
    }


def make_rollout_error_plot(errors: torch.Tensor, stats: dict[str, torch.Tensor]):
    errors = errors.detach().cpu()
    stats = {key: value.detach().cpu() for key, value in stats.items()}
    steps = torch.arange(errors.shape[1]).numpy()

    fig, ax = plt.subplots()

    for curve in errors:
        ax.plot(steps, curve.numpy(), alpha=0.25, linewidth=0.8)

    ax.plot(steps, stats["mean"].numpy(), linewidth=2, label="mean")
    ax.fill_between(
        steps,
        stats["ci_low"].numpy(),
        stats["ci_high"].numpy(),
        alpha=0.25,
        label="95% CI",
    )

    ax.set_xlabel("horizon step")
    ax.set_ylabel("position L2 error [m]")
    ax.set_title("Validation rollout position error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


@torch.no_grad()
def validate(
    args: ExperimentConfig,
    nets,
    val_dataloader,
    noise_scheduler,
    epoch_step: int | None,
) -> float:
    was_training = nets.training
    nets.eval()
    errors = []

    for batch in tqdm(val_dataloader, desc="Validation", leave=False, unit="batch"):
        error = compute_val_rollout_pos_error(args, nets, noise_scheduler, batch, val_dataloader.dataset)
        errors.append(error.detach().cpu())
    if was_training:
        nets.train()

    if not errors:
        return float("nan")
    errors = torch.concat(errors, dim=0) # [n_steps, action_horizon]

    stats = rollout_error_stats(errors)

    max_error = stats["max"][-1].item()
    mean_error = stats["mean"].mean()
    final_step_error = stats["mean"][-1]
    max_step_error, max_step_idx = stats["mean"].max(dim=0)

    if epoch_step is not None:
        log_data = {
            "val/rollout_pos_l2_mean": mean_error.item(),
            "val/rollout_pos_l2_final": final_step_error.item(),
            "val/rollout_pos_l2_max": max_step_error.item(),
            "val/rollout_pos_l2_argmax": max_step_idx.item(),
        }

        for i, error in enumerate(stats["mean"]):
            log_data[f"val/rollout_pos_l2_step_{i:02d}"] = error.item()

        fig = make_rollout_error_plot(errors, stats)
        log_data["val/rollout_pos_l2_curve_ci"] = wandb.Image(fig)
        plt.close(fig)

        wandb.log(log_data, step=epoch_step)

    return max_error


def train(
    args: ExperimentConfig,
    nets,
    dataloader,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    ema,
    stats: dict,
    val_dataloader=None,
    env: Optional = None,
):
    nets.train()

    folder = os.path.join("saved_weights", args.task_name, args.process_name())
    os.makedirs(folder, exist_ok=True)

    config = tyro.extras.to_yaml(args)
    with open(os.path.join(folder, "config.yaml"), "w") as f:
        f.write(config)
    shutil.copyfile(args.dataset_path / "stats.pkl", os.path.join(folder, "stats.pkl"))

    # make dir if not exist
    n_epochs = args.training_params.num_epochs

    train_step = 0
    keypoint_metrics_log_interval = args.training_params.keypoint_metrics_log_interval
    log_keypoint_metrics_enabled = (
        keypoint_metrics_log_interval > 0
        and args.model.vision_model is not None
        and args.model.vision_model.use_spatial_softmax
    )

    for epoch in range(n_epochs+1):
        epoch_loss = []
        start_time = time.time()
        with tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{n_epochs + 1}",
            leave=False,
            unit="batch",
        ) as tepoch:
            for batch in tepoch:
                loss = compute_batch_loss(
                    args,
                    nets,
                    noise_scheduler,
                    batch,
                    train_step=train_step,
                    log_keypoint_metrics_enabled=(
                        log_keypoint_metrics_enabled and train_step % keypoint_metrics_log_interval == 0
                    ),
                )

                # If loss is 0, skip backprop and log flag in wandb
                if loss == 0:
                    wandb.log({"zero_loss": 1}, step=train_step)
                else:
                    loss.backward()
                    if args.model.use_clamping:
                        torch.nn.utils.clip_grad_norm_(nets.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(nets.parameters())
                    wandb.log({"zero_loss": 0}, step=train_step)

                wandb.log({"loss": loss.item()}, step=train_step)

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
                train_step += 1

                eval_interval = args.training_params.eval_interval
                if env is not None and eval_interval > 0 and train_step % eval_interval == 0:
                    evaluate_pusht_policy(
                        args,
                        nets,
                        noise_scheduler,
                        ema,
                        env,
                        stats,
                        train_step,
                    )

        ema_nets = copy.deepcopy(nets)
        ema.copy_to(ema_nets.parameters())

        # save a checkpoint every 50 epochs
        if (epoch) % args.training_params.save_period == 0:
            torch.save(nets.state_dict(), f"{folder}/net_epoch_{epoch:04d}.pth")
            shutil.copy(f"{folder}/net_epoch_{epoch:04d}.pth", f"{folder}/net_epoch_last.pth")
            torch.save(ema_nets.state_dict(), f"{folder}/ema_net_epoch_{epoch:04d}.pth")
            shutil.copy(
                f"{folder}/ema_net_epoch_{epoch:04d}.pth",
                f"{folder}/ema_net_epoch_last.pth",
            )

        avg_loss = np.mean(epoch_loss)
        log_data = {"avg_train_loss": avg_loss}
        val_loss = None
        if val_dataloader is not None:
            val_loss = validate(args, nets, val_dataloader, noise_scheduler, epoch_step=epoch)
            log_data["avg_val_loss"] = val_loss

        wandb.log(log_data, step=epoch)
        val_msg = "" if val_loss is None else f" - Avg. Val Loss: {val_loss:.4f}"
        print(
            f"Epoch {epoch + 1}/{n_epochs + 1} - Avg. Loss: {avg_loss:.4f}"
            f"{val_msg} - Time: {time.time() - start_time:.2f}s"
        )
        # If the loss is 0 for a whole epoch, log flag in wandb
        if avg_loss == 0:
            wandb.log({"zero_loss_epoch": 1}, step=train_step)
        else:
            wandb.log({"zero_loss_epoch": 0}, step=train_step)
    return


def build_dataloader(
    config: ExperimentConfig,
    dataset: DatasetT,
    *,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    num_workers = 0 if config.debug else config.training_params.num_workers
    persistent_workers = num_workers > 0

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training_params.batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )


def run_training(
    config: ExperimentConfig,
    dataset: Dataset,
    val_dataset: Dataset | None = None,
    env: Optional = None,
) -> None:
    exp_name = config.process_name()
    wandb.init(project=config.task_name, config=asdict(config), name=exp_name)

    try:
        dataloader = build_dataloader(config, dataset, shuffle=True)
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = build_dataloader(config, val_dataset, shuffle=False)

        policy = Policy(config=config, training=True, dataset=dataset)
        train(
            config,
            policy.nets,
            dataloader,
            policy.noise_scheduler,
            policy.optimizer,
            policy.lr_scheduler,
            policy.ema,
            dataset.stats,
            val_dataloader=val_dataloader,
            env=env,
        )
    finally:
        if env is not None:
            env.close()
        wandb.finish()


def main():
    from rs_imle_policy.configs.experiment_configs import FrankaExperimentConfigChoice

    config = tyro.cli(FrankaExperimentConfigChoice)
    train_dataset, val_dataset = build_franka_datasets(config)
    run_training(config, train_dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    main()

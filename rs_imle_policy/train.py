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


def evaluate_pusht_policy(
    args: ExperimentConfig,
    nets,
    noise_scheduler,
    ema,
    env,
    stats: dict,
    train_step: int,
    max_steps: int = 200,
) -> None:
    if env is None:
        return

    was_training = nets.training
    ema_nets = copy.deepcopy(nets)
    ema.copy_to(ema_nets.parameters())
    ema_nets.eval()

    final_rewards = []
    max_rewards = []
    successes = []
    video_frames = []

    try:
        for episode_idx in range(args.training_params.num_eval_episodes):
            env.seed(episode_idx)
            obs, _ = env.reset()
            obs_history = deque([obs] * args.model.obs_horizon, maxlen=args.model.obs_horizon)
            action_queue = deque()
            reward = 0.0
            episode_max_reward = 0.0
            success = False
            record_video = episode_idx == 0

            for _ in range(max_steps):
                if not action_queue:
                    obs_cond = pusht_obs_cond(args, ema_nets, obs_history, stats)
                    naction = sample_normalized_actions(args, ema_nets, noise_scheduler, obs_cond)
                    action_seq = unnormalize_data(naction[0].detach().cpu().numpy(), stats["action"])
                    start = args.model.obs_horizon - 1
                    end = start + args.model.action_horizon
                    action_queue.extend(action_seq[start:end])

                action = np.asarray(action_queue.popleft())
                obs, reward, terminated, truncated, _ = env.step(action)
                obs_history.append(obs)
                if record_video:
                    video_frames.append(env.render("rgb_array"))
                episode_max_reward = max(episode_max_reward, float(reward))
                if terminated or truncated:
                    success = bool(terminated)
                    break

            final_rewards.append(float(reward))
            max_rewards.append(episode_max_reward)
            successes.append(float(success))

        log_data = {
            "eval/pusht_final_reward": float(np.mean(final_rewards)),
            "eval/pusht_max_reward": float(np.mean(max_rewards)),
            "eval/pusht_success_rate": float(np.mean(successes)),
        }
        if video_frames:
            video = np.moveaxis(np.stack(video_frames).astype(np.uint8), -1, 1)
            fps = env.metadata.get("video.frames_per_second", 10)
            log_data["eval/pusht_video"] = wandb.Video(video, fps=fps, format="mp4")

        wandb.log(
            log_data,
            step=train_step,
        )
    finally:
        if was_training:
            nets.train()


def train(
    args: ExperimentConfig,
    nets,
    dataloader,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    ema,
    stats: dict,
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
    device = args.model.device
    obs_horizon = args.model.obs_horizon

    cams_names = args.data.vision.cameras
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
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False) as tepoch:
            for batch in tepoch:
                nagent = batch["state"][:, :obs_horizon].to(device)
                naction = batch["action"].to(device)
                B = naction.shape[0]

                images = [batch[f"frame_{cam}"][:, :obs_horizon].to(device) for cam in cams_names]
                image_features = [
                    process_image(img, nets[f"vision_encoder_{cam}"], device) for img, cam in zip(images, cams_names)
                ]
                if log_keypoint_metrics_enabled and train_step % keypoint_metrics_log_interval == 0:
                    log_keypoint_metrics(nets, cams_names, train_step)

                obs_features = torch.cat([*image_features, nagent], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)

                if isinstance(args.model, Diffusion):
                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                    ).long()
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = nets["noise_pred_net"](noisy_actions, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(noise_pred, noise)
                elif isinstance(args.model, RSIMLE):
                    noise = torch.randn(
                        B * args.model.n_samples_per_condition,
                        *naction.shape[1:],
                        device=device,
                    )
                    repeated_obs_cond = obs_cond.repeat_interleave(args.model.n_samples_per_condition, dim=0)

                    fake_actions = nets["generator"](noise, global_cond=repeated_obs_cond)
                    fake_actions = fake_actions.reshape(B, args.model.n_samples_per_condition, *naction.shape[1:])

                    loss = rs_imle_loss(naction, fake_actions, args.model.epsilon, train_step=train_step)
                elif isinstance(args.model, FlowMatching):
                    noise = torch.randn(naction.shape, device=device)
                    t = torch.rand(B, device=device)
                    t_shaped = t.reshape(-1, *([1] * (noise.dim() - 1)))
                    xt = t_shaped * naction + (1 - t_shaped) * noise
                    vector = naction - noise
                    timesteps = (t * args.model.timestep_integer_scaler).long()
                    pred = nets["noise_pred_net"](
                        xt, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(pred, vector)

                else:
                    raise NotImplementedError

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
        wandb.log({"avg_train_loss": avg_loss, "epoch": epoch}, step=train_step)
        print(f"Epoch {epoch + 1}/{n_epochs} - Avg. Loss: {avg_loss:.4f} - Time: {time.time() - start_time:.2f}s")
        # If the loss is 0 for a whole epoch, log flag in wandb
        if avg_loss == 0:
            wandb.log({"zero_loss_epoch": 1}, step=train_step)
        else:
            wandb.log({"zero_loss_epoch": 0}, step=train_step)
    return


def build_dataloader(config: ExperimentConfig, dataset: DatasetT) -> torch.utils.data.DataLoader:
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


def build_franka_dataset(config: ExperimentConfig):
    from rs_imle_policy.datasets.single_franka import PandaPolicyDataset

    return PandaPolicyDataset(
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


def run_training(config: ExperimentConfig, dataset: Dataset, env: Optional=None) -> None:
    exp_name = config.process_name()
    wandb.init(project=config.task_name, config=asdict(config), name=exp_name)

    try:
        dataloader = build_dataloader(config, dataset)
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
            env=env,
        )
    finally:
        if env is not None:
            env.close()
        wandb.finish()


def main():
    from rs_imle_policy.configs.experiment_configs import FrankaExperimentConfigChoice

    config = tyro.cli(FrankaExperimentConfigChoice)
    run_training(config, build_franka_dataset(config))


if __name__ == "__main__":
    main()

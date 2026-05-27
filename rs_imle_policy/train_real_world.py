import torch
import tyro
import shutil
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import numpy as np
import copy
import time
from rs_imle_policy.policy import Policy
import os

from rs_imle_policy.configs.train_config import ExperimentConfig, Diffusion, RSIMLE


def rs_imle_loss(real_samples, fake_samples, epsilon=0.1):
    B, T, D = real_samples.shape
    n_samples = fake_samples.shape[1]

    real_flat = real_samples.reshape(B, 1, -1)
    fake_flat = fake_samples.reshape(B, n_samples, -1)

    distances = torch.cdist(real_flat, fake_flat).squeeze(1)
    valid_samples = (distances > epsilon).float()
    wandb.log(
        {
            "max_distance": distances.max().item(),
            "min_distance": distances.min().item(),
            "mean_distance": distances.mean().item(),
            "epsilon": epsilon,
        }
    )
    min_distances, _ = (distances + (1 - valid_samples) * distances.max()).min(dim=1)
    valid_real_samples = (min_distances < distances.max()).float()
    if valid_real_samples.sum() > 0:
        loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
    else:
        loss = torch.tensor(0.0, device=real_samples.device)
    return loss


def process_image(images, vision_encoder, device):
    B, T, C, H, W = images.shape
    images = images.flatten(end_dim=1).to(device)
    image_features = vision_encoder(images)
    image_features = image_features.reshape(B, T, -1)
    return image_features


def train(
    args: ExperimentConfig,
    nets,
    dataloader,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    ema,
):
    nets.train()

    folder = os.path.join("saved_weights", args.task_name, args.model.name + "_" + args.exp_name)
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

    for epoch in range(n_epochs):
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

                    loss = rs_imle_loss(naction, fake_actions, args.model.epsilon)
                else:
                    raise NotImplementedError

                # If loss is 0, skip backprop and log flag in wandb
                if loss == 0:
                    wandb.log({"zero_loss": 1})
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(nets.parameters())
                    wandb.log({"zero_loss": 0})

                wandb.log({"loss": loss.item()})

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        ema_nets = copy.deepcopy(nets)
        ema.copy_to(ema_nets.parameters())

        # save a checkpoint every 10 epochs
        if (epoch) % args.training_params.save_period == 0:
            torch.save(nets.state_dict(), f"{folder}/net_epoch_{epoch:04d}.pth")
            shutil.copy(f"{folder}/net_epoch_{epoch:04d}.pth", f"{folder}/net_epoch_last.pth")
            torch.save(ema_nets.state_dict(), f"{folder}/ema_net_epoch_{epoch:04d}.pth")
            shutil.copy(
                f"{folder}/ema_net_epoch_{epoch:04d}.pth",
                f"{folder}/ema_net_epoch_last.pth",
            )

        avg_loss = np.mean(epoch_loss)
        wandb.log({"avg_train_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch + 1}/{n_epochs} - Avg. Loss: {avg_loss:.4f} - Time: {time.time() - start_time:.2f}s")
        # If the loss is 0 for a whole epoch, log flag in wandb
        if avg_loss == 0:
            wandb.log({"zero_loss_epoch": 1})
        else:
            wandb.log({"zero_loss_epoch": 0})

    return


def main():
    from dataclasses import asdict
    from rs_imle_policy.configs.experiment_configs import FrankaExperimentConfigChoice
    
    from rs_imle_policy.datasets.single_franka import PandaPolicyDataset

    args = tyro.cli(FrankaExperimentConfigChoice)
    exp_name = args.process_name()
    wandb.init(project=args.task_name, config=asdict(args), name = exp_name)

    dataset = PandaPolicyDataset(
        args.dataset_path,
        args.model.pred_horizon,
        args.model.obs_horizon,
        args.model.action_horizon,
        low_dim_obs_keys=args.data.lowdim_obs_keys,
        action_keys=args.data.action_keys,
        vision_config=args.data.vision,
        use_next_state=args.data.use_next_state,
    )
    if args.debug:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.training_params.batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            persistent_workers=False,
        )
        print("debug")
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.training_params.batch_size,
            num_workers=args.training_params.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )
        print("no debug")

    policy = Policy(config=args, training = True, dataset = dataset)
    nets = policy.nets

    train(
        args,
        nets,
        dataloader,
        policy.noise_scheduler,
        policy.optimizer,
        policy.lr_scheduler,
        policy.ema,
    )

    wandb.finish()


if __name__ == "__main__":
    main()

"""Policy module for robot manipulation learning.

This module provides the Policy class which manages model initialization,
training setup, and inference for both diffusion-based and RS-IMLE policies.
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from rs_imle_policy.configs.train_config import Diffusion, ExperimentConfig, RSIMLE
from rs_imle_policy.datasets import BaseDataset
from rs_imle_policy.network import (
    DiffusionConditionalUnet1D,
    GeneratorConditionalUnet1D,
    get_resnet,
    replace_bn_with_gn,
)


class Policy:
    """Manages policy model initialization, training, and inference.

    Supports both diffusion-based policies and RS-IMLE generative policies
    for robot manipulation tasks.

    Attributes:
        config: Experiment configuration
        device: Compute device (CPU or CUDA)
        precision: Model precision (float32 or float16)
        noise_scheduler: Diffusion scheduler (for diffusion models only)
        nets: Model networks
        ema: Exponential moving average model
        stats: Data normalization statistics
        transform: Image preprocessing transforms
    """

    def __init__(
        self,
        config: ExperimentConfig,
        training: bool = True,
        dataset: BaseDataset | None = None,
    ):
        """Initialize the policy.

        Args:
            config: Experiment configuration object
        """
        self.config = config

        if isinstance(self.config.model, Diffusion):
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.config.model.num_diffusion_iters,
                beta_schedule=self.config.model.beta_schedule,
                clip_sample=self.config.model.clip_sample,
                prediction_type=self.config.model.prediction_type,
            )
        else:
            self.noise_scheduler = None

        self.precision = torch.float32
        self.device = self.config.model.device
        if training:
            assert dataset is not None, "Dataset must be provided for training mode"
            self.dataset = dataset
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.training_params.batch_size,
                num_workers=self.config.training_params.num_workers,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
            )
            self.config.action_shape = self.dataset.action_shape
            self.config.obs_shape = self.dataset.obs_shape
            self.nets = self.create_networks()
            self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)

            self.optimizer = torch.optim.AdamW(
                params=self.nets.parameters(),
                lr=self.config.training_params.lr,
                weight_decay=self.config.training_params.weight_decay,
            )
            self.lr_scheduler = get_scheduler(
                name=self.config.training_params.lr_scheduler_profile,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.training_params.num_warmup_steps,
                num_training_steps=len(self.dataloader) * self.config.training_params.num_epochs,
            )

            print("Training Mode.")
        else:
            self.folder = os.path.join(
                "saved_weights",
                self.config.task_name,
                self.config.model.name + "_" + self.config.exp_name,
            )
            stats_path = os.path.join(self.folder, "stats.pkl")
            self.stats = np.load(stats_path, allow_pickle=True)

            self.nets = self.create_networks()
            self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)

            self.load_weights()
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((240, 320)),
                    transforms.CenterCrop((216, 288)),
                    transforms.ToTensor(),
                ]
            )

            print("Inference Mode.")

    def load_weights(self):
        """Load pretrained model weights from checkpoint."""
        if self.config.epoch is None:
            epoch_str = "last"
        else:
            epoch_str = f"{self.config.epoch:04d}"
        print(f"Loading pretrained weights from epoch {epoch_str}...")
        if isinstance(self.config.model, Diffusion):
            print("Loading pretrained weights for diffusion model")
            self.ema_nets = copy.deepcopy(self.nets)

            fpath_ema = os.path.join(self.folder, f"ema_net_epoch_{epoch_str}.pth")
            state_dict_ema = torch.load(fpath_ema, map_location=self.device)
            self.ema_nets.load_state_dict(state_dict_ema)

            if self.precision == torch.float16:
                self.nets.half()
                self.ema_nets.half()

            self.ema = EMAModel(parameters=self.ema_nets.parameters(), power=0.75)

        elif isinstance(self.config.model, RSIMLE):
            print("Loading pretrained weights for RS-IMLE model")
            fpath = os.path.join(self.folder, f"net_epoch_{epoch_str}.pth")
            self.nets.load_state_dict(torch.load(fpath, map_location=self.device))

        print("Pretrained weights loaded.")

    def create_networks(self) -> nn.ModuleDict:
        """Create and initialize neural networks for the policy.

        Returns:
            ModuleDict containing vision encoders and policy network
        """
        cameras = self.config.data.vision.cameras
        vision_encoders = {f"vision_encoder_{camera}": replace_bn_with_gn(get_resnet("resnet18")) for camera in cameras}

        if isinstance(self.config.model, Diffusion):
            noise_pred_net = DiffusionConditionalUnet1D(
                input_dim=self.config.action_shape,
                global_cond_dim=self.config.obs_shape * self.config.model.obs_horizon,
            )

            nets = nn.ModuleDict(
                {
                    **vision_encoders,
                    "noise_pred_net": noise_pred_net,
                }
            )
        elif isinstance(self.config.model, RSIMLE):
            generator = GeneratorConditionalUnet1D(
                input_dim=self.config.action_shape,
                global_cond_dim=self.config.obs_shape * self.config.model.obs_horizon,
            )
            nets = nn.ModuleDict(
                {
                    **vision_encoders,
                    "generator": generator,
                }
            )

        return nets.to(self.config.model.device)

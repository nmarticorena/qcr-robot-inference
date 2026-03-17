"""Neural network architectures for robot policy learning.

This module provides network components including ResNet encoders,
UNet architectures for diffusion models and generative models,
and utility functions for network modification.
"""

import math
from typing import Callable, Union

import torch
import torch.nn as nn
import torchvision


def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """Get a ResNet model with the final FC layer removed.

    Args:
        name: ResNet architecture name (e.g., 'resnet18', 'resnet34', 'resnet50')
        weights: Pre-trained weights to load (e.g., 'IMAGENET1K_V1'), or None
        **kwargs: Additional arguments passed to the ResNet constructor

    Returns:
        ResNet model with Identity layer replacing the final FC layer
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # Remove the final fully connected layer
    # For resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace all submodules selected by predicate with the output of func.

    Args:
        root_module: Root module to traverse
        predicate: Function that returns True if the module should be replaced
        func: Function that returns the new module to use

    Returns:
        Modified root module with replacements applied
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # Verify that all modules are replaced
    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    """Replace all BatchNorm layers with GroupNorm.

    Args:
        root_module: Module to modify
        features_per_group: Number of features per group for GroupNorm

    Returns:
        Modified module with GroupNorm replacing BatchNorm layers
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(num_groups=x.num_features // features_per_group, num_channels=x.num_features),
    )
    return root_module


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Attributes:
        dim: Embedding dimension
    """

    def __init__(self, dim: int):
        """Initialize sinusoidal positional embedding.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings.

        Args:
            x: Input tensor of shape (batch_size,)

        Returns:
            Positional embeddings of shape (batch_size, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    """1D downsampling layer using strided convolution.

    Attributes:
        conv: 1D convolution layer
    """

    def __init__(self, dim: int):
        """Initialize downsampling layer.

        Args:
            dim: Number of input/output channels
        """
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling.

        Args:
            x: Input tensor of shape (batch, dim, length)

        Returns:
            Downsampled tensor of shape (batch, dim, length//2)
        """
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D upsampling layer using transposed convolution.

    Attributes:
        conv: 1D transposed convolution layer
    """

    def __init__(self, dim: int):
        """Initialize upsampling layer.

        Args:
            dim: Number of input/output channels
        """
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling.

        Args:
            x: Input tensor of shape (batch, dim, length)

        Returns:
            Upsampled tensor of shape (batch, dim, length*2)
        """
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """1D convolutional block with GroupNorm and Mish activation.

    Architecture: Conv1d -> GroupNorm -> Mish

    Attributes:
        block: Sequential container with conv, norm, and activation layers
    """

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        """Initialize Conv1d block.

        Args:
            inp_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution block.

        Args:
            x: Input tensor

        Returns:
            Output tensor after convolution, normalization, and activation
        """
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """1D residual block with FiLM conditioning.

    Applies Feature-wise Linear Modulation (FiLM) based on conditioning input.
    Reference: https://arxiv.org/abs/1709.07871

    Attributes:
        blocks: Two Conv1d blocks
        cond_encoder: Conditioning encoder for FiLM parameters
        residual_conv: Convolution for residual connection (if needed)
        out_channels: Number of output channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        """Initialize conditional residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            cond_dim: Dimension of conditioning input
            kernel_size: Convolution kernel size
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation: predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1)))

        # Make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply conditional residual block.

        Args:
            x: Input tensor of shape (batch_size, in_channels, horizon)
            cond: Conditioning tensor of shape (batch_size, cond_dim)

        Returns:
            Output tensor of shape (batch_size, out_channels, horizon)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class DiffusionConditionalUnet1D(nn.Module):
    """1D UNet for diffusion-based policy learning.

    Processes action sequences with diffusion timestep and observation conditioning.

    Attributes:
        diffusion_step_encoder: Encodes diffusion timestep
        down_modules: Downsampling path of UNet
        mid_modules: Middle processing blocks
        up_modules: Upsampling path of UNet
        final_conv: Final convolution to output dimension
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] = None,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        """Initialize diffusion conditional UNet.

        Args:
            input_dim: Dimension of action space
            global_cond_dim: Dimension of global conditioning (usually obs_horizon * obs_dim),
                applied with FiLM in addition to diffusion step embedding
            diffusion_step_embed_dim: Size of positional encoding for diffusion iteration
            down_dims: Channel sizes for each UNet level. Length determines number of levels.
            kernel_size: Convolution kernel size
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        if down_dims is None:
            down_dims = [256, 512, 1024]

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print(f"DiffusionConditionalUnet1D parameters: {sum(p.numel() for p in self.parameters()):e}")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through the diffusion UNet.

        Args:
            sample: Input tensor of shape (B, T, input_dim)
            timestep: Diffusion timestep, either (B,) tensor or scalar int/float
            global_cond: Optional global conditioning of shape (B, global_cond_dim)

        Returns:
            Output tensor of shape (B, T, input_dim)
        """
        # Convert from (B, T, C) to (B, C, T)
        sample = sample.moveaxis(-1, -2)

        # Encode timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class GeneratorConditionalUnet1D(nn.Module):
    """1D UNet for generative policy learning (RS-IMLE).

    Processes noise input with observation conditioning to generate action sequences.

    Attributes:
        down_modules: Downsampling path of UNet
        mid_modules: Middle processing blocks
        up_modules: Upsampling path of UNet
        final_conv: Final convolution to output dimension
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        down_dims: list[int] = None,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        """Initialize generator conditional UNet.

        Args:
            input_dim: Dimension of action space
            global_cond_dim: Dimension of global conditioning (usually obs_horizon * obs_dim),
                applied with FiLM
            down_dims: Channel sizes for each UNet level. Length determines number of levels.
            kernel_size: Convolution kernel size
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        if down_dims is None:
            down_dims = [256, 512, 1024]

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        cond_dim = global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print(f"GeneratorConditionalUnet1D parameters: {sum(p.numel() for p in self.parameters()):e}")

    def forward(self, sample: torch.Tensor, global_cond: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the generator UNet.

        Args:
            sample: Input noise tensor of shape (B, T, input_dim)
            global_cond: Optional global conditioning of shape (B, global_cond_dim)

        Returns:
            Generated action tensor of shape (B, T, input_dim)
        """
        # Convert from (B, T, C) to (B, C, T)
        sample = sample.moveaxis(-1, -2)

        global_feature = global_cond

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x

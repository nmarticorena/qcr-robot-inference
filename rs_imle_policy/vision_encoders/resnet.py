from typing import Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np



class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            #self.temperature = (torch.ones(1)*temperature).to('cuda')
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float().to('cuda')
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float().to('cuda')
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """

        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)
        
        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints

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

def output_shape(input_shape, feature_dim: int = 512):
    """
    Function to compute output shape from inputs to this module. 

    Args:
        input_shape (iterable of int): shape of input. Does not include batch dimension.
            Some modules may not need this argument, if their output does not depend 
            on the size of the input, or if they assume fixed size input.

    Returns:
        out_shape ([int]): list of integers corresponding to output shape
    """
    assert(len(input_shape) == 3)
    out_h = int(math.ceil(input_shape[1] / 32.))
    out_w = int(math.ceil(input_shape[2] / 32.))
    return [feature_dim, out_h, out_w]
    
def get_resnet(
    name: str,
    weights=None,
    input_res=None,
    use_spatial_softmax: bool = True,
    num_kp: int = 256,
    **kwargs,
) -> nn.Module:
    """
    Get a ResNet model with the final FC layer removed.

    Args:
        name: ResNet architecture name (e.g., 'resnet18', 'resnet34', 'resnet50')
        weights: Pre-trained weights to load (e.g., 'IMAGENET1K_V1'), or None
        input_res: Input image shape as (C, H, W)
        use_spatial_softmax: Whether to replace average pooling with SpatialSoftmax
        num_kp: Number of spatial softmax keypoints
        **kwargs: Additional arguments passed to the ResNet constructor

    Returns:
        ResNet model with Identity layer replacing the final FC layer
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    if use_spatial_softmax:
        if input_res is None:
            raise ValueError("input_res is required when use_spatial_softmax=True")
        o_shape = output_shape(input_res, feature_dim=resnet.fc.in_features)
        resnet.avgpool = SpatialSoftmax(input_shape=o_shape, num_kp=num_kp)
    resnet.fc = torch.nn.Identity()
    return resnet

def keypoint_spread_metrics(
    kps: torch.Tensor,
    collapse_threshold: float = 0.05,
    grid_size: int = 8,
) -> dict[str, torch.Tensor]:
    """
    kps: [B, K, 2], normalized coordinates in [-1, 1]

    Returns metrics that measure how spatially spread/collapsed the keypoints are.
    """

    if isinstance(kps, tuple):
        kps = kps[0]

    B, K, _ = kps.shape
    eps = 1e-8

    # Spread around the mean keypoint location per sample
    kp_mean = kps.mean(dim=1, keepdim=True)  # [B, 1, 2]
    centered = kps - kp_mean

    std_xy = kps.std(dim=1)  # [B, 2]
    std_x = std_xy[:, 0].mean()
    std_y = std_xy[:, 1].mean()
    std_mean = std_xy.mean()

    # Distance from image/crop center
    radius_from_center = torch.linalg.norm(kps, dim=-1)  # [B, K]
    mean_radius_from_center = radius_from_center.mean()

    # Distance from keypoint cloud center
    radius_from_kp_mean = torch.linalg.norm(centered, dim=-1)  # [B, K]
    mean_radius_from_kp_mean = radius_from_kp_mean.mean()

    # Pairwise distances between keypoints
    dists = torch.cdist(kps, kps)  # [B, K, K]

    # Remove diagonal self-distances
    eye = torch.eye(K, device=kps.device, dtype=torch.bool).unsqueeze(0)
    dists_no_diag = dists.masked_fill(eye, float("inf"))

    nearest_neighbor_dist = dists_no_diag.min(dim=-1).values.mean()

    # Mean pairwise distance excluding diagonal
    valid_dists = dists[~eye.expand_as(dists)]
    mean_pairwise_dist = valid_dists.mean()

    # Fraction of keypoint pairs closer than threshold
    close_pairs = valid_dists < collapse_threshold
    collapse_fraction = close_pairs.float().mean()

    # Grid occupancy: how many spatial bins contain at least one keypoint
    # Map [-1, 1] -> [0, grid_size - 1]
    grid_xy = ((kps + 1.0) * 0.5 * grid_size).long()
    grid_xy = torch.clamp(grid_xy, 0, grid_size - 1)

    occupancies = []
    for b in range(B):
        linear_idx = grid_xy[b, :, 1] * grid_size + grid_xy[b, :, 0]
        occupied = torch.unique(linear_idx).numel()
        occupancies.append(occupied / float(grid_size * grid_size))

    grid_occupancy = torch.tensor(occupancies, device=kps.device).mean()

    return {
        "kp/std_x": std_x,
        "kp/std_y": std_y,
        "kp/std_mean": std_mean,
        "kp/mean_radius_from_center": mean_radius_from_center,
        "kp/mean_radius_from_kp_mean": mean_radius_from_kp_mean,
        "kp/mean_pairwise_dist": mean_pairwise_dist,
        "kp/nearest_neighbor_dist": nearest_neighbor_dist,
        "kp/collapse_fraction": collapse_fraction,
        "kp/grid_occupancy": grid_occupancy,
    }

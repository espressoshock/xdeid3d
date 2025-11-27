# Copyright 2024 X-DeID3D Authors
# SPDX-License-Identifier: Apache-2.0
#
# This module contains code adapted from:
# - EG3D (https://github.com/NVlabs/eg3d) - NVIDIA Source Code License
# - SphereHead (https://github.com/lhyfst/spherehead) - See LICENSE_SPHEREHEAD
#
# The original code is licensed under NVIDIA proprietary license.
# Modifications and integration are licensed under Apache-2.0.
"""
SphereHead 3D Face Synthesis Engine.

This module provides the neural network architecture and rendering pipeline
for generating 3D full-head models using spherical tri-plane representation.

Key Components:
    - TriPlaneGenerator: Main generator network with spherical tri-planes
    - ImportanceRenderer: Volume renderer with importance sampling
    - StyleGAN2/3 backbones: Feature synthesis networks
    - Camera utilities: Pose sampling and camera matrix generation

Example:
    >>> from xdeid3d.spherehead import load_network, TriPlaneGenerator
    >>> from xdeid3d.spherehead.camera_utils import LookAtPoseSampler
    >>>
    >>> # Load pretrained model
    >>> G = load_network('path/to/checkpoint.pkl')
    >>>
    >>> # Generate image
    >>> z = torch.randn(1, G.z_dim)
    >>> cam = LookAtPoseSampler.sample(...)
    >>> img = G(z, cam)

Requirements:
    - PyTorch >= 1.11.0
    - CUDA 11.3+ (optional, for GPU acceleration)
    - Custom CUDA ops will fall back to reference implementation if unavailable
"""

from xdeid3d.spherehead.training.triplane import TriPlaneGenerator, OSGDecoder
from xdeid3d.spherehead.training.volumetric_rendering.renderer import ImportanceRenderer
from xdeid3d.spherehead.training.volumetric_rendering.ray_sampler import RaySampler
from xdeid3d.spherehead.training.volumetric_rendering.ray_marcher import MipRayMarcher2
from xdeid3d.spherehead.camera_utils import (
    LookAtPoseSampler,
    GaussianCameraPoseSampler,
    UniformCameraPoseSampler,
    create_cam2world_matrix,
    FOV_to_intrinsics,
)
from xdeid3d.spherehead.legacy import load_network_pkl
from xdeid3d.spherehead import dnnlib


def load_network(path: str, device: str = 'cuda', force_fp16: bool = False):
    """Load a pretrained SphereHead network.

    Args:
        path: Path to the .pkl checkpoint file
        device: Device to load the model to ('cuda', 'cpu', 'cuda:0', etc.)
        force_fp16: Force FP16 precision for faster inference

    Returns:
        dict: Dictionary containing 'G', 'D', 'G_ema' networks

    Example:
        >>> data = load_network('checkpoint.pkl', device='cuda:0')
        >>> G = data['G_ema'].eval()  # Use EMA generator for inference
    """
    import torch
    with open(path, 'rb') as f:
        data = load_network_pkl(f, force_fp16=force_fp16)

    # Move to device
    for key in ['G', 'D', 'G_ema']:
        if key in data and data[key] is not None:
            data[key] = data[key].to(device)

    return data


__all__ = [
    # Main generator
    "TriPlaneGenerator",
    "OSGDecoder",
    # Volume rendering
    "ImportanceRenderer",
    "RaySampler",
    "MipRayMarcher2",
    # Camera utilities
    "LookAtPoseSampler",
    "GaussianCameraPoseSampler",
    "UniformCameraPoseSampler",
    "create_cam2world_matrix",
    "FOV_to_intrinsics",
    # Loading utilities
    "load_network",
    "load_network_pkl",
    # Submodules
    "dnnlib",
]

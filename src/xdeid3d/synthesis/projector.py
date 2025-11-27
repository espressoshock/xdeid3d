"""
Image projection to latent space.

Provides utilities for projecting images into generator latent spaces
using optimization-based methods like PTI (Pivotal Tuning Inversion).

SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "ProjectorProtocol",
    "ProjectionResult",
    "ProjectionConfig",
    "BaseProjector",
    "LatentProjector",
    "PTIProjector",
]


@dataclass
class ProjectionResult:
    """Result of image projection.

    Attributes:
        latent: Projected latent code
        reconstruction: Reconstructed image
        loss_history: Loss values during optimization
        iterations: Number of iterations performed
        metadata: Additional projection metadata
    """
    latent: np.ndarray
    reconstruction: np.ndarray
    loss_history: List[float] = field(default_factory=list)
    iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionConfig:
    """Configuration for projection.

    Attributes:
        num_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer
        w_avg_samples: Number of samples for W average computation
        initial_noise_factor: Factor for initial noise
        noise_ramp_length: Length of noise ramping period
        regularize_noise_weight: Weight for noise regularization
        verbose: Print progress during optimization
    """
    num_steps: int = 1000
    learning_rate: float = 0.1
    w_avg_samples: int = 10000
    initial_noise_factor: float = 0.05
    noise_ramp_length: float = 0.75
    regularize_noise_weight: float = 1e5
    verbose: bool = False


class ProjectorProtocol(Protocol):
    """Protocol for image projectors.

    Defines the interface that all projectors must implement
    for projecting images into generator latent spaces.
    """

    def project(
        self,
        target_image: np.ndarray,
        config: Optional[ProjectionConfig] = None,
    ) -> ProjectionResult:
        """Project image to latent space.

        Args:
            target_image: Target image to project (H, W, C)
            config: Projection configuration

        Returns:
            ProjectionResult with latent code and reconstruction
        """
        ...

    def project_batch(
        self,
        target_images: List[np.ndarray],
        config: Optional[ProjectionConfig] = None,
    ) -> List[ProjectionResult]:
        """Project batch of images.

        Args:
            target_images: List of target images
            config: Projection configuration

        Returns:
            List of ProjectionResults
        """
        ...


class BaseProjector(ABC):
    """Abstract base class for projectors.

    Provides common functionality for all projector implementations.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize projector.

        Args:
            device: Device for computation
        """
        self.device = device
        self._initialized = False

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize projector resources."""
        pass

    def _ensure_initialized(self) -> None:
        """Ensure projector is initialized."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    @abstractmethod
    def project(
        self,
        target_image: np.ndarray,
        config: Optional[ProjectionConfig] = None,
    ) -> ProjectionResult:
        """Project image to latent space."""
        pass

    def project_batch(
        self,
        target_images: List[np.ndarray],
        config: Optional[ProjectionConfig] = None,
    ) -> List[ProjectionResult]:
        """Project batch of images sequentially.

        Override for parallel batch projection.
        """
        return [self.project(img, config) for img in target_images]


class LatentProjector(BaseProjector):
    """Latent code optimization projector.

    Projects images by optimizing latent codes to minimize
    reconstruction loss with optional perceptual losses.

    Example:
        >>> projector = LatentProjector(generator, latent_dim=512)
        >>> result = projector.project(target_image)
        >>> latent_code = result.latent
    """

    def __init__(
        self,
        generator: Any,
        latent_dim: int = 512,
        w_space: bool = True,
        num_ws: int = 14,
        device: str = "cuda",
        perceptual_loss: Optional[Callable] = None,
    ):
        """Initialize latent projector.

        Args:
            generator: Generator network with synthesis method
            latent_dim: Latent space dimensionality
            w_space: Project to W space (True) or Z space (False)
            num_ws: Number of W vectors for W+ space
            device: Device for computation
            perceptual_loss: Optional perceptual loss function
        """
        super().__init__(device)
        self.generator = generator
        self.latent_dim = latent_dim
        self.w_space = w_space
        self.num_ws = num_ws
        self.perceptual_loss = perceptual_loss
        self._w_avg = None

    def _initialize(self) -> None:
        """Initialize W average and move to device."""
        try:
            import torch
            self._torch = torch

            # Move generator to device
            if hasattr(self.generator, 'to'):
                self.generator = self.generator.to(self.device)
                self.generator.eval()

        except ImportError:
            raise ImportError("PyTorch is required for LatentProjector")

    def _compute_w_avg(self, num_samples: int = 10000) -> "torch.Tensor":
        """Compute average W vector."""
        if self._w_avg is not None:
            return self._w_avg

        torch = self._torch
        with torch.no_grad():
            z_samples = torch.randn(num_samples, self.latent_dim, device=self.device)

            if hasattr(self.generator, 'mapping'):
                w_samples = self.generator.mapping(z_samples, None)
            else:
                # Fallback: assume generator takes z directly
                w_samples = z_samples

            self._w_avg = w_samples.mean(dim=0, keepdim=True)

        return self._w_avg

    def project(
        self,
        target_image: np.ndarray,
        config: Optional[ProjectionConfig] = None,
    ) -> ProjectionResult:
        """Project image to latent space.

        Args:
            target_image: Target image (H, W, C) in [0, 255] uint8 or [0, 1] float
            config: Projection configuration

        Returns:
            ProjectionResult with optimized latent code
        """
        self._ensure_initialized()
        torch = self._torch

        if config is None:
            config = ProjectionConfig()

        # Preprocess target image
        if target_image.dtype == np.uint8:
            target = target_image.astype(np.float32) / 255.0
        else:
            target = target_image.astype(np.float32)

        # Convert to tensor (B, C, H, W)
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
        target_tensor = target_tensor.to(self.device)

        # Normalize to [-1, 1] for StyleGAN-style generators
        target_tensor = target_tensor * 2 - 1

        # Initialize latent code
        if self.w_space:
            w_avg = self._compute_w_avg(config.w_avg_samples)
            # Start from W average
            if self.num_ws > 1:
                # W+ space
                w_opt = w_avg.repeat(1, self.num_ws, 1).clone()
            else:
                w_opt = w_avg.clone()
            w_opt.requires_grad_(True)
            latent_param = w_opt
        else:
            # Z space
            z_opt = torch.randn(1, self.latent_dim, device=self.device, requires_grad=True)
            latent_param = z_opt

        # Setup optimizer
        optimizer = torch.optim.Adam([latent_param], lr=config.learning_rate)

        # Optimization loop
        loss_history = []

        for step in range(config.num_steps):
            optimizer.zero_grad()

            # Generate image
            if self.w_space:
                if hasattr(self.generator, 'synthesis'):
                    synth_image = self.generator.synthesis(latent_param)
                else:
                    synth_image = self.generator(latent_param)
            else:
                if hasattr(self.generator, 'mapping'):
                    w = self.generator.mapping(latent_param, None)
                    synth_image = self.generator.synthesis(w)
                else:
                    synth_image = self.generator(latent_param)

            # Compute loss
            loss = torch.nn.functional.mse_loss(synth_image, target_tensor)

            # Add perceptual loss if available
            if self.perceptual_loss is not None:
                loss = loss + self.perceptual_loss(synth_image, target_tensor)

            # Noise regularization (ramp down)
            if config.regularize_noise_weight > 0 and hasattr(self, '_noise_buffers'):
                noise_ramp = min(step / (config.num_steps * config.noise_ramp_length), 1.0)
                noise_weight = config.regularize_noise_weight * (1 - noise_ramp)
                for noise in self._noise_buffers:
                    loss = loss + noise_weight * (noise * noise).mean()

            loss_history.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()

            if config.verbose and step % 100 == 0:
                print(f"Step {step}/{config.num_steps}, Loss: {loss.item():.4f}")

        # Generate final reconstruction
        with torch.no_grad():
            if self.w_space:
                if hasattr(self.generator, 'synthesis'):
                    final_image = self.generator.synthesis(latent_param)
                else:
                    final_image = self.generator(latent_param)
            else:
                if hasattr(self.generator, 'mapping'):
                    w = self.generator.mapping(latent_param, None)
                    final_image = self.generator.synthesis(w)
                else:
                    final_image = self.generator(latent_param)

            # Convert back to numpy
            final_image = (final_image + 1) / 2  # [-1, 1] -> [0, 1]
            final_image = final_image.clamp(0, 1)
            final_image = final_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            final_image = (final_image * 255).astype(np.uint8)

            latent_np = latent_param.detach().cpu().numpy()

        return ProjectionResult(
            latent=latent_np,
            reconstruction=final_image,
            loss_history=loss_history,
            iterations=config.num_steps,
            metadata={
                "final_loss": loss_history[-1] if loss_history else None,
                "w_space": self.w_space,
                "latent_shape": latent_np.shape,
            }
        )


class PTIProjector(BaseProjector):
    """Pivotal Tuning Inversion projector.

    Two-stage projection method:
    1. First stage: Optimize latent code (like LatentProjector)
    2. Second stage: Fine-tune generator to better fit target

    This provides higher-fidelity reconstructions at the cost
    of modifying the generator weights.

    Example:
        >>> projector = PTIProjector(generator)
        >>> result = projector.project(target_image)
        >>> fine_tuned_generator = projector.get_tuned_generator()
    """

    def __init__(
        self,
        generator: Any,
        latent_dim: int = 512,
        num_ws: int = 14,
        device: str = "cuda",
        lpips_net: str = "vgg",
    ):
        """Initialize PTI projector.

        Args:
            generator: Generator network
            latent_dim: Latent dimensionality
            num_ws: Number of W vectors
            device: Computation device
            lpips_net: Network for perceptual loss
        """
        super().__init__(device)
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_ws = num_ws
        self.lpips_net = lpips_net
        self._tuned_generator = None
        self._pivot_latent = None

    def _initialize(self) -> None:
        """Initialize projector and LPIPS."""
        try:
            import torch
            self._torch = torch

            if hasattr(self.generator, 'to'):
                self.generator = self.generator.to(self.device)
                self.generator.eval()

            # Try to load LPIPS
            try:
                import lpips
                self._lpips = lpips.LPIPS(net=self.lpips_net).to(self.device)
            except ImportError:
                print("Warning: LPIPS not available, using MSE loss only")
                self._lpips = None

        except ImportError:
            raise ImportError("PyTorch is required for PTIProjector")

    def project(
        self,
        target_image: np.ndarray,
        config: Optional[ProjectionConfig] = None,
    ) -> ProjectionResult:
        """Project image using PTI method.

        Args:
            target_image: Target image (H, W, C)
            config: Projection configuration

        Returns:
            ProjectionResult with pivot latent and fine-tuned reconstruction
        """
        self._ensure_initialized()
        torch = self._torch

        if config is None:
            config = ProjectionConfig()

        # Stage 1: Latent optimization to find pivot
        print("PTI Stage 1: Finding pivot latent...")
        latent_projector = LatentProjector(
            generator=self.generator,
            latent_dim=self.latent_dim,
            w_space=True,
            num_ws=self.num_ws,
            device=self.device,
            perceptual_loss=self._lpips_loss if self._lpips else None,
        )
        latent_projector._initialized = True
        latent_projector._torch = self._torch

        stage1_result = latent_projector.project(target_image, config)
        self._pivot_latent = torch.from_numpy(stage1_result.latent).to(self.device)

        # Stage 2: Generator tuning
        print("PTI Stage 2: Fine-tuning generator...")

        # Preprocess target
        if target_image.dtype == np.uint8:
            target = target_image.astype(np.float32) / 255.0
        else:
            target = target_image.astype(np.float32)

        target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
        target_tensor = target_tensor.to(self.device)
        target_tensor = target_tensor * 2 - 1  # [-1, 1]

        # Clone generator for tuning
        import copy
        self._tuned_generator = copy.deepcopy(self.generator)
        self._tuned_generator.train()

        # Only tune synthesis network
        if hasattr(self._tuned_generator, 'synthesis'):
            tunable_params = self._tuned_generator.synthesis.parameters()
        else:
            tunable_params = self._tuned_generator.parameters()

        optimizer = torch.optim.Adam(tunable_params, lr=config.learning_rate * 0.01)

        pti_steps = config.num_steps // 2
        pti_loss_history = []

        for step in range(pti_steps):
            optimizer.zero_grad()

            # Generate with pivot latent
            if hasattr(self._tuned_generator, 'synthesis'):
                synth_image = self._tuned_generator.synthesis(self._pivot_latent)
            else:
                synth_image = self._tuned_generator(self._pivot_latent)

            # Compute losses
            mse_loss = torch.nn.functional.mse_loss(synth_image, target_tensor)

            if self._lpips is not None:
                lpips_loss = self._lpips(synth_image, target_tensor).mean()
                loss = mse_loss + lpips_loss
            else:
                loss = mse_loss

            pti_loss_history.append(loss.item())
            loss.backward()
            optimizer.step()

            if config.verbose and step % 50 == 0:
                print(f"PTI Step {step}/{pti_steps}, Loss: {loss.item():.4f}")

        # Generate final result
        self._tuned_generator.eval()
        with torch.no_grad():
            if hasattr(self._tuned_generator, 'synthesis'):
                final_image = self._tuned_generator.synthesis(self._pivot_latent)
            else:
                final_image = self._tuned_generator(self._pivot_latent)

            final_image = (final_image + 1) / 2
            final_image = final_image.clamp(0, 1)
            final_image = final_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            final_image = (final_image * 255).astype(np.uint8)

        combined_loss = stage1_result.loss_history + pti_loss_history

        return ProjectionResult(
            latent=self._pivot_latent.cpu().numpy(),
            reconstruction=final_image,
            loss_history=combined_loss,
            iterations=config.num_steps + pti_steps,
            metadata={
                "stage1_iterations": config.num_steps,
                "stage2_iterations": pti_steps,
                "stage1_final_loss": stage1_result.loss_history[-1] if stage1_result.loss_history else None,
                "stage2_final_loss": pti_loss_history[-1] if pti_loss_history else None,
                "method": "pti",
            }
        )

    def _lpips_loss(self, synth: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        """Compute LPIPS loss."""
        if self._lpips is not None:
            return self._lpips(synth, target).mean()
        return self._torch.tensor(0.0, device=self.device)

    def get_tuned_generator(self) -> Any:
        """Get the fine-tuned generator after PTI.

        Returns:
            Fine-tuned generator or None if not yet projected
        """
        return self._tuned_generator

    def get_pivot_latent(self) -> Optional[np.ndarray]:
        """Get the pivot latent code.

        Returns:
            Pivot latent code or None if not yet projected
        """
        if self._pivot_latent is not None:
            return self._pivot_latent.cpu().numpy()
        return None


def create_projector(
    generator: Any,
    method: str = "latent",
    **kwargs,
) -> BaseProjector:
    """Factory function to create projectors.

    Args:
        generator: Generator network
        method: Projection method ("latent" or "pti")
        **kwargs: Additional arguments for projector

    Returns:
        Configured projector instance

    Example:
        >>> projector = create_projector(generator, method="pti")
        >>> result = projector.project(image)
    """
    if method == "latent":
        return LatentProjector(generator, **kwargs)
    elif method == "pti":
        return PTIProjector(generator, **kwargs)
    else:
        raise ValueError(f"Unknown projection method: {method}")

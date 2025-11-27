"""
Background Replacement Service

This service handles background customization for generated identities by:
1. Loading latent codes from Stage 1 (generation)
2. Re-synthesizing the head with mask/alpha channel
3. Compositing onto custom backgrounds (color/image/text-generated)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import PIL.Image
from typing import Optional, Tuple, Callable
import torch.nn.functional as F

# Add core directory to path for imports
CORE_DIR = Path(__file__).parent.parent.parent.parent / "core"
sys.path.insert(0, str(CORE_DIR))

import dnnlib
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


class BackgroundReplacementService:
    """
    Service for replacing backgrounds in generated 3D head identities.

    Workflow:
    1. Load model and latent code from generation stage
    2. Synthesize head with segmentation mask
    3. Composite onto custom background
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the service with a SphereHead model.

        Args:
            model_path: Path to the checkpoint .pkl file
            device: Device to use (cuda:0, cpu, etc.). Auto-detects if None.
        """
        self.model_path = model_path
        self.device = torch.device(device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.G = None
        self.loaded = False

    def load_model(self):
        """Load the SphereHead generator model."""
        if self.loaded:
            return

        print(f'Loading SphereHead model from {self.model_path}...')

        with dnnlib.util.open_url(self.model_path) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        # Reload modules for code modifications
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(self.device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs

        self.G = G_new
        self.loaded = True
        print('Model loaded successfully')

    def load_latent_code(self, latent_path: Optional[str], generator_path: Optional[str] = None) -> torch.Tensor:
        """
        Load latent code from .npz file (and optionally fine-tuned generator for PTI).

        If latent_path is None or doesn't exist, this assumes seed-based generation
        and will require the seed to be passed separately.

        Args:
            latent_path: Path to the latent code .npz file (from projector.py), or None for seed-based
            generator_path: Optional path to fine-tuned generator .pkl (for PTI results)

        Returns:
            Latent code tensor (ws), or None if latent_path not provided
        """
        if not self.loaded:
            self.load_model()

        # Load PTI fine-tuned generator if provided
        if generator_path and os.path.exists(generator_path):
            print(f'Loading fine-tuned generator from {generator_path}...')
            with dnnlib.util.open_url(generator_path) as f:
                G_finetuned = legacy.load_network_pkl(f)['G_ema'].to(self.device)

            # Reload modules
            G_new = TriPlaneGenerator(*G_finetuned.init_args, **G_finetuned.init_kwargs).eval().requires_grad_(False).to(self.device)
            misc.copy_params_and_buffers(G_finetuned, G_new, require_all=True)
            G_new.neural_rendering_resolution = G_finetuned.neural_rendering_resolution
            G_new.rendering_kwargs = G_finetuned.rendering_kwargs

            self.G = G_new
            print('Fine-tuned generator loaded')

        # Load latent code if path provided
        if latent_path and os.path.exists(latent_path):
            print(f'Loading latent code from {latent_path}...')
            latent_data = np.load(latent_path)
            ws = torch.tensor(latent_data['w']).to(self.device)
            return ws

        # Return None if no latent code (will use seed instead)
        return None

    def generate_latent_from_seed(
        self,
        seed: int,
        truncation_psi: float = 0.7,
        truncation_cutoff: int = 14
    ) -> torch.Tensor:
        """
        Generate latent code from seed (for seed-based generation).

        Args:
            seed: Random seed
            truncation_psi: Truncation parameter
            truncation_cutoff: Truncation cutoff

        Returns:
            Latent code tensor (ws)
        """
        if not self.loaded:
            self.load_model()

        print(f'Generating latent code from seed {seed}...')

        # Generate z from seed
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)

        # Create conditioning camera (frontal view)
        intrinsics = FOV_to_intrinsics(18.837, device=self.device)
        cam_pivot = torch.tensor([0, 0, 0.2], device=self.device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)

        conditioning_cam2world_pose = LookAtPoseSampler.sample(
            np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=self.device
        )
        conditioning_params = torch.cat([
            conditioning_cam2world_pose.reshape(-1, 16),
            intrinsics.reshape(-1, 9)
        ], 1)

        # Map to W space
        with torch.no_grad():
            ws = self.G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

        return ws

    def synthesize_with_mask(
        self,
        ws: torch.Tensor,
        truncation_psi: float = 0.7,
        fov_deg: float = 18.837,
        angle_y: float = 0.0,
        angle_p: float = -0.2,
        neural_rendering_resolution: Optional[int] = None
    ) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
        """
        Synthesize head image with segmentation mask.

        Args:
            ws: Latent code tensor
            truncation_psi: Truncation parameter
            fov_deg: Field of view in degrees
            angle_y: Yaw angle for camera
            angle_p: Pitch angle for camera
            neural_rendering_resolution: Override neural rendering resolution

        Returns:
            (image, mask) tuple where:
                - image: PIL Image of the head (RGB)
                - mask: PIL Image of the alpha mask (L mode, single channel)
        """
        if not self.loaded:
            self.load_model()

        # Camera setup
        intrinsics = FOV_to_intrinsics(fov_deg, device=self.device)
        cam_pivot = torch.tensor([0, 0, 0.2], device=self.device)  # Head config
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)

        # Create camera parameters for desired viewpoint
        cam2world_pose = LookAtPoseSampler.sample(
            np.pi/2 + angle_y,
            np.pi/2 + angle_p,
            cam_pivot,
            radius=cam_radius,
            device=self.device
        )
        camera_params = torch.cat([
            cam2world_pose.reshape(-1, 16),
            intrinsics.reshape(-1, 9)
        ], 1)

        # Set neural rendering resolution if specified
        if neural_rendering_resolution:
            original_nrr = self.G.neural_rendering_resolution
            self.G.neural_rendering_resolution = neural_rendering_resolution

        # Synthesize
        with torch.no_grad():
            G_out = self.G.synthesis(ws, camera_params, noise_mode='const')

        # Restore original resolution
        if neural_rendering_resolution:
            self.G.neural_rendering_resolution = original_nrr

        # Extract image and mask
        img_tensor = G_out['image']  # Shape: [1, 3, H, W], range [-1, 1]
        mask_tensor = G_out['image_mask']  # Shape: [1, 1, H_mask, W_mask], range ~[0, 1]

        # Convert to PIL Images
        img_np = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        image = PIL.Image.fromarray(img_np, 'RGB')

        mask_np = (mask_tensor[0, 0] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        mask = PIL.Image.fromarray(mask_np, 'L')

        # Resize mask to match image size (mask is at neural rendering resolution, image is super-resolved)
        if mask.size != image.size:
            mask = mask.resize(image.size, PIL.Image.LANCZOS)

        return image, mask

    def apply_solid_background(
        self,
        ws: torch.Tensor,
        bg_color: Tuple[int, int, int],
        **synthesis_kwargs
    ) -> PIL.Image.Image:
        """
        Replace background with a solid color.

        Args:
            ws: Latent code
            bg_color: RGB tuple (0-255)
            **synthesis_kwargs: Additional arguments for synthesize_with_mask

        Returns:
            Composited image
        """
        # Synthesize head with mask
        head_image, mask = self.synthesize_with_mask(ws, **synthesis_kwargs)

        # Create solid color background
        bg = PIL.Image.new('RGB', head_image.size, bg_color)

        # Composite: bg + head (using mask as alpha)
        result = PIL.Image.composite(head_image, bg, mask)

        return result

    def apply_image_background(
        self,
        ws: torch.Tensor,
        bg_image_path: str,
        fit_mode: str = 'cover',
        **synthesis_kwargs
    ) -> PIL.Image.Image:
        """
        Replace background with a custom image.

        Args:
            ws: Latent code
            bg_image_path: Path to background image
            fit_mode: How to fit background ('cover', 'contain', 'stretch')
            **synthesis_kwargs: Additional arguments for synthesize_with_mask

        Returns:
            Composited image
        """
        # Synthesize head with mask
        head_image, mask = self.synthesize_with_mask(ws, **synthesis_kwargs)

        # Load and process background image
        bg_image = PIL.Image.open(bg_image_path).convert('RGB')

        # Resize background to match head image size based on fit mode
        target_size = head_image.size

        if fit_mode == 'stretch':
            # Stretch to fill
            bg_image = bg_image.resize(target_size, PIL.Image.LANCZOS)
        elif fit_mode == 'contain':
            # Fit entire image, may have letterboxing
            bg_image.thumbnail(target_size, PIL.Image.LANCZOS)
            # Create canvas and paste centered
            bg_canvas = PIL.Image.new('RGB', target_size, (0, 0, 0))
            offset = ((target_size[0] - bg_image.size[0]) // 2,
                     (target_size[1] - bg_image.size[1]) // 2)
            bg_canvas.paste(bg_image, offset)
            bg_image = bg_canvas
        else:  # cover (default)
            # Fill entire area, may crop image
            bg_ratio = bg_image.size[0] / bg_image.size[1]
            target_ratio = target_size[0] / target_size[1]

            if bg_ratio > target_ratio:
                # Background wider, fit height
                new_height = target_size[1]
                new_width = int(new_height * bg_ratio)
            else:
                # Background taller, fit width
                new_width = target_size[0]
                new_height = int(new_width / bg_ratio)

            bg_image = bg_image.resize((new_width, new_height), PIL.Image.LANCZOS)

            # Crop to center
            left = (new_width - target_size[0]) // 2
            top = (new_height - target_size[1]) // 2
            bg_image = bg_image.crop((left, top, left + target_size[0], top + target_size[1]))

        # Composite
        result = PIL.Image.composite(head_image, bg_image, mask)

        return result

    def apply_text_background(
        self,
        ws: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        output_dir: Optional[Path] = None,
        **synthesis_kwargs
    ) -> Tuple[PIL.Image.Image, str]:
        """
        Generate background from text prompt using Stable Diffusion and composite.

        Args:
            ws: Latent code
            prompt: Text description of background
            negative_prompt: What to avoid
            output_dir: Where to save generated background (optional)
            **synthesis_kwargs: Additional arguments for synthesize_with_mask

        Returns:
            (result_image, bg_image_path) tuple
        """
        # Synthesize head first to know the size
        head_image, mask = self.synthesize_with_mask(ws, **synthesis_kwargs)

        # Import Stable Diffusion here to avoid circular dependencies
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise RuntimeError("diffusers library not installed. Install with: pip install diffusers transformers")

        # Load SD model
        print(f"Generating background from prompt: {prompt}")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)

        if self.device.type == "cuda":
            pipe.enable_attention_slicing()

        # Enhance prompt for backgrounds
        enhanced_prompt = f"{prompt}, high quality, detailed, professional photography"
        if not negative_prompt:
            negative_prompt = "people, faces, text, watermark, blurry, low quality"

        # Generate background (512x512 default)
        with torch.no_grad():
            bg_generated = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,  # Faster generation
                guidance_scale=7.5,
                height=512,
                width=512,
            ).images[0]

        # Resize to match head image
        bg_image = bg_generated.resize(head_image.size, PIL.Image.LANCZOS)

        # Save generated background if output_dir provided
        bg_image_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            bg_image_path = output_dir / "generated_background.png"
            bg_image.save(bg_image_path)

        # Composite
        result = PIL.Image.composite(head_image, bg_image, mask)

        # Clean up SD model
        del pipe
        torch.cuda.empty_cache()

        return result, str(bg_image_path) if bg_image_path else None

    def cleanup(self):
        """Free GPU memory."""
        if self.G is not None:
            del self.G
            self.G = None
            self.loaded = False
            torch.cuda.empty_cache()


# Singleton instance
_background_service: Optional[BackgroundReplacementService] = None


def get_background_service(model_path: str) -> BackgroundReplacementService:
    """
    Get or create the singleton BackgroundReplacementService.

    Args:
        model_path: Path to SphereHead model checkpoint

    Returns:
        BackgroundReplacementService instance
    """
    global _background_service

    if _background_service is None or _background_service.model_path != model_path:
        _background_service = BackgroundReplacementService(model_path)

    return _background_service

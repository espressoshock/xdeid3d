"""
Generate a single frontal identity image from a seed.
Simplified version of gen_samples.py that outputs only one frontal view.
"""
import os
import click
import numpy as np
import PIL.Image
import torch
from xdeid3d.spherehead import dnnlib
from xdeid3d.spherehead import legacy
from xdeid3d.spherehead.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from xdeid3d.spherehead.torch_utils import misc
from xdeid3d.spherehead.training.triplane import TriPlaneGenerator


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', type=int, help='Random seed', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True)
@click.option('--fov-deg', help='Field of View in degrees', type=float, default=18.837, show_default=True)
@click.option('--pose-cond', type=int, help='Camera conditioned pose angle', default=90, show_default=True)
def generate_single_image(
    network_pkl: str,
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    fov_deg: float,
    pose_cond: int
):
    """Generate a single frontal identity image from a seed."""

    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Reload modules for code modifications
    print("Reloading modules...")
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    print(f'Output directory: {outdir}')

    # Camera setup
    pose_cond_rad = pose_cond / 180 * np.pi
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor([0, 0, 0], device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)

    # Conditioning camera (frontal view)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(
        pose_cond_rad, np.pi/2, cam_pivot, radius=cam_radius, device=device
    )
    conditioning_params = torch.cat([
        conditioning_cam2world_pose.reshape(-1, 16),
        intrinsics.reshape(-1, 9)
    ], 1)

    print(f'Generating frontal view for seed {seed}...')

    # Generate latent code
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    # Generate frontal view ONLY (angle_y=0, angle_p=-0.2)
    angle_y = 0.0
    angle_p = -0.2

    cam2world_pose = LookAtPoseSampler.sample(
        np.pi/2 + angle_y,
        np.pi/2 + angle_p,
        cam_pivot,
        radius=cam_radius,
        device=device
    )
    camera_params = torch.cat([
        cam2world_pose.reshape(-1, 16),
        intrinsics.reshape(-1, 9)
    ], 1)

    # Map to W space
    ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    # Synthesize image
    G_out = G.synthesis(ws, camera_params)
    img = G_out['image']

    # Convert to uint8
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    # Save single frontal view
    output_path = os.path.join(outdir, f'seed{seed:04d}.png')
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(output_path)

    print(f'Saved: {output_path}')


if __name__ == '__main__':
    generate_single_image()

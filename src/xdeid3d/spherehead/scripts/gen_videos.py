''' Generate videos using pretrained network pickle.
Code adapted from following paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."
See LICENSES/LICENSE_EG3D for original license.
'''

import os
import re
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import click
from xdeid3d.spherehead import dnnlib
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio
import numpy as np
import scipy.interpolate
import torch
import tqdm
import mrcfile

from xdeid3d.spherehead import legacy
import pickle
import io

from xdeid3d.spherehead.camera_utils import LookAtPoseSampler
from xdeid3d.spherehead.torch_utils import misc
from xdeid3d.spherehead.training.triplane import TriPlaneGenerator

# Disable CUDA custom ops if no CUDA toolkit
NO_CUDA_TOOLKIT = not os.path.exists('/usr/local/cuda/bin/nvcc')
if NO_CUDA_TOOLKIT:
    os.environ['DISABLE_CUSTOM_OPS'] = '1'
    print("CUDA toolkit not found, will use reference implementations...")
    
    # Monkey patch all custom ops to force reference implementation
    from xdeid3d.spherehead.torch_utils.ops import bias_act
    original_bias_act = torch_utils.ops.bias_act.bias_act
    def patched_bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda'):
        # Force reference implementation
        return original_bias_act(x, b, dim, act, alpha, gain, clamp, impl='ref')
    torch_utils.ops.bias_act.bias_act = patched_bias_act
    
    # Patch upfirdn2d
    from xdeid3d.spherehead.torch_utils.ops import upfirdn2d
    original_upfirdn2d = torch_utils.ops.upfirdn2d.upfirdn2d
    def patched_upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
        # Force reference implementation
        return original_upfirdn2d(x, f, up, down, padding, flip_filter, gain, impl='ref')
    torch_utils.ops.upfirdn2d.upfirdn2d = patched_upfirdn2d
    
    # Patch filtered_lrelu
    from xdeid3d.spherehead.torch_utils.ops import filtered_lrelu
    original_filtered_lrelu = torch_utils.ops.filtered_lrelu.filtered_lrelu
    def patched_filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda'):
        # Force reference implementation
        return original_filtered_lrelu(x, fu, fd, b, up, down, padding, gain, slope, clamp, flip_filter, impl='ref')
    torch_utils.ops.filtered_lrelu.filtered_lrelu = patched_filtered_lrelu
#----------------------------------------------------------------------------

def load_network_pkl_cpu_safe(f, device):
    """Load network pickle with proper device handling."""
    # Always load to CPU first to avoid GPU OOM during unpickling
    import io
    buffer = io.BytesIO(f.read())
    
    # Custom unpickler that maps CUDA tensors to CPU
    class CPUUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            if isinstance(pid, tuple) and len(pid) > 0:
                # Handle storage objects
                if hasattr(pid[0], 'device') or str(pid[0]).startswith('cuda'):
                    # Force CPU storage
                    return torch.storage._TypedStorage(wrap_storage=pid[1].cpu(), dtype=pid[2])
            return super().persistent_load(pid)
    
    # First try direct torch.load with map_location
    try:
        buffer.seek(0)
        data = torch.load(buffer, map_location='cpu')
        if isinstance(data, dict) and 'G_ema' in data:
            return data
    except:
        pass
    
    # If that fails, use legacy loader but ensure CPU mapping
    buffer.seek(0)
    # Temporarily override torch.load to use CPU mapping
    original_load = torch.load
    torch.load = lambda f, **kwargs: original_load(f, map_location='cpu', **kwargs)
    try:
        data = legacy.load_network_pkl(buffer)
    finally:
        torch.load = original_load
    return data

#----------------------------------------------------------------------------

def setup_progress_file_monitoring(progress_file_path: str):
    """
    Setup real-time progress file monitoring by patching tqdm.

    When enabled, every tqdm progress bar will write its progress to a JSON file
    that can be monitored by external processes (e.g., GUI backend).

    Args:
        progress_file_path: Path to JSON file where progress will be written
    """
    import tqdm as tqdm_module

    progress_path = Path(progress_file_path)
    print(f"[Progress File] Enabled progress monitoring: {progress_path}", flush=True)

    # Store original tqdm class
    _OriginalTqdm = tqdm_module.tqdm

    class ProgressFileTqdm(_OriginalTqdm):
        """Patched tqdm that writes progress to file"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_write_time = 0
            self._write_interval = 0.1  # Write at most every 100ms

        def update(self, n=1):
            """Override update to write progress to file"""
            result = super().update(n)

            # Write progress to file (throttled)
            now = time.time()
            if now - self._last_write_time >= self._write_interval or self.n >= self.total:
                self._last_write_time = now

                progress_data = {
                    "current": self.n,
                    "total": self.total if self.total else 100,
                    "progress": self.n / self.total if self.total and self.total > 0 else 0,
                    "desc": str(self.desc) if self.desc else "",
                    "rate": self.format_dict.get('rate', 0) if hasattr(self, 'format_dict') else 0,
                    "timestamp": now,
                }

                try:
                    # Atomic write using temp file
                    temp_file = progress_path.with_suffix('.tmp')
                    with open(temp_file, 'w') as f:
                        json.dump(progress_data, f)
                    temp_file.replace(progress_path)
                except Exception as e:
                    print(f"[Progress File] Error writing: {e}", flush=True)

            return result

    # Replace tqdm globally
    tqdm_module.tqdm = ProgressFileTqdm
    print(f"[Progress File] tqdm patched successfully", flush=True)

#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    
    # Check for NaN or infinite values
    if torch.isnan(img).any() or torch.isinf(img).any():
        print("Warning: NaN or Inf values detected in image!")
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Handle different channel counts
    if channels == 4:
        print("Warning: 4-channel image detected, using only RGB channels")
        img = img[:, :3, :, :]
        channels = 3
    elif channels == 1:
        # Convert single-channel depth to 3-channel grayscale
        print("Converting single-channel depth to RGB")
        img = img.repeat(1, 3, 1, 1)
        channels = 3
    elif channels != 3:
        print(f"Warning: Unexpected number of channels: {channels}")
    
    if float_to_uint8:
        # Ensure proper clamping before conversion
        img = img.clamp(-1, 1)  # Ensure input is in expected range
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, seeds, pose_cond, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), voxel_res=None, shapes_only=False, white_back=True, trajectory_poses=None, **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)
    
    print(f"Video generation settings:")
    print(f"  Seeds: {seeds}")
    print(f"  Grid: {grid_w}x{grid_h}")
    print(f"  Num keyframes: {num_keyframes}")
    print(f"  Frames per transition: {w_frames}")
    print(f"  Total frames: {num_keyframes * w_frames}")

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

    # Handle DataParallel wrapped model
    G_module = G.module if hasattr(G, 'module') else G
    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G_module.z_dim) for seed in all_seeds])).to(device)
    pose_cond_rad = pose_cond/180*np.pi
    cam2world_pose = LookAtPoseSampler.sample(pose_cond_rad, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)
    ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    _ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    # Use custom voxel resolution if provided, otherwise use defaults
    if voxel_res is not None:
        voxel_resolution = voxel_res
        print(f"Using custom voxel resolution: {voxel_resolution}")
    elif device.type == 'cpu' and gen_shapes:
        voxel_resolution = 256  # Lower resolution for CPU
        print(f"Using reduced voxel resolution {voxel_resolution} for CPU shape generation")
    else:
        voxel_resolution = 512
    
    # Reduce batch size for CPU or lower resolution
    if device.type == 'cpu' or (gen_shapes and voxel_resolution <= 256):
        max_batch = 1000000  # Smaller batch
    else:
        max_batch = 10000000
    # Create video writer only if not shapes_only mode
    video_out = None
    all_frames = []  # Collect frames if video writing fails
    
    if not shapes_only:
        # Clear ALL video_kwargs to prevent any parameter leakage
        bitrate = video_kwargs.get('bitrate', '10M')
        video_kwargs = {}  # Complete reset
        
        try:
            # Use the absolute simplest video writer
            video_out = imageio.get_writer(mp4, fps=60)
            print(f"Video writer created successfully for: {mp4}")
        except Exception as e:
            print(f"Warning: Could not create video writer: {e}")
            print("Will collect frames and try alternative method at the end...")
            video_out = None

    if gen_shapes:
        # Handle single seed case
        if len(all_seeds) > 1:
            outdir = 'interpolation_{}_{}/'.format(all_seeds[0], all_seeds[1])
        else:
            outdir = 'shape_seed_{}/'.format(all_seeds[0])
        os.makedirs(outdir, exist_ok=True)
    all_poses = []

    # Print trajectory info if provided
    if trajectory_poses is not None:
        print(f"Using custom camera trajectory with {len(trajectory_poses)} poses")
        if len(trajectory_poses) != num_keyframes * w_frames:
            print(f"WARNING: Trajectory has {len(trajectory_poses)} poses but video has {num_keyframes * w_frames} frames")
            print(f"         Trajectory will be resampled/repeated to match video length")

    for frame_idx in tqdm.tqdm(range(num_keyframes * w_frames)):
        imgs = []
        synthesis_outputs = []  # Store synthesis outputs for debugging
        for yi in range(grid_h):
            for xi in range(grid_w):
                # Use trajectory if provided, otherwise use default parametric motion
                if trajectory_poses is not None:
                    # Get pose from trajectory (with wrapping if trajectory is shorter)
                    pose_idx = frame_idx % len(trajectory_poses)
                    pose = trajectory_poses[pose_idx]

                    # Extract yaw, pitch, radius from trajectory
                    yaw = pose['yaw']
                    pitch = pose['pitch']
                    radius = pose['radius']

                    # Sample camera pose using trajectory parameters
                    cam2world_pose = LookAtPoseSampler.sample(
                        yaw, pitch, camera_lookat_point, radius=radius, device=device
                    )
                elif cfg == "Head":
                    pitch_range = 0.5
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 2 * 3.14 * frame_idx / (num_keyframes * w_frames),  3.14/2 -0.05 + pitch_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            camera_lookat_point, radius=2.7, device=device)
                else:
                    pitch_range = 0.25
                    yaw_range = 1.5 # 0.35
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            camera_lookat_point, radius=2.7, device=device)
                if not shapes_only:
                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                
                # Generate image only if not shapes_only
                if not shapes_only:
                    entangle = 'camera'
                    if entangle == 'conditioning':
                        c_forward = torch.cat([LookAtPoseSampler.sample(3.14/2,
                                                                        3.14/2,
                                                                        camera_lookat_point,
                                                                        radius=2.7, device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                        w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                        img = G.synthesis(ws=w_c, c=c_forward, noise_mode='const')[image_mode][0]
                    elif entangle == 'camera':
                        # if 'delta_c_temp' in locals():
                        #     delta_c = delta_c_temp
                        # else:
                        #     delta_c = G.t_mapping(zs[yi*grid_w+xi:yi*grid_w+xi+1], c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
                        #     delta_c = torch.squeeze(delta_c, 1)
                        #     delta_c_temp = delta_c
                        # delta_c = G.t_mapping(zs[yi*grid_w+xi:yi*grid_w+xi+1], c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
                        # delta_c = torch.squeeze(delta_c, 1)
                        # c[:,3] += delta_c[:,0]
                        # c[:,7] += delta_c[:,1]
                        # c[:,11] += delta_c[:,2]
                        synthesis_output = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')
                        synthesis_outputs.append(synthesis_output)  # Store for debugging
                        
                        # Debug: Check what synthesis returns
                        if frame_idx == 0 and yi == 0 and xi == 0:
                            print(f"Synthesis output keys: {synthesis_output.keys()}")
                            print(f"Image mode: {image_mode}")
                            if image_mode in synthesis_output:
                                print(f"Image shape from synthesis: {synthesis_output[image_mode].shape}")
                            if 'image_mask' in synthesis_output:
                                print(f"Alpha mask available, shape: {synthesis_output['image_mask'].shape}")
                                mask = synthesis_output['image_mask'][0]
                                print(f"Mask range: [{mask.min().item():.3f}, {mask.max().item():.3f}]")
                        
                        img = synthesis_output[image_mode][0]
                        
                        # Only manually composite if we have a mask and white_back wasn't already handled by the renderer
                        # This is a fallback in case the model doesn't respect the white_back flag
                        if 'image_mask' in synthesis_output and image_mode == 'image' and white_back:
                            # Check if the background is already white by sampling corners
                            corners = [img[:, 0, 0], img[:, 0, -1], img[:, -1, 0], img[:, -1, -1]]
                            avg_corner = torch.stack(corners).mean(0)
                            if avg_corner.mean() < 0.9:  # If corners aren't white, apply manual compositing
                                mask = synthesis_output['image_mask'][0]
                                
                                # Resize mask to match image dimensions
                                if mask.shape[-2:] != img.shape[-2:]:
                                    mask = torch.nn.functional.interpolate(
                                        mask.unsqueeze(0), 
                                        size=img.shape[-2:], 
                                        mode='bilinear', 
                                        align_corners=False
                                    ).squeeze(0)
                                
                                # Ensure mask has 3 channels to match RGB
                                if mask.shape[0] == 1:
                                    mask = mask.repeat(3, 1, 1)
                                
                                # Composite with white background
                                white_bg = torch.ones_like(img)
                                img = img * mask + white_bg * (1 - mask)
                                if frame_idx == 0 and yi == 0 and xi == 0:
                                    print("Applied manual white background compositing")
                    elif entangle == 'both':
                        w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                        if 'delta_c_temp' in locals():
                            delta_c = delta_c_temp
                        else:
                            delta_c = G.t_mapping(zs[yi*grid_w+xi:yi*grid_w+xi+1], c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
                            delta_c = torch.squeeze(delta_c, 1)
                            delta_c_temp = delta_c
                        c[:,3] += delta_c[:,0]
                        c[:,7] += delta_c[:,1]
                        c[:,11] += delta_c[:,2]
                        img = G.synthesis(ws=w_c, c=c[0:1], noise_mode='const')[image_mode][0]

                    if image_mode == 'image_depth':
                        img = -img
                        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1
                        # Expand depth to 3 channels for RGB video
                        if img.shape[0] == 1:
                            img = img.repeat(3, 1, 1)
                    else:
                        img = img.clamp(min=-1,max=1)
                    
                    # Debug: Check image properties
                    if frame_idx == 0:
                        print(f"Generated image shape: {img.shape}")
                        print(f"Generated image range: [{img.min().item():.3f}, {img.max().item():.3f}]")
                        print(f"Image dtype: {img.dtype}")
                        
                        # Check background values
                        corners = [img[:, 0, 0], img[:, 0, -1], img[:, -1, 0], img[:, -1, -1]]
                        corner_values = torch.stack(corners).cpu().numpy()
                        print(f"Corner pixel values (should be ~1.0 for white):")
                        for i, corner in enumerate(['top-left', 'top-right', 'bottom-left', 'bottom-right']):
                            print(f"  {corner}: {corner_values[i]}")

                    imgs.append(img)

                if gen_shapes:
                    # generate shapes
                    print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))
                    
                    samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=G_module.rendering_kwargs['box_warp'])
                    samples = samples.to(device)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                    transformed_ray_directions_expanded[..., -1] = -1

                    head = 0
                    with tqdm.tqdm(total = samples.shape[1]) as pbar:
                        with torch.no_grad():
                            while head < samples.shape[1]:
                                # Only set CUDA seed if using CUDA
                                if device.type == 'cuda':
                                    torch.manual_seed(0)
                                sigma = G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], w.unsqueeze(0), truncation_psi=psi, noise_mode='const')['sigma']
                                sigmas[:, head:head+max_batch] = sigma
                                head += max_batch
                                pbar.update(max_batch)

                    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                    sigmas = np.flip(sigmas, 0)
                    
                    pad = int(30 * voxel_resolution / 256)
                    pad_top = int(38 * voxel_resolution / 256)
                    sigmas[:pad] = 0
                    sigmas[-pad:] = 0
                    sigmas[:, :pad] = 0
                    sigmas[:, -pad_top:] = 0
                    sigmas[:, :, :pad] = 0
                    sigmas[:, :, -pad:] = 0

                    output_ply = True
                    if output_ply:
                        from shape_utils import convert_sdf_samples_to_ply
                        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)
                    else: # output mrc
                        with mrcfile.new_mmap(outdir + f'{frame_idx:04d}_shape.mrc', overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                            mrc.data[:] = sigmas

        if not shapes_only and len(imgs) > 0:
            # Debug frame composition
            if frame_idx == 0:
                print(f"Number of images to stack: {len(imgs)}")
                print(f"Grid dimensions: {grid_w}x{grid_h}")
                for i, img in enumerate(imgs):
                    print(f"  Image {i} shape: {img.shape}")
            
            frame = layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h)
            
            if frame_idx == 0:
                print(f"Frame shape after layout_grid: {frame.shape}")
                print(f"Frame dtype: {frame.dtype}")
                print(f"Frame range: [{frame.min()}, {frame.max()}]")
                
                # Save debug frame
                debug_path = mp4.replace('.mp4', '_debug_frame0.png')
                imageio.imwrite(debug_path, frame)
                print(f"Debug: Saved first frame to {debug_path}")
                
                # Also save the raw tensor before conversion
                if len(imgs) > 0:
                    raw_img = imgs[0].cpu()
                    raw_debug_path = mp4.replace('.mp4', '_debug_raw0.png')
                    # Convert from [-1,1] to [0,1] for saving
                    raw_img_save = (raw_img.clamp(-1, 1) + 1) / 2
                    raw_img_save = raw_img_save.permute(1, 2, 0).numpy()
                    raw_img_save = (raw_img_save * 255).astype('uint8')
                    imageio.imwrite(raw_debug_path, raw_img_save)
                    print(f"Debug: Saved raw image to {raw_debug_path}")
                    
                    # Save mask if available
                    if len(synthesis_outputs) > 0 and 'image_mask' in synthesis_outputs[0]:
                        mask = synthesis_outputs[0]['image_mask'][0].cpu()
                        mask_debug_path = mp4.replace('.mp4', '_debug_mask0.png')
                        
                        # Resize mask to match image dimensions for saving
                        if mask.shape[-2:] != (512, 512):  # Assuming image is 512x512
                            mask = torch.nn.functional.interpolate(
                                mask.unsqueeze(0), 
                                size=(512, 512), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                        
                        # Convert mask to [0,1] range
                        mask_save = mask.clamp(0, 1)
                        if mask_save.shape[0] == 1:
                            mask_save = mask_save.repeat(3, 1, 1)
                        mask_save = mask_save.permute(1, 2, 0).numpy()
                        mask_save = (mask_save * 255).astype('uint8')
                        imageio.imwrite(mask_debug_path, mask_save)
                        print(f"Debug: Saved mask to {mask_debug_path} (resized from {synthesis_outputs[0]['image_mask'][0].shape[-2:]} to 512x512)")
            
            if video_out is not None:
                try:
                    video_out.append_data(frame)
                except Exception as e:
                    print(f"Error appending frame to video: {e}")
                    print("Collecting frames for alternative save method...")
                    all_frames.append(frame)
                    if video_out is not None:
                        video_out.close()
                        video_out = None
            else:
                all_frames.append(frame)
    if video_out is not None:
        video_out.close()
        print(f"Video saved successfully to: {mp4}")
    elif len(all_frames) > 0 and not shapes_only:
        # Fallback: save frames as images
        print(f"Video writing failed. Saving {len(all_frames)} frames as images...")
        frames_dir = mp4.replace('.mp4', '_frames')
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(all_frames):
            imageio.imwrite(os.path.join(frames_dir, f'frame_{i:04d}.png'), frame)
        print(f"Frames saved to: {frames_dir}")
        print(f"You can create a video manually with: ffmpeg -i {frames_dir}/frame_%04d.png -c:v libx264 -r 60 {mp4}")
    
    # Save trajectory if we have poses
    if not shapes_only and len(all_poses) > 0:
        all_poses = np.stack(all_poses)
        if gen_shapes:
            print(f"Camera trajectory shape: {all_poses.shape}")
            with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
                np.save(f, all_poses)

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename (default: models/spherehead-ckpt-025000.pkl)', default='../models/spherehead-ckpt-025000.pkl', show_default=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--outdir', help='Output path', type=str, required=False)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=240)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi: controls quality vs diversity trade-off. Lower values (0.5-0.8) produce higher quality, more typical results. 1.0 = full diversity but more artifacts. 0 = average face.', default=0.6, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff: number of layers to apply truncation to (lower layers control coarse features)', default=14, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
@click.option('--cfg', help='Config (FFHQ, Cats, Head)', type=click.Choice(['FFHQ', 'Cats', 'Head']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode: image=RGB, image_depth=depth map only, image_raw=raw neural rendering', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=1, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--interpolate', type=bool, help='Interpolate between seeds', default=True, show_default=True)
@click.option('--pose_cond', type=int, help='camera conditioned pose angle', default=90, show_default=True)
@click.option('--gpu', type=str, help='GPU device(s) to use (e.g., "0", "0,1,2", "all", or "cpu")', default=None, show_default=True)
@click.option('--voxel_res', type=int, help='Voxel resolution for 3D shape extraction', default=None, show_default=True)
@click.option('--shapes_only', type=bool, help='Only generate 3D shapes, skip video generation', default=False, show_default=True)
@click.option('--white_back', type=bool, help='Force white background in rendered images', default=True, show_default=True)
@click.option('--disable_background', type=bool, help='Disable neural background rendering if present', default=False, show_default=True)
@click.option('--progress-file', type=str, help='Enable real-time progress monitoring by writing to JSON file (for GUI integration)', default=None, show_default=True)
@click.option('--camera-trajectory', type=str, help='Path to camera trajectory JSON file for custom camera paths (GUI integration)', default=None, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    outdir: str,
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    shapes: bool,
    interpolate: bool,
    pose_cond: int,
    gpu: Optional[str],
    voxel_res: Optional[int],
    shapes_only: bool,
    white_back: bool,
    disable_background: bool,
    progress_file: Optional[str],
    camera_trajectory: Optional[str],
):
    """Render a latent vector interpolation video.

    Args:
        progress_file: Optional path to JSON file for real-time progress monitoring

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate depth map video only
    python gen_videos.py --seeds=0 --image_mode=image_depth --outdir=output/depth

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    # Setup progress file monitoring if requested
    if progress_file:
        setup_progress_file_monitoring(progress_file)

    # Load camera trajectory if provided
    trajectory_poses = None
    if camera_trajectory:
        print(f'Loading camera trajectory from "{camera_trajectory}"...')
        try:
            with open(camera_trajectory, 'r') as f:
                trajectory_data = json.load(f)

            # Validate trajectory format
            if 'num_frames' not in trajectory_data or 'poses' not in trajectory_data:
                raise ValueError("Invalid trajectory format: missing 'num_frames' or 'poses'")

            trajectory_poses = trajectory_data['poses']
            print(f'  Loaded {len(trajectory_poses)} camera poses')

            # Validate pose format
            for i, pose in enumerate(trajectory_poses):
                if not all(k in pose for k in ['yaw', 'pitch', 'radius']):
                    raise ValueError(f"Pose {i} missing required fields (yaw, pitch, radius)")

            print(f'  Trajectory validation successful')
        except Exception as e:
            print(f'ERROR: Failed to load camera trajectory: {e}')
            raise

    print('Loading networks from "%s"...' % network_pkl)
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        # Clear all GPU cache
        torch.cuda.empty_cache()
        
        # If specific GPUs are requested, try to clear their memory
        if gpu is not None and gpu != 'cpu':
            try:
                if gpu == 'all':
                    gpu_list = list(range(torch.cuda.device_count()))
                elif ',' in gpu:
                    gpu_list = [int(x.strip()) for x in gpu.split(',')]
                else:
                    gpu_list = [int(gpu)]
                
                print(f"Clearing GPU memory on devices: {gpu_list}")
                for gpu_id in gpu_list:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(gpu_id)
                    # Force garbage collection
                    import gc
                    gc.collect()
                print("GPU memory cleared")
            except Exception as e:
                print(f"Warning: Could not clear GPU memory: {e}")
    
    # Check CUDA availability and set device
    print('\n' + '='*60)
    print('DEVICE INFORMATION:')
    print('='*60)
    
    device_ids = []
    use_multi_gpu = False
    
    if gpu == 'cpu':
        device = torch.device('cpu')
        print(f'Using CPU (as requested)')
        print('Warning: Running on CPU will be very slow!')
    elif torch.cuda.is_available():
        if gpu is not None:
            if gpu == 'all':
                # Use all available GPUs
                device_ids = list(range(torch.cuda.device_count()))
                use_multi_gpu = True
                device = torch.device('cuda:0')  # Primary device
                print(f'Using all {len(device_ids)} GPUs: {device_ids}')
            elif ',' in gpu:
                # Use specific GPUs
                device_ids = [int(x.strip()) for x in gpu.split(',')]
                use_multi_gpu = True
                device = torch.device(f'cuda:{device_ids[0]}')  # Primary device
                print(f'Using specified GPUs: {device_ids}')
            else:
                # Use single specified GPU
                device = torch.device(f'cuda:{gpu}')
                print(f'Using specified GPU: {device}')
        else:
            # Find GPU with most free memory
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                gpu_memory = []
                for line in result.stdout.strip().split('\n'):
                    idx, mem_free = line.split(', ')
                    gpu_memory.append((int(idx), int(mem_free)))
                # Sort by free memory (descending)
                gpu_memory.sort(key=lambda x: x[1], reverse=True)
                
                # Find GPUs with at least 10GB free
                min_required_mb = 10000  # 10GB in MB
                suitable_gpus = [(idx, mem) for idx, mem in gpu_memory if mem >= min_required_mb]
                
                if suitable_gpus:
                    best_gpu = suitable_gpus[0][0]
                    device = torch.device(f'cuda:{best_gpu}')
                    print(f'Auto-selected GPU with most free memory: cuda:{best_gpu} ({suitable_gpus[0][1]} MB free)')
                else:
                    print('No GPU has enough free memory (>10GB). Falling back to CPU.')
                    device = torch.device('cpu')
                    print('Warning: Running on CPU will be very slow!')
            else:
                device = torch.device('cuda:0')
                print(f'Using default device: {device}')
        
        if device.type == 'cuda':
            print(f'CUDA version: {torch.version.cuda}')
            print(f'Number of CUDA devices: {torch.cuda.device_count()}')
            
            if use_multi_gpu:
                print('\nGPU Memory Status:')
                for gpu_id in device_ids:
                    torch.cuda.set_device(gpu_id)
                    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    allocated_mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    free_mem = (torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)) / 1024**3
                    print(f'GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} - Total: {total_mem:.2f} GB, Free: {free_mem:.2f} GB')
                
                # Check if we have enough memory
                min_required_gb = 10  # Estimate for model loading
                suitable_gpus = [gpu_id for gpu_id in device_ids 
                                 if (torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)) / 1024**3 > min_required_gb]
                
                if not suitable_gpus:
                    print(f'\nWarning: No GPUs have enough free memory (>{min_required_gb} GB). Model loading may fail.')
                    print('Consider using --gpu cpu or specify GPUs with more free memory.')
                else:
                    print(f'\nGPUs with sufficient memory: {suitable_gpus}')
            else:
                print(f'CUDA device: {torch.cuda.get_device_name(device)}')
                torch.cuda.set_device(device)
                print(f'Selected GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB total')
    else:
        device = torch.device('cpu')
        print(f'CUDA not available, using device: {device}')
        print('Warning: Running on CPU will be very slow!')
    print('='*60 + '\n')
    
    try:
        with dnnlib.util.open_url(network_pkl) as f:
            print('Opening pickle file...')
            data = load_network_pkl_cpu_safe(f, device)
            print('Pickle loaded successfully, extracting G_ema...')
            G = data['G_ema'].to(device) # type: ignore
            
            if use_multi_gpu:
                # Wrap model with DataParallel for multi-GPU
                G = torch.nn.DataParallel(G, device_ids=device_ids)
                print(f'Model wrapped with DataParallel for GPUs: {device_ids}')
            else:
                print(f'Model loaded to device successfully: {device}')
    except Exception as e:
        print(f'Error loading model: {type(e).__name__}: {str(e)}')
        import traceback
        traceback.print_exc()
        raise

    # Handle DataParallel wrapped model
    if use_multi_gpu and isinstance(G, torch.nn.DataParallel):
        G_module = G.module
    else:
        G_module = G
    
    G_module.rendering_kwargs['depth_resolution'] = int(G_module.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G_module.rendering_kwargs['depth_resolution_importance'] = int(G_module.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G_module.neural_rendering_resolution = nrr

    G_new = TriPlaneGenerator(*G_module.init_args, **G_module.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G_module, G_new, require_all=True)
    G_new.neural_rendering_resolution = G_module.neural_rendering_resolution
    G_new.rendering_kwargs = G_module.rendering_kwargs
    
    print(f"Model's neural rendering resolution: {G_module.neural_rendering_resolution}")
    print(f"Model's rendering kwargs:")
    for k, v in G_module.rendering_kwargs.items():
        print(f"  {k}: {v}")
    
    # Disable neural background if requested
    if disable_background and 'use_background' in G_module.rendering_kwargs:
        print("Disabling neural background rendering...")
        G_module.rendering_kwargs['use_background'] = False
        G_new.rendering_kwargs['use_background'] = False
    
    # Set white background if requested
    if white_back:
        print("Enabling white background...")
        G_module.rendering_kwargs['white_back'] = True
        G_new.rendering_kwargs['white_back'] = True
    
    if use_multi_gpu:
        G = torch.nn.DataParallel(G_new, device_ids=device_ids)
    else:
        G = G_new

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    network_pkl = os.path.basename(network_pkl)
    
    output_mp4 = os.path.splitext(network_pkl)[0] + '_' + str(pose_cond) + '.mp4'
    output = os.path.join(outdir, output_mp4)
    
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    if interpolate:
        # Clear GPU cache before video generation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if shapes_only:
            print(f'\nGenerating 3D shapes only (no video) on device: {device}')
        else:
            print(f'\nStarting video generation on device: {device}')
        gen_interp_video(G=G, mp4=output, pose_cond = pose_cond, bitrate='100M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes, device=device, voxel_res=voxel_res, shapes_only=shapes_only, white_back=white_back, trajectory_poses=trajectory_poses)
    else:
        os.makedirs(output)
        for seed in seeds:
            output = os.path.join(output, f'{seed}.mp4')
            seeds_ = [seed]
            gen_interp_video(G=G, mp4=output, pose_cond = pose_cond, bitrate='30M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds_, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, device=device, trajectory_poses=trajectory_poses)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

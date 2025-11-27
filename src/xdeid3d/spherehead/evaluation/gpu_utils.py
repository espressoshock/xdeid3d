"""
GPU utility functions for selecting the best available GPU.
Shared across all pose_robustness scripts.
"""

import torch
import subprocess


def get_best_gpu_device():
    """
    Automatically select the GPU with the most free memory.
    Falls back to CPU if no GPU has enough memory (>10GB).
    
    Returns:
        torch.device: The selected device
    """
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        print('Warning: Running on CPU will be very slow!')
        return torch.device('cpu')
    
    try:
        # Query GPU memory using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], 
            capture_output=True, 
            text=True,
            check=True
        )
        
        # Parse GPU memory information
        gpu_memory = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():  # Skip empty lines
                parts = line.split(', ')
                if len(parts) == 2:
                    idx, mem_free = parts
                    gpu_memory.append((int(idx), int(mem_free)))
        
        if not gpu_memory:
            print('No GPU information found, using cuda:0')
            return torch.device('cuda:0')
        
        # Sort by free memory (descending)
        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        
        # Print available GPUs
        print('Available GPUs:')
        for idx, mem in gpu_memory:
            print(f'  GPU {idx}: {mem} MB free')
        
        # Find GPUs with at least 10GB free
        min_required_mb = 10000  # 10GB in MB
        suitable_gpus = [(idx, mem) for idx, mem in gpu_memory if mem >= min_required_mb]
        
        if suitable_gpus:
            best_gpu = suitable_gpus[0][0]
            device = torch.device(f'cuda:{best_gpu}')
            print(f'Auto-selected GPU with most free memory: cuda:{best_gpu} ({suitable_gpus[0][1]} MB free)')
            return device
        else:
            # Try with lower requirement
            min_required_mb = 5000  # 5GB in MB
            suitable_gpus = [(idx, mem) for idx, mem in gpu_memory if mem >= min_required_mb]
            
            if suitable_gpus:
                best_gpu = suitable_gpus[0][0]
                device = torch.device(f'cuda:{best_gpu}')
                print(f'Auto-selected GPU: cuda:{best_gpu} ({suitable_gpus[0][1]} MB free)')
                print('Warning: GPU has less than 10GB free, may run out of memory')
                return device
            else:
                print('No GPU has enough free memory (>5GB). Falling back to CPU.')
                print('Warning: Running on CPU will be very slow!')
                return torch.device('cpu')
                
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # nvidia-smi not available or failed
        print(f'Could not query GPU memory: {e}')
        print('Using default CUDA device')
        return torch.device('cuda:0')
    except Exception as e:
        print(f'Unexpected error querying GPU: {e}')
        print('Using default CUDA device')
        return torch.device('cuda:0')


def clear_gpu_memory(device):
    """Clear GPU memory cache if using CUDA."""
    if device.type == 'cuda':
        try:
            # First empty the cache
            torch.cuda.empty_cache()
            # Only synchronize if we can
            if torch.cuda.is_initialized():
                # Set the device first
                torch.cuda.set_device(device)
                # Then synchronize only if there's no error
                try:
                    torch.cuda.synchronize(device)
                except RuntimeError:
                    # If synchronize fails, just continue
                    pass
            print(f'Cleared GPU memory on {device}')
        except Exception as e:
            print(f'Warning: Could not fully clear GPU memory: {e}')
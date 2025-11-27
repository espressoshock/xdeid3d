"""
GPU and device management utilities.

Provides functions for GPU selection, memory management,
and device handling for PyTorch operations.
"""

import gc
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

__all__ = [
    "get_device",
    "get_best_gpu",
    "get_gpu_memory_info",
    "clear_gpu_memory",
    "DeviceManager",
    "GPUInfo",
]


@dataclass
class GPUInfo:
    """Information about a GPU device.

    Attributes:
        index: GPU index
        name: Device name
        total_memory: Total memory in bytes
        free_memory: Free memory in bytes
        used_memory: Used memory in bytes
        utilization: GPU utilization percentage (0-100)
    """
    index: int
    name: str
    total_memory: int
    free_memory: int
    used_memory: int
    utilization: float = 0.0

    @property
    def free_memory_gb(self) -> float:
        """Free memory in GB."""
        return self.free_memory / (1024 ** 3)

    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory / (1024 ** 3)

    @property
    def used_memory_gb(self) -> float:
        """Used memory in GB."""
        return self.used_memory / (1024 ** 3)

    @property
    def memory_percent(self) -> float:
        """Memory usage percentage."""
        if self.total_memory == 0:
            return 0.0
        return (self.used_memory / self.total_memory) * 100


def _check_torch_cuda() -> bool:
    """Check if PyTorch CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(
    device: Optional[str] = None,
    fallback_cpu: bool = True,
) -> str:
    """Get the device to use for computation.

    Args:
        device: Explicit device string (cpu, cuda, cuda:0, etc.)
            If None, auto-selects best available device.
        fallback_cpu: Fall back to CPU if CUDA unavailable

    Returns:
        Device string (e.g., "cuda:0" or "cpu")

    Example:
        >>> device = get_device()  # Auto-select
        >>> device = get_device("cuda:1")  # Explicit GPU
        >>> device = get_device("cpu")  # Force CPU
    """
    if device is not None:
        device = device.lower()

        if device == "cpu":
            return "cpu"

        if device.startswith("cuda"):
            if not _check_torch_cuda():
                if fallback_cpu:
                    return "cpu"
                raise RuntimeError("CUDA requested but not available")

            # Validate specific GPU index if provided
            if ":" in device:
                try:
                    import torch
                    idx = int(device.split(":")[1])
                    if idx >= torch.cuda.device_count():
                        if fallback_cpu:
                            return "cpu"
                        raise RuntimeError(f"GPU {idx} not available")
                except ValueError:
                    pass

            return device

        # Unknown device
        if fallback_cpu:
            return "cpu"
        raise ValueError(f"Unknown device: {device}")

    # Auto-select
    if _check_torch_cuda():
        # Select GPU with most free memory
        best_gpu = get_best_gpu()
        if best_gpu is not None:
            return f"cuda:{best_gpu}"
        return "cuda:0"

    return "cpu"


def get_best_gpu(
    min_free_memory_gb: float = 1.0,
    prefer_empty: bool = True,
) -> Optional[int]:
    """Get the best GPU based on available memory.

    Args:
        min_free_memory_gb: Minimum free memory required
        prefer_empty: Prefer GPUs with no active processes

    Returns:
        GPU index or None if no suitable GPU found

    Example:
        >>> gpu_idx = get_best_gpu(min_free_memory_gb=4.0)
        >>> if gpu_idx is not None:
        ...     device = f"cuda:{gpu_idx}"
    """
    if not _check_torch_cuda():
        return None

    gpus = get_gpu_memory_info()

    if not gpus:
        return None

    min_free_bytes = min_free_memory_gb * (1024 ** 3)

    # Filter by minimum memory
    candidates = [
        gpu for gpu in gpus
        if gpu.free_memory >= min_free_bytes
    ]

    if not candidates:
        # Fall back to GPU with most free memory
        candidates = gpus

    # Sort by free memory (descending)
    candidates.sort(key=lambda g: g.free_memory, reverse=True)

    if prefer_empty:
        # Prefer GPUs with low utilization
        empty = [g for g in candidates if g.utilization < 10]
        if empty:
            return empty[0].index

    return candidates[0].index


def get_gpu_memory_info() -> List[GPUInfo]:
    """Get memory information for all available GPUs.

    Returns:
        List of GPUInfo objects

    Example:
        >>> for gpu in get_gpu_memory_info():
        ...     print(f"GPU {gpu.index}: {gpu.free_memory_gb:.1f} GB free")
    """
    if not _check_torch_cuda():
        return []

    import torch

    gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i)

            # Estimate free memory
            free = total - reserved

            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory=total,
                free_memory=free,
                used_memory=allocated,
            ))
        except Exception:
            continue

    # Try to get more accurate info using nvidia-smi
    try:
        gpus = _get_gpu_info_nvml(gpus)
    except Exception:
        pass

    return gpus


def _get_gpu_info_nvml(existing: List[GPUInfo]) -> List[GPUInfo]:
    """Get GPU info using NVML (more accurate than PyTorch)."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except (ImportError, Exception):
        return existing

    try:
        device_count = pynvml.nvmlDeviceGetCount()

        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = 0

            gpus.append(GPUInfo(
                index=i,
                name=name,
                total_memory=mem_info.total,
                free_memory=mem_info.free,
                used_memory=mem_info.used,
                utilization=gpu_util,
            ))

        pynvml.nvmlShutdown()
        return gpus

    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return existing


def clear_gpu_memory(device: Optional[str] = None) -> None:
    """Clear GPU memory cache.

    Args:
        device: Specific device to clear, or None for all

    Example:
        >>> clear_gpu_memory()  # Clear all
        >>> clear_gpu_memory("cuda:0")  # Clear specific GPU
    """
    gc.collect()

    if not _check_torch_cuda():
        return

    import torch

    if device is None:
        # Clear all GPUs
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


class DeviceManager:
    """Context manager for device management.

    Handles device selection, memory monitoring, and cleanup.

    Args:
        device: Device string or None for auto-selection
        clear_on_exit: Clear GPU cache on context exit
        min_memory_gb: Minimum free memory for GPU selection

    Example:
        >>> with DeviceManager("cuda:0") as dm:
        ...     tensor = torch.zeros(1000, device=dm.device)
        ...     print(f"Memory: {dm.memory_used_gb:.2f} GB")

        >>> with DeviceManager() as dm:  # Auto-select
        ...     model = model.to(dm.device)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        clear_on_exit: bool = True,
        min_memory_gb: float = 1.0,
    ):
        self._requested_device = device
        self._device = None
        self._clear_on_exit = clear_on_exit
        self._min_memory_gb = min_memory_gb
        self._initial_memory = 0

    @property
    def device(self) -> str:
        """Get the active device."""
        if self._device is None:
            self._device = get_device(self._requested_device)
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self.device.startswith("cuda")

    @property
    def memory_used_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if not self.is_cuda or not _check_torch_cuda():
            return 0.0

        import torch
        return torch.cuda.memory_allocated(self.device) / (1024 ** 3)

    @property
    def memory_reserved_gb(self) -> float:
        """Get current GPU memory reserved in GB."""
        if not self.is_cuda or not _check_torch_cuda():
            return 0.0

        import torch
        return torch.cuda.memory_reserved(self.device) / (1024 ** 3)

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        clear_gpu_memory(self.device if self.is_cuda else None)

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if self.is_cuda and _check_torch_cuda():
            import torch
            torch.cuda.synchronize(self.device)

    def __enter__(self) -> "DeviceManager":
        """Enter context."""
        # Initialize device
        self._device = get_device(
            self._requested_device,
            fallback_cpu=True,
        )

        # Record initial memory
        if self.is_cuda and _check_torch_cuda():
            import torch
            self._initial_memory = torch.cuda.memory_allocated(self.device)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        if self._clear_on_exit:
            self.clear_cache()

    def get_info(self) -> Dict:
        """Get device information.

        Returns:
            Dictionary with device info
        """
        info = {
            "device": self.device,
            "is_cuda": self.is_cuda,
        }

        if self.is_cuda and _check_torch_cuda():
            import torch
            info.update({
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(self.device),
                "memory_allocated_gb": self.memory_used_gb,
                "memory_reserved_gb": self.memory_reserved_gb,
            })

        return info


def set_cuda_visible_devices(devices: Union[int, List[int], str]) -> None:
    """Set CUDA_VISIBLE_DEVICES environment variable.

    Args:
        devices: Device index, list of indices, or comma-separated string

    Example:
        >>> set_cuda_visible_devices(0)  # Only GPU 0
        >>> set_cuda_visible_devices([0, 2])  # GPUs 0 and 2
        >>> set_cuda_visible_devices("0,1")  # GPUs 0 and 1
    """
    if isinstance(devices, int):
        devices_str = str(devices)
    elif isinstance(devices, list):
        devices_str = ",".join(map(str, devices))
    else:
        devices_str = str(devices)

    os.environ["CUDA_VISIBLE_DEVICES"] = devices_str


def get_torch_device(device_str: str) -> "torch.device":
    """Convert device string to torch.device.

    Args:
        device_str: Device string (cpu, cuda, cuda:0, etc.)

    Returns:
        torch.device object
    """
    import torch
    return torch.device(device_str)

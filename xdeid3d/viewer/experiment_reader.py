"""
Experiment reader for loading and parsing evaluation results.

Scans experiment directories and provides structured access to
results, metrics, and media files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["ExperimentReader"]


class ExperimentReader:
    """Reader for experiment results.

    Scans a directory structure for experiments organized by
    configuration and seed, providing access to metrics and media.

    Args:
        experiments_dir: Root directory containing experiments

    Expected directory structure:
        experiments_dir/
            cfg_name/
                seed_0/
                    metrics.json
                    results.npz
                    videos/
                    images/
                seed_1/
                    ...
    """

    def __init__(self, experiments_dir: str):
        self.experiments_dir = Path(experiments_dir)
        self._cache = {}

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get list of all experiments.

        Returns:
            List of experiment info dictionaries
        """
        experiments = []

        if not self.experiments_dir.exists():
            return experiments

        # Scan for config directories
        for cfg_dir in self.experiments_dir.iterdir():
            if not cfg_dir.is_dir():
                continue

            cfg_name = cfg_dir.name

            # Scan for seed directories
            for seed_dir in cfg_dir.iterdir():
                if not seed_dir.is_dir():
                    continue

                # Parse seed from directory name
                try:
                    # Handle formats like "seed_0", "0", "run_0"
                    seed_name = seed_dir.name
                    if seed_name.startswith("seed_"):
                        seed = int(seed_name[5:])
                    elif seed_name.startswith("run_"):
                        seed = int(seed_name[4:])
                    else:
                        seed = int(seed_name)
                except ValueError:
                    continue

                # Get experiment info
                info = self._get_experiment_basic_info(cfg_name, seed, seed_dir)
                if info:
                    experiments.append(info)

        # Sort by config and seed
        experiments.sort(key=lambda x: (x["cfg"], x["seed"]))

        return experiments

    def _get_experiment_basic_info(
        self,
        cfg: str,
        seed: int,
        exp_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Get basic info for an experiment."""
        info = {
            "cfg": cfg,
            "seed": seed,
            "path": str(exp_dir.relative_to(self.experiments_dir)),
            "has_metrics": False,
            "has_video": False,
            "has_heatmap": False,
        }

        # Check for metrics
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            info["has_metrics"] = True

        results_file = exp_dir / "evaluation_results.json"
        if results_file.exists():
            info["has_metrics"] = True

        # Check for video
        video_patterns = ["*.mp4", "video.mp4", "output.mp4"]
        for pattern in video_patterns:
            if list(exp_dir.glob(pattern)):
                info["has_video"] = True
                break

        # Check for heatmap mesh
        ply_files = list(exp_dir.glob("*.ply"))
        if ply_files:
            info["has_heatmap"] = True

        return info

    def get_experiment_info(self, cfg: str, seed: int) -> Optional[Dict[str, Any]]:
        """Get detailed info for a specific experiment.

        Args:
            cfg: Configuration name
            seed: Random seed

        Returns:
            Experiment info dictionary or None if not found
        """
        # Find experiment directory
        exp_dir = self._find_experiment_dir(cfg, seed)
        if exp_dir is None:
            return None

        info = {
            "cfg": cfg,
            "seed": seed,
            "path": str(exp_dir.relative_to(self.experiments_dir)),
        }

        # Load metrics
        metrics = self._load_metrics(exp_dir)
        if metrics:
            info["metrics"] = metrics

        # Find media files
        info["media"] = self._find_media_files(exp_dir)

        # Get summary statistics
        if "metrics" in info and "frame_metrics" in info["metrics"]:
            info["summary"] = self._compute_summary(info["metrics"]["frame_metrics"])

        return info

    def _find_experiment_dir(self, cfg: str, seed: int) -> Optional[Path]:
        """Find experiment directory for cfg/seed."""
        cfg_dir = self.experiments_dir / cfg

        if not cfg_dir.exists():
            return None

        # Try different naming conventions
        for name in [f"seed_{seed}", f"run_{seed}", str(seed)]:
            exp_dir = cfg_dir / name
            if exp_dir.exists():
                return exp_dir

        return None

    def _load_metrics(self, exp_dir: Path) -> Optional[Dict[str, Any]]:
        """Load metrics from experiment directory."""
        # Try different metric file names
        for name in ["metrics.json", "evaluation_results.json", "results.json"]:
            metrics_file = exp_dir / name
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    continue

        # Try NPZ format
        npz_file = exp_dir / "results.npz"
        if npz_file.exists():
            try:
                data = np.load(npz_file)
                return {key: data[key].tolist() for key in data.files}
            except Exception:
                pass

        return None

    def _find_media_files(self, exp_dir: Path) -> Dict[str, List[str]]:
        """Find all media files in experiment directory."""
        media = {
            "videos": [],
            "images": [],
            "meshes": [],
            "data": [],
        }

        rel_path = exp_dir.relative_to(self.experiments_dir)

        # Videos
        for ext in [".mp4", ".avi", ".webm"]:
            for f in exp_dir.rglob(f"*{ext}"):
                media["videos"].append(str(f.relative_to(self.experiments_dir)))

        # Images
        for ext in [".png", ".jpg", ".jpeg"]:
            for f in exp_dir.rglob(f"*{ext}"):
                media["images"].append(str(f.relative_to(self.experiments_dir)))

        # Meshes
        for f in exp_dir.rglob("*.ply"):
            media["meshes"].append(str(f.relative_to(self.experiments_dir)))

        # Data files
        for ext in [".json", ".npz", ".csv"]:
            for f in exp_dir.rglob(f"*{ext}"):
                media["data"].append(str(f.relative_to(self.experiments_dir)))

        return media

    def _compute_summary(self, frame_metrics: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics from frame metrics."""
        if not frame_metrics:
            return {}

        summary = {}

        # Find numeric metrics
        sample = frame_metrics[0]
        for key, value in sample.items():
            if isinstance(value, (int, float)) and key not in ["frame_idx", "timestamp"]:
                values = [m.get(key) for m in frame_metrics if m.get(key) is not None]
                if values:
                    arr = np.array(values)
                    summary[key] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "median": float(np.median(arr)),
                    }

        return summary

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary across all experiments.

        Returns:
            Dictionary with aggregated metrics statistics
        """
        all_metrics = {}
        experiment_count = 0

        for exp in self.get_all_experiments():
            if not exp.get("has_metrics"):
                continue

            info = self.get_experiment_info(exp["cfg"], exp["seed"])
            if info is None or "summary" not in info:
                continue

            experiment_count += 1

            for metric, stats in info["summary"].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(stats["mean"])

        # Compute overall statistics
        summary = {
            "experiment_count": experiment_count,
            "metrics": {},
        }

        for metric, values in all_metrics.items():
            arr = np.array(values)
            summary["metrics"][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        return summary

    def get_experiment_comparisons(
        self,
        cfg: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get experiments grouped for comparison.

        Args:
            cfg: Filter by configuration (optional)

        Returns:
            Dictionary grouping experiments by config
        """
        experiments = self.get_all_experiments()

        if cfg:
            experiments = [e for e in experiments if e["cfg"] == cfg]

        # Group by config
        groups = {}
        for exp in experiments:
            c = exp["cfg"]
            if c not in groups:
                groups[c] = []
            groups[c].append(exp)

        return {
            "groups": groups,
            "configs": list(groups.keys()),
            "total": len(experiments),
        }

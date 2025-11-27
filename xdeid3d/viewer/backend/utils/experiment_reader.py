"""
Utility to read and analyze experiment data from the experiments directory
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .metrics_extractor import MetricsExtractor


class ExperimentReader:
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = experiments_dir
        self.metrics_extractor = MetricsExtractor(experiments_dir)
        
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments organized by cfg and seed"""
        experiments = []
        
        if not os.path.exists(self.experiments_dir):
            return experiments
            
        # Scan for cfg directories
        for cfg in ["Head", "FFHQ", "Cats"]:
            cfg_path = os.path.join(self.experiments_dir, cfg)
            if not os.path.exists(cfg_path):
                continue
                
            # Scan for seed directories
            for seed_dir in os.listdir(cfg_path):
                seed_path = os.path.join(cfg_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue
                    
                try:
                    seed = int(seed_dir)
                except ValueError:
                    continue
                    
                experiment_info = self.get_experiment_info(cfg, seed)
                if experiment_info:
                    experiments.append(experiment_info)
                    
        return sorted(experiments, key=lambda x: (x['cfg'], x['seed']))
    
    def get_experiment_info(self, cfg: str, seed: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific experiment"""
        exp_path = os.path.join(self.experiments_dir, cfg, str(seed))
        
        if not os.path.exists(exp_path):
            return None
            
        info = {
            'cfg': cfg,
            'seed': seed,
            'path': exp_path,
            'files': {},
            'metrics': {},
            'metadata': {}
        }
        
        # Check for standard files
        file_patterns = {
            'original_video': 'original_video.mp4',
            'original_video_depth': 'original_video_depth.mp4',
            'anonymized_video': 'anonymized_video.mp4',
            'guard_video': 'guard_*.mp4',
            'comparison_video': '*_comparison.mp4',
            'depth_video': '*_depth.mp4',
            'heatmap_video': '*_heatmap_mesh_video.mp4',
            'mesh_video': '3dmesh_visualization.mp4',
            'mesh_video_compact': '3dmesh_compact_visualization.mp4',
            'grid_image': '*_grid.png',
            'depth_grid': '*_depth_grid.png',
            'heatmap_grid': '*_heatmap_mesh_video_grid.png',
            'depth_mesh': 'depth_colored_mesh.ply',
            'shape_data': '3d_shape_data.npy',
            'metrics_json': 'metrics_*.json',
            'summary_json': 'experiment_summary.json'
        }
        
        for file_type, pattern in file_patterns.items():
            matches = glob.glob(os.path.join(exp_path, pattern))
            if matches:
                # Use relative path from experiments dir
                info['files'][file_type] = os.path.relpath(matches[0], self.experiments_dir)
        
        # Also collect all mesh PLY files
        mesh_files = glob.glob(os.path.join(exp_path, 'mesh_*.ply'))
        if mesh_files:
            info['files']['mesh_files'] = [os.path.relpath(f, self.experiments_dir) for f in mesh_files]
        
        # Load metrics if available
        if 'metrics_json' in info['files']:
            metrics_path = os.path.join(self.experiments_dir, info['files']['metrics_json'])
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    info['metrics'] = metrics_data.get('metrics', {})
            except:
                pass
                
        # Load summary if available
        if 'summary_json' in info['files']:
            summary_path = os.path.join(self.experiments_dir, info['files']['summary_json'])
            try:
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)
                    info['metadata'] = {
                        'timestamp': summary_data.get('timestamp', ''),
                        'num_frames': summary_data.get('num_frames', 0),
                        'resolution': summary_data.get('resolution', []),
                        'status': summary_data.get('status', 'completed'),
                        'voxel_resolution': summary_data.get('voxel_resolution', 128),
                        'metrics_visualized': summary_data.get('metrics_visualized', [])
                    }
                    
                    # If no separate metrics file, try to extract metrics
                    if not info['metrics']:
                        # Try to get metrics from the metrics extractor
                        extracted_metrics = self.metrics_extractor.get_metrics_from_summary(cfg, seed)
                        if extracted_metrics:
                            info['metrics'] = extracted_metrics
            except:
                pass
        
        # Get creation time from directory
        try:
            stat = os.stat(exp_path)
            info['metadata']['created'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except:
            pass
            
        return info
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all metrics across experiments"""
        all_experiments = self.get_all_experiments()
        metrics_summary = {}
        
        # Collect all metrics
        all_metrics = {}
        for exp in all_experiments:
            for metric_name, metric_value in exp.get('metrics', {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append({
                    'cfg': exp['cfg'],
                    'seed': exp['seed'],
                    'value': metric_value
                })
        
        # Calculate statistics
        for metric_name, values in all_metrics.items():
            metric_values = [v['value'] for v in values]
            metrics_summary[metric_name] = {
                'count': len(metric_values),
                'mean': sum(metric_values) / len(metric_values) if metric_values else 0,
                'min': min(metric_values) if metric_values else 0,
                'max': max(metric_values) if metric_values else 0,
                'values': values
            }
            
        return metrics_summary
    
    def get_experiment_comparisons(self, cfg: Optional[str] = None) -> Dict[str, Any]:
        """Get experiments grouped for comparison"""
        all_experiments = self.get_all_experiments()
        
        if cfg:
            experiments = [exp for exp in all_experiments if exp['cfg'] == cfg]
        else:
            experiments = all_experiments
            
        # Group by cfg
        by_cfg = {}
        for exp in experiments:
            if exp['cfg'] not in by_cfg:
                by_cfg[exp['cfg']] = []
            by_cfg[exp['cfg']].append(exp)
            
        return by_cfg
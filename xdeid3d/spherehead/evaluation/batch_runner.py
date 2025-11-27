#!/usr/bin/env python3
"""
Batch runner for SphereHead-GUARD 3D mesh evaluation
Runs evaluation for multiple seeds and configurations
"""

import os
import sys
import subprocess
import argparse
from typing import List, Tuple
import time


def parse_seed_range(seed_range: str) -> List[int]:
    """Parse seed range string (e.g., '1-3' -> [1, 2, 3])"""
    if '-' in seed_range:
        start, end = map(int, seed_range.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(seed_range)]


def run_evaluation(seed: int, cfg: str, num_frames: int, script_path: str, 
                  spherehead_model: str, guard_script: str, output_dir: str,
                  device: str = None) -> bool:
    """Run evaluation for a single seed and configuration"""
    
    print(f"\n{'='*60}")
    print(f"Running evaluation: cfg={cfg}, seed={seed}")
    print(f"{'='*60}\n")
    
    # Construct command
    cmd = [
        sys.executable,
        script_path,
        "--seed", str(seed),
        "--cfg", cfg,
        "--num_frames", str(num_frames),
        "--spherehead_model", spherehead_model,
        "--guard_script", guard_script,
        "--output_dir", output_dir,
    ]
    
    if device:
        cmd.extend(["--device", device])
    
    # Run evaluation with real-time output
    try:
        start_time = time.time()
        # Use subprocess.run without capture_output to show real-time output
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Completed {cfg}/seed_{seed} in {elapsed_time:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed {cfg}/seed_{seed}")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch runner for SphereHead-GUARD evaluation")
    
    parser.add_argument(
        "seed_range",
        type=str,
        help="Seed range (e.g., '1-3' for seeds 1, 2, 3 or '5' for just seed 5)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["Head", "FFHQ", "Cats"],
        choices=["Head", "FFHQ", "Cats"],
        help="Configurations to run (default: all)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=90,
        help="Number of frames per video (default: 90)"
    )
    parser.add_argument(
        "--spherehead_model",
        type=str,
        default="../models/spherehead-ckpt-025000.pkl",
        help="Path to SphereHead model"
    )
    parser.add_argument(
        "--guard_script",
        type=str,
        default="GUARD/video_anonymization/video_anonymization_FIVA_temporal_metrics.py",
        help="Path to GUARD anonymization script"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Base output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0')"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing"
    )
    
    args = parser.parse_args()
    
    # Parse seed range
    seeds = parse_seed_range(args.seed_range)
    
    # Get evaluation script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "spherehead_guard_3dmesh_evaluation.py")
    
    if not os.path.exists(eval_script):
        print(f"Error: Evaluation script not found at {eval_script}")
        sys.exit(1)
    
    # Check dependencies
    if not os.path.exists(args.spherehead_model):
        print(f"Error: SphereHead model not found at {args.spherehead_model}")
        sys.exit(1)
    
    if not os.path.exists(args.guard_script):
        print(f"Error: GUARD script not found at {args.guard_script}")
        sys.exit(1)
    
    # Print configuration
    print(f"Batch evaluation configuration:")
    print(f"  Seeds: {seeds} ({len(seeds)} total)")
    print(f"  Configs: {args.configs}")
    print(f"  Frames per video: {args.num_frames}")
    print(f"  Total evaluations: {len(seeds) * len(args.configs)}")
    print(f"  Output directory: {args.output_dir}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No commands will be executed]")
    
    # Run evaluations
    successful = 0
    failed = 0
    start_time = time.time()
    
    for cfg in args.configs:
        for seed in seeds:
            if args.dry_run:
                output_path = os.path.join(args.output_dir, cfg, str(seed))
                print(f"\nWould run: cfg={cfg}, seed={seed}")
                print(f"  Output directory: {output_path}")
                continue
            
            success = run_evaluation(
                seed=seed,
                cfg=cfg,
                num_frames=args.num_frames,
                script_path=eval_script,
                spherehead_model=args.spherehead_model,
                guard_script=args.guard_script,
                output_dir=args.output_dir,
                device=args.device
            )
            
            if success:
                successful += 1
            else:
                failed += 1
    
    if not args.dry_run:
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Batch evaluation complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average time per run: {total_time / (successful + failed):.1f}s")
        print(f"{'='*60}")
        
        # Print results locations
        print(f"\nResults saved in:")
        for cfg in args.configs:
            for seed in seeds:
                result_dir = os.path.join(args.output_dir, cfg, str(seed))
                if os.path.exists(result_dir):
                    print(f"  - {result_dir}")


if __name__ == "__main__":
    main()
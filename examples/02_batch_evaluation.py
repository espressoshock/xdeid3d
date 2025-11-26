#!/usr/bin/env python3
"""
Example 02: Batch Evaluation

This example demonstrates how to evaluate a directory of images
using the evaluation pipeline.

Usage:
    python 02_batch_evaluation.py <original_dir> <anonymized_dir>
"""

import sys
from pathlib import Path


def main():
    """Run batch evaluation example."""
    if len(sys.argv) < 3:
        print("Usage: python 02_batch_evaluation.py <original_dir> <anonymized_dir>")
        print("\nThis example evaluates all image pairs in the directories.")
        sys.exit(1)

    original_dir = Path(sys.argv[1])
    anonymized_dir = Path(sys.argv[2])

    # Import X-DeID3D components
    from xdeid3d.evaluation import EvaluationPipeline, EvaluationConfig

    # Configure the evaluation
    config = EvaluationConfig(
        metrics=["psnr", "ssim"],  # Use quality metrics
        batch_size=1,
        num_workers=0,
    )

    # Create pipeline
    print("Creating evaluation pipeline...")
    pipeline = EvaluationPipeline(config)

    # Run evaluation
    print(f"\nEvaluating images:")
    print(f"  Original: {original_dir}")
    print(f"  Anonymized: {anonymized_dir}")
    print()

    results = pipeline.evaluate_directory(
        str(original_dir),
        str(anonymized_dir),
    )

    # Print results
    print(f"\nResults:")
    print(f"  Total samples: {results.total_samples}")
    print(f"  Successful: {results.successful_samples}")
    print(f"  Failed: {results.failed_samples}")

    print(f"\nMetric Statistics:")
    for metric_name, stats in results.metric_stats.items():
        print(f"\n  {metric_name}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std:  {stats['std']:.4f}")
        print(f"    Min:  {stats['min']:.4f}")
        print(f"    Max:  {stats['max']:.4f}")

    # Save results
    output_path = Path("evaluation_results.json")
    results.save(str(output_path))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

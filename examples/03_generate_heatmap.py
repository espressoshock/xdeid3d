#!/usr/bin/env python3
"""
Example 03: Generate Heatmap

This example demonstrates how to generate spherical heatmaps
from evaluation data.

Usage:
    python 03_generate_heatmap.py [results.json]
"""

import sys
from pathlib import Path

import numpy as np


def main():
    """Generate heatmap example."""
    # Import X-DeID3D components
    from xdeid3d.visualization import (
        HeatmapGenerator,
        create_2d_heatmap,
        create_polar_heatmap,
        save_figure,
    )

    # Use provided file or generate synthetic data
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
        print(f"Loading results from: {results_path}")

        import json
        with open(results_path) as f:
            data = json.load(f)

        # Extract data points
        if 'frame_metrics' in data:
            frame_data = data['frame_metrics']
        elif isinstance(data, list):
            frame_data = data
        else:
            print("Unknown data format")
            sys.exit(1)

        # Find metric column
        metric_name = None
        for key in frame_data[0].keys():
            if key not in ('yaw', 'pitch', 'frame_idx'):
                if isinstance(frame_data[0][key], (int, float)):
                    metric_name = key
                    break

        evaluation_data = [
            (d.get('yaw', 0), d.get('pitch', np.pi/2), d.get(metric_name, 0))
            for d in frame_data
        ]
    else:
        print("Generating synthetic evaluation data...")
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 500

        # Sample random angles
        yaws = np.random.uniform(-np.pi, np.pi, n_samples)
        pitches = np.random.uniform(0.3, 2.8, n_samples)

        # Create a score pattern (higher for frontal views)
        scores = 0.5 + 0.4 * np.cos(yaws) * np.sin(pitches)
        scores += 0.1 * np.random.randn(n_samples)
        scores = np.clip(scores, 0, 1)

        evaluation_data = list(zip(yaws, pitches, scores))
        metric_name = "synthetic_score"

    # Create heatmap generator
    print(f"\nGenerating heatmap for metric: {metric_name}")
    generator = HeatmapGenerator(resolution=72)
    generator.set_metric_name(metric_name)

    # Add data points
    for yaw, pitch, score in evaluation_data:
        generator.add_score(float(yaw), float(pitch), float(score))

    print(f"Added {len(evaluation_data)} data points")

    # Generate heatmap
    print("\nGenerating heatmap...")
    heatmap = generator.generate()

    # Get statistics
    stats = heatmap.get_statistics()
    print(f"\nHeatmap statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")

    # Generate different visualizations
    print("\nCreating visualizations...")

    # Rectangular projection
    rect_img = heatmap.to_rgb(colormap="magma")
    save_figure(rect_img, "heatmap_rectangular.png")
    print("  Saved: heatmap_rectangular.png")

    # Polar projection
    polar_img = create_polar_heatmap(heatmap, colormap="magma")
    save_figure(polar_img, "heatmap_polar.png")
    print("  Saved: heatmap_polar.png")

    print("\nHeatmap generation complete!")


if __name__ == "__main__":
    main()

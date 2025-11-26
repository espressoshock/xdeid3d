#!/usr/bin/env python3
"""
Example 06: Mesh Visualization

This example demonstrates how to generate colored 3D meshes
from evaluation data for visualization.

Usage:
    python 06_mesh_visualization.py [results.json]
"""

import sys
from pathlib import Path

import numpy as np


def create_sphere_mesh(radius: float = 1.0, resolution: int = 30):
    """Create a simple UV sphere mesh."""
    # Generate vertices
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution * 2)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.cos(phi)
    z = radius * np.sin(phi) * np.sin(theta)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

    # Generate faces
    faces = []
    for i in range(resolution * 2 - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])

    return vertices, np.array(faces, dtype=np.int32)


def main():
    """Mesh visualization example."""
    from xdeid3d.visualization import MeshExporter, write_ply

    # Generate or load evaluation data
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
        print(f"Loading results from: {results_path}")

        import json
        with open(results_path) as f:
            data = json.load(f)

        if 'frame_metrics' in data:
            frame_data = data['frame_metrics']
        elif isinstance(data, list):
            frame_data = data
        else:
            print("Unknown data format")
            sys.exit(1)

        # Find metric
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
        np.random.seed(42)
        n_samples = 300

        # Create pattern: high scores at front, low at sides
        yaws = np.random.uniform(-np.pi, np.pi, n_samples)
        pitches = np.random.uniform(0.3, 2.8, n_samples)

        # Score pattern based on viewing angle
        scores = 0.7 - 0.4 * np.abs(yaws) / np.pi
        scores *= np.sin(pitches)
        scores += 0.1 * np.random.randn(n_samples)
        scores = np.clip(scores, 0, 1)

        evaluation_data = list(zip(yaws, pitches, scores))
        metric_name = "synthetic_score"

    print(f"\nCreating mesh visualization for: {metric_name}")
    print(f"Data points: {len(evaluation_data)}")

    # Create mesh exporter
    exporter = MeshExporter(
        colormap="magma",
        bandwidth=0.5,
    )
    exporter.set_metric_name(metric_name)

    # Add scores
    for yaw, pitch, score in evaluation_data:
        exporter.add_score(float(yaw), float(pitch), float(score))

    # Create base mesh (sphere)
    print("\nCreating sphere mesh...")
    vertices, faces = create_sphere_mesh(radius=1.0, resolution=50)
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")

    # Export colored mesh
    output_path = "evaluation_mesh.ply"
    print(f"\nExporting colored mesh to: {output_path}")

    mesh = exporter.export_ply(output_path, vertices, faces)

    print(f"\nMesh statistics:")
    print(f"  Vertices: {mesh.n_vertices}")
    print(f"  Faces: {mesh.n_faces}")
    print(f"  Score range: [{mesh.min_score:.4f}, {mesh.max_score:.4f}]")

    print(f"\nMesh saved to: {output_path}")
    print("You can view this file in MeshLab, Blender, or any PLY viewer.")
    print("\nMesh visualization example complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example 01: Basic Evaluation

This example demonstrates how to evaluate a single image pair
and access the results.

Usage:
    python 01_basic_evaluation.py original.jpg anonymized.jpg
"""

import sys
from pathlib import Path

import numpy as np


def main():
    """Run basic evaluation example."""
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python 01_basic_evaluation.py <original> <anonymized>")
        print("\nThis example evaluates a single image pair.")
        sys.exit(1)

    original_path = Path(sys.argv[1])
    anonymized_path = Path(sys.argv[2])

    # Import X-DeID3D components
    from xdeid3d.utils.io import load_image
    from xdeid3d.metrics import get_metric, list_metrics

    # Show available metrics
    print("Available metrics:")
    for name, info in list_metrics().items():
        print(f"  - {name}: {info.get('description', 'No description')}")
    print()

    # Load images
    print(f"Loading images...")
    original = load_image(str(original_path), color_space="RGB")
    anonymized = load_image(str(anonymized_path), color_space="RGB")
    print(f"  Original: {original.shape}")
    print(f"  Anonymized: {anonymized.shape}")
    print()

    # Compute individual metrics
    print("Computing metrics:")

    # PSNR (Peak Signal-to-Noise Ratio)
    psnr = get_metric("psnr")
    psnr_score = psnr.compute(original, anonymized)
    print(f"  PSNR: {psnr_score:.2f} dB")

    # SSIM (Structural Similarity)
    try:
        ssim = get_metric("ssim")
        ssim_score = ssim.compute(original, anonymized)
        print(f"  SSIM: {ssim_score:.4f}")
    except Exception as e:
        print(f"  SSIM: Not available ({e})")

    # Identity metrics (require face recognition models)
    try:
        cosine = get_metric("cosine_similarity")
        cosine_score = cosine.compute(original, anonymized)
        print(f"  Cosine Similarity: {cosine_score:.4f}")
    except Exception as e:
        print(f"  Cosine Similarity: Not available ({e})")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

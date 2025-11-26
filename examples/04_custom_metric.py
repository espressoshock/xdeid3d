#!/usr/bin/env python3
"""
Example 04: Custom Metric

This example demonstrates how to create and register a custom
evaluation metric.

Usage:
    python 04_custom_metric.py <original> <anonymized>
"""

import sys
from pathlib import Path

import numpy as np


def main():
    """Custom metric example."""
    # Import base classes
    from xdeid3d.metrics import BaseMetric, register_metric, get_metric

    # Define a custom metric
    class ColorHistogramMetric(BaseMetric):
        """Custom metric comparing color histograms.

        Computes the histogram intersection between original
        and anonymized images as a measure of color preservation.
        """

        @property
        def name(self) -> str:
            return "color_histogram"

        @property
        def description(self) -> str:
            return "Color histogram intersection score"

        @property
        def higher_is_better(self) -> bool:
            return True  # Higher = more similar colors

        @property
        def value_range(self) -> tuple:
            return (0.0, 1.0)

        def compute(
            self,
            original: np.ndarray,
            anonymized: np.ndarray,
            **kwargs,
        ) -> float:
            """Compute color histogram intersection."""
            # Compute histograms for each channel
            score = 0.0
            for channel in range(3):
                hist_orig, _ = np.histogram(
                    original[:, :, channel].ravel(),
                    bins=256,
                    range=(0, 255),
                    density=True,
                )
                hist_anon, _ = np.histogram(
                    anonymized[:, :, channel].ravel(),
                    bins=256,
                    range=(0, 255),
                    density=True,
                )

                # Histogram intersection
                intersection = np.minimum(hist_orig, hist_anon).sum()
                score += intersection

            # Average over channels
            return score / 3.0

    # Register the custom metric
    register_metric("color_histogram", ColorHistogramMetric)
    print("Registered custom metric: color_histogram")

    # Verify registration
    metric = get_metric("color_histogram")
    print(f"\nMetric info:")
    print(f"  Name: {metric.name}")
    print(f"  Description: {metric.description}")
    print(f"  Higher is better: {metric.higher_is_better}")

    # Use the metric if images are provided
    if len(sys.argv) >= 3:
        from xdeid3d.utils.io import load_image

        original_path = Path(sys.argv[1])
        anonymized_path = Path(sys.argv[2])

        print(f"\nLoading images...")
        original = load_image(str(original_path))
        anonymized = load_image(str(anonymized_path))

        # Compute the custom metric
        score = metric.compute(original, anonymized)
        print(f"\nColor histogram intersection: {score:.4f}")
    else:
        # Demo with synthetic images
        print("\nDemo with synthetic images:")
        original = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        anonymized = original.copy()
        anonymized[:, :, 0] = np.clip(anonymized[:, :, 0].astype(int) + 20, 0, 255)

        score = metric.compute(original, anonymized)
        print(f"  Score (slightly modified): {score:.4f}")

        # Very different image
        different = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        score_diff = metric.compute(original, different)
        print(f"  Score (random different): {score_diff:.4f}")

    print("\nCustom metric example complete!")


if __name__ == "__main__":
    main()

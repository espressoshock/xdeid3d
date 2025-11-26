#!/usr/bin/env python3
"""
Example 05: Custom Anonymizer

This example demonstrates how to create and register a custom
anonymizer that implements the AnonymizerProtocol.

Usage:
    python 05_custom_anonymizer.py <input_image> [output_image]
"""

import sys
from pathlib import Path

import numpy as np


def main():
    """Custom anonymizer example."""
    # Import base classes
    from xdeid3d.anonymizers import (
        AnonymizerProtocol,
        register_anonymizer,
        get_anonymizer,
        list_anonymizers,
    )

    # Define a custom anonymizer
    class PixelateAnonymizer:
        """Custom anonymizer that pixelates faces.

        This anonymizer reduces the resolution of the face region
        to create a pixelation effect for anonymization.
        """

        def __init__(self, block_size: int = 10):
            """Initialize with block size for pixelation.

            Args:
                block_size: Size of pixelation blocks in pixels
            """
            self.block_size = block_size
            self._initialized = False

        @property
        def name(self) -> str:
            return "pixelate"

        def initialize(self, device: str = "cpu") -> None:
            """Initialize the anonymizer (no-op for this simple example)."""
            self._initialized = True
            print(f"Pixelate anonymizer initialized on {device}")

        def anonymize(
            self,
            image: np.ndarray,
            face_bbox: tuple = None,
            **kwargs,
        ) -> np.ndarray:
            """Apply pixelation to the image.

            Args:
                image: Input image (H, W, C)
                face_bbox: Optional (x, y, w, h) bounding box
                **kwargs: Additional arguments (ignored)

            Returns:
                Pixelated image
            """
            result = image.copy()
            h, w = image.shape[:2]

            # Determine region to pixelate
            if face_bbox is not None:
                x, y, bw, bh = face_bbox
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)
            else:
                # Pixelate entire image
                x1, y1, x2, y2 = 0, 0, w, h

            # Apply pixelation
            region = result[y1:y2, x1:x2]
            small = self._downsample(region, self.block_size)
            upsampled = self._upsample(small, region.shape[:2])
            result[y1:y2, x1:x2] = upsampled

            return result

        def _downsample(self, image: np.ndarray, block_size: int) -> np.ndarray:
            """Downsample image by averaging blocks."""
            h, w = image.shape[:2]
            new_h = max(1, h // block_size)
            new_w = max(1, w // block_size)

            # Simple block averaging
            result = np.zeros((new_h, new_w, 3), dtype=np.float32)
            for i in range(new_h):
                for j in range(new_w):
                    y1 = i * block_size
                    y2 = min((i + 1) * block_size, h)
                    x1 = j * block_size
                    x2 = min((j + 1) * block_size, w)
                    result[i, j] = image[y1:y2, x1:x2].mean(axis=(0, 1))

            return result.astype(np.uint8)

        def _upsample(self, image: np.ndarray, target_shape: tuple) -> np.ndarray:
            """Upsample image using nearest neighbor."""
            h, w = image.shape[:2]
            target_h, target_w = target_shape

            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            for i in range(target_h):
                for j in range(target_w):
                    src_i = min(int(i * h / target_h), h - 1)
                    src_j = min(int(j * w / target_w), w - 1)
                    result[i, j] = image[src_i, src_j]

            return result

    # Register the custom anonymizer
    register_anonymizer("pixelate", PixelateAnonymizer)
    print("Registered custom anonymizer: pixelate")

    # Show all available anonymizers
    print("\nAvailable anonymizers:")
    for name, info in list_anonymizers().items():
        print(f"  - {name}")

    # Use the anonymizer
    anonymizer = get_anonymizer("pixelate")
    anonymizer.initialize()

    if len(sys.argv) >= 2:
        from xdeid3d.utils.io import load_image, save_image

        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_stem(
            input_path.stem + "_pixelated"
        )

        print(f"\nLoading: {input_path}")
        image = load_image(str(input_path))

        print("Applying pixelation...")
        result = anonymizer.anonymize(image)

        print(f"Saving: {output_path}")
        save_image(result, str(output_path))
    else:
        # Demo with synthetic image
        print("\nDemo with synthetic image:")
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = anonymizer.anonymize(image)
        print(f"  Input shape: {image.shape}")
        print(f"  Output shape: {result.shape}")

    print("\nCustom anonymizer example complete!")


if __name__ == "__main__":
    main()

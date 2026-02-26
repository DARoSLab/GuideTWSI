"""
Binary mask generator for GuideTWSI.

Generates binary ground-truth masks from COCO-format RLE JSON annotations.
Combines all annotation masks per image into a single binary mask.

Usage:
    python mask_generator.py --input /path/to/json_dir --output /path/to/mask_dir
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def generate_masks(input_dir: str, output_dir: str) -> None:
    """Generate binary masks from RLE-encoded JSON annotations.

    Args:
        input_dir: Directory containing JSON files with RLE segmentation data.
        output_dir: Directory for output binary mask PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        if not annotations:
            continue

        # Decode first annotation to get mask shape
        seg = annotations[0]["segmentation"]
        combined_mask = mask_utils.decode(seg)

        # Combine all annotations with bitwise OR
        for ann in annotations[1:]:
            seg = ann["segmentation"]
            combined_mask |= mask_utils.decode(seg)

        # Save as 0/255 grayscale PNG
        output_path = os.path.join(
            output_dir, filename.replace(".json", ".png")
        )
        Image.fromarray(combined_mask * 255).save(output_path)
        count += 1

    print(f"Generated {count} binary masks in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="GuideTWSI Mask Generator: Create binary masks from RLE annotations."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing JSON files with RLE segmentation",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for binary mask PNG files",
    )
    args = parser.parse_args()

    generate_masks(args.input, args.output)


if __name__ == "__main__":
    main()

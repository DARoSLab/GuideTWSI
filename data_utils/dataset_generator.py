"""
Dataset split generator for GuideTWSI.

Splits a flat image + label dataset into train/val/test subsets and generates
a YOLO-compatible data.yaml configuration file.

Usage:
    python dataset_generator.py \
        --image-dir /path/to/images \
        --label-dir /path/to/labels \
        --output-dir /path/to/output \
        --train-split 0.88 \
        --val-split 0.06 \
        --test-split 0.06
"""

import argparse
import os
import random
import shutil

import yaml


def split_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    train_split: float = 0.88,
    val_split: float = 0.06,
    test_split: float = 0.06,
    image_ext: str = ".jpg",
    label_ext: str = ".json",
    seed: int = 42,
) -> None:
    """Split a dataset into train/val/test subsets.

    Args:
        image_dir: Directory containing all images.
        label_dir: Directory containing corresponding label files.
        output_dir: Output directory for the split dataset.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        image_ext: Image file extension to look for.
        label_ext: Label file extension.
        seed: Random seed for reproducibility.
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, (
        f"Split ratios must sum to 1.0, got {train_split + val_split + test_split}"
    )

    # Create output directories
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Gather images
    images = [f for f in os.listdir(image_dir) if f.endswith(image_ext)]
    labels = [f.replace(image_ext, label_ext) for f in images]

    # Shuffle with fixed seed
    random.seed(seed)
    dataset = list(zip(images, labels))
    random.shuffle(dataset)

    # Calculate split indices
    total = len(dataset)
    train_count = int(total * train_split)
    val_count = int(total * val_split)

    train_set = dataset[:train_count]
    val_set = dataset[train_count : train_count + val_count]
    test_set = dataset[train_count + val_count :]

    # Copy files to split directories
    def copy_split(data_set, split_name):
        dest = os.path.join(output_dir, split_name)
        for image, label in data_set:
            src_image = os.path.join(image_dir, image)
            src_label = os.path.join(label_dir, label)
            if os.path.exists(src_image):
                shutil.copy2(src_image, os.path.join(dest, image))
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(dest, label))

    copy_split(train_set, "train")
    copy_split(val_set, "val")
    copy_split(test_set, "test")

    # Generate data.yaml for YOLO training
    data_yaml = {
        "train": os.path.abspath(os.path.join(output_dir, "train")),
        "val": os.path.abspath(os.path.join(output_dir, "val")),
        "test": os.path.abspath(os.path.join(output_dir, "test")),
        "nc": 1,
        "names": ["tactile_paving"],
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Dataset split complete:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")
    print(f"  YAML:  {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GuideTWSI Dataset Generator: Split datasets into train/val/test."
    )
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing all images"
    )
    parser.add_argument(
        "--label-dir", required=True, help="Directory containing label files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for split dataset"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.88,
        help="Training split ratio (default: 0.88)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.06,
        help="Validation split ratio (default: 0.06)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.06,
        help="Test split ratio (default: 0.06)",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        help="Image file extension (default: .jpg)",
    )
    parser.add_argument(
        "--label-ext",
        type=str,
        default=".json",
        help="Label file extension (default: .json)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    split_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        image_ext=args.image_ext,
        label_ext=args.label_ext,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

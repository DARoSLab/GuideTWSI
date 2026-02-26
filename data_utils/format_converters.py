"""
Unified format converter for GuideTWSI dataset annotations.

Supports conversions between AirSim JSON, YOLO, SAM2, COCO, and RLE formats.
Consolidates functionality from multiple NavAble Data_utils scripts.

Usage:
    python format_converters.py --from airsim --to yolo --input <dir> --output <dir>
    python format_converters.py --from yolo --to sam2 --input <dir> --output <dir>
    python format_converters.py --from sam2 --to yolo --input <dir> --output <dir>
    python format_converters.py --from coco --to rle --input <dir> --output <dir>
"""

import argparse
import json
import os
from typing import Optional

import numpy as np


def airsim_json_to_yolo(
    input_dir: str,
    output_dir: str,
    image_width: int = 1920,
    image_height: int = 1080,
    target_class: str = "tactile block",
    class_id: int = 0,
    max_bbox_ratio: float = 0.25,
) -> None:
    """Convert AirSim JSON bounding box labels to YOLO format.

    Args:
        input_dir: Directory containing AirSim JSON label files.
        output_dir: Directory for output YOLO .txt files.
        image_width: Width of source images in pixels.
        image_height: Height of source images in pixels.
        target_class: Class name to filter for.
        class_id: YOLO class ID to assign.
        max_bbox_ratio: Maximum normalized bbox dimension to accept.
    """
    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(input_dir):
        if not json_file.endswith(".json"):
            continue

        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file.replace(".json", ".txt"))

        with open(input_path, "r") as f:
            data = json.load(f)

        with open(output_path, "w") as out_file:
            objects = data if isinstance(data, list) else data.get("objects", [])
            for obj in objects:
                obj_class = obj.get("class") or obj.get("object_name", "")
                if obj_class != target_class:
                    continue

                if "bounding_box" not in obj:
                    continue

                bbox = obj["bounding_box"]
                if "top_left" in bbox:
                    x1, y1 = bbox["top_left"]
                    x2, y2 = bbox["bottom_right"]
                else:
                    x1 = bbox["x_min"]
                    y1 = bbox["y_min"]
                    x2 = bbox["x_max"]
                    y2 = bbox["y_max"]

                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = abs(x2 - x1) / image_width
                height = abs(y2 - y1) / image_height

                if (width < max_bbox_ratio and height < max_bbox_ratio) or (
                    width > 5 or height > 5
                ):
                    continue

                out_file.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}\n"
                )

    print(f"AirSim -> YOLO conversion complete. Output: {output_dir}")


def yolo_to_sam2(
    input_dir: str,
    output_dir: str,
    image_width: int = 1024,
    image_height: int = 1024,
    image_ext: str = ".jpg",
    object_color: Optional[list[int]] = None,
) -> None:
    """Convert YOLO .txt label files to SAM2 JSON format.

    Args:
        input_dir: Directory containing YOLO .txt label files.
        output_dir: Directory for output SAM2 JSON files.
        image_width: Target image width in pixels.
        image_height: Target image height in pixels.
        image_ext: Image file extension.
        object_color: RGB color for object annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    if object_color is None:
        object_color = [153, 108, 6]

    n = 0
    for label_file in sorted(os.listdir(input_dir)):
        if not label_file.endswith(".txt"):
            continue

        input_path = os.path.join(input_dir, label_file)
        has_annotations = False
        sam2_annotation = {}

        with open(input_path, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                has_annotations = True
                x_c_rel, y_c_rel, w_rel, h_rel = map(float, parts[1:])

                x_c = x_c_rel * image_width
                y_c = y_c_rel * image_height
                width = w_rel * image_width
                height = h_rel * image_height
                x_min = x_c - width / 2
                y_min = y_c - height / 2
                area = width * height

                sam2_annotation = {
                    "image": {
                        "image_id": n + 1,
                        "license": 1,
                        "file_name": label_file.replace(".txt", image_ext),
                        "height": image_height,
                        "width": image_width,
                        "date_captured": "",
                    },
                    "annotations": [
                        {
                            "id": i + 1,
                            "bbox": [x_min, y_min, width, height],
                            "area": area,
                            "segmentation": {
                                "counts": "",
                                "size": [image_height, image_width],
                            },
                            "object_color": object_color,
                        }
                    ],
                }

        if has_annotations:
            output_path = os.path.join(
                output_dir, label_file.replace(".txt", ".json")
            )
            with open(output_path, "w") as f:
                json.dump(sam2_annotation, f)

        n += 1

    print(f"YOLO -> SAM2 conversion complete. Output: {output_dir}")


def sam2_to_yolo(
    input_dir: str,
    output_dir: str,
    class_id: int = 0,
) -> None:
    """Convert SAM2 JSON annotations to YOLO .txt format.

    Args:
        input_dir: Directory containing SAM2 JSON files.
        output_dir: Directory for output YOLO .txt files.
        class_id: YOLO class ID to assign.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(input_dir, filename), "r") as f:
            data = json.load(f)

        yolo_lines = []
        if data:
            image_info = data["image"]
            annotations = data["annotations"]
            img_width = image_info["width"]
            img_height = image_info["height"]

            for ann in annotations:
                x_min, y_min, width, height = ann["bbox"]
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                width /= img_width
                height /= img_height

                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}"
                )

        label_name = filename.replace(".json", ".txt")
        label_path = os.path.join(output_dir, label_name)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"SAM2 -> YOLO conversion complete. Output: {output_dir}")


def generate_coco_json(
    image_dir: str,
    annotation_dir: str,
    output_path: str,
    image_width: int = 1024,
    image_height: int = 1024,
    category_name: str = "tactile_paving",
    category_id: int = 1,
) -> None:
    """Generate a COCO-format JSON annotation file from individual annotations.

    Args:
        image_dir: Directory containing images.
        annotation_dir: Directory containing per-image annotation JSONs.
        output_path: Output path for the COCO JSON file.
        image_width: Image width.
        image_height: Image height.
        category_name: Category name.
        category_id: Category ID.
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_id, "name": category_name}],
    }

    ann_id = 1
    for img_id, filename in enumerate(sorted(os.listdir(image_dir)), start=1):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": image_width,
            "height": image_height,
        })

        ann_file = os.path.splitext(filename)[0] + ".json"
        ann_path = os.path.join(annotation_dir, ann_file)
        if not os.path.exists(ann_path):
            continue

        with open(ann_path, "r") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        for ann in annotations:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "segmentation": ann.get("segmentation", {}),
                "iscrowd": 0,
            })
            ann_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"COCO JSON generated: {output_path}")


def polygon_to_rle(
    input_dir: str,
    output_dir: str,
) -> None:
    """Generate binary masks from COCO-format RLE annotations.

    This is a thin wrapper around mask_generator.py functionality,
    provided here for CLI convenience.

    Args:
        input_dir: Directory containing JSON files with RLE segmentation.
        output_dir: Directory for output binary mask PNGs.
    """
    from pycocotools import mask as mask_utils
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(input_dir, filename), "r") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        if not annotations:
            continue

        seg = annotations[0]["segmentation"]
        combined_mask = mask_utils.decode(seg)

        for ann in annotations:
            seg = ann["segmentation"]
            combined_mask |= mask_utils.decode(seg)

        Image.fromarray(combined_mask * 255).save(
            os.path.join(output_dir, filename.replace(".json", ".png"))
        )

    print(f"RLE -> Binary mask conversion complete. Output: {output_dir}")


# Conversion dispatch table
CONVERTERS = {
    ("airsim", "yolo"): airsim_json_to_yolo,
    ("yolo", "sam2"): yolo_to_sam2,
    ("sam2", "yolo"): sam2_to_yolo,
    ("coco", "rle"): polygon_to_rle,
}


def main():
    parser = argparse.ArgumentParser(
        description="GuideTWSI Format Converter: Convert between annotation formats."
    )
    parser.add_argument(
        "--from",
        dest="from_format",
        required=True,
        choices=["airsim", "yolo", "sam2", "coco"],
        help="Source annotation format",
    )
    parser.add_argument(
        "--to",
        dest="to_format",
        required=True,
        choices=["yolo", "sam2", "rle", "coco"],
        help="Target annotation format",
    )
    parser.add_argument(
        "--input", required=True, help="Input directory with source annotations"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for converted annotations"
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=1024,
        help="Image height in pixels (default: 1024)",
    )
    args = parser.parse_args()

    key = (args.from_format, args.to_format)
    if key not in CONVERTERS:
        parser.error(
            f"Conversion from '{args.from_format}' to '{args.to_format}' is not supported. "
            f"Supported conversions: {list(CONVERTERS.keys())}"
        )

    converter = CONVERTERS[key]

    # Build kwargs based on what the converter accepts
    kwargs = {"input_dir": args.input, "output_dir": args.output}
    if key in [("airsim", "yolo"), ("yolo", "sam2")]:
        kwargs["image_width"] = args.image_width
        kwargs["image_height"] = args.image_height

    converter(**kwargs)


if __name__ == "__main__":
    main()

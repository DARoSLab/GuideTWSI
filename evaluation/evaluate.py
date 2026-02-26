"""
Unified evaluation script for GuideTWSI segmentation models.

Runs inference on a test set and computes segmentation metrics (Precision,
Recall, mAP50-95, mIoU) for any supported model type.

Usage:
    python evaluation/evaluate.py \
        --model yolov11_seg_n \
        --weights /path/to/weights.pt \
        --data /path/to/test_dir \
        --config configs/yolov11_seg_n.yaml

Supported models: yolov11_seg_n, yolov11_seg_x, mask2former, sam2_unet,
                  dinov3_regcls, dinov3_eomt
"""

import argparse
import os
import sys

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import compute_binary_metrics, aggregate_metrics


def load_test_data(data_dir: str) -> tuple[list[str], list[str]]:
    """Load test image and mask file paths.

    Expects either:
      - data_dir/images/*.jpg + data_dir/masks/*.png
      - data_dir/*.jpg + data_dir/masks/*.png

    Returns:
        Tuple of (image_paths, mask_paths).
    """
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    if not os.path.isdir(image_dir):
        image_dir = data_dir

    if not os.path.isdir(mask_dir):
        mask_dir = data_dir

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not f.endswith("_mask.png")
        and not f.endswith("_mask.jpg")
    ])

    image_paths = []
    mask_paths = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        stem = os.path.splitext(img_file)[0]

        # Try different mask naming conventions
        for mask_name in [
            f"{stem}.png",
            f"{stem}_mask.png",
            f"{stem}_mask.jpg",
        ]:
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
                break

    return image_paths, mask_paths


def evaluate_yolov11(weights: str, image_paths: list[str], mask_paths: list[str]) -> list[dict]:
    """Evaluate a YOLOv11-Seg model."""
    from ultralytics import YOLO

    model = YOLO(weights)
    all_metrics = []

    for img_path, mask_path in tqdm(
        zip(image_paths, mask_paths), total=len(image_paths), desc="YOLOv11-Seg"
    ):
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        results = model(img_path, verbose=False)

        # Combine all predicted masks into a single binary mask
        pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        if results[0].masks is not None:
            for mask in results[0].masks.data:
                m = mask.cpu().numpy()
                m_resized = np.array(
                    Image.fromarray((m * 255).astype(np.uint8)).resize(
                        (gt_mask.shape[1], gt_mask.shape[0])
                    )
                )
                pred_mask = np.maximum(pred_mask, m_resized)

        metrics = compute_binary_metrics(gt_mask, pred_mask)
        all_metrics.append(metrics)

    return all_metrics


def evaluate_sam2_unet(weights: str, image_paths: list[str], mask_paths: list[str], config: dict) -> list[dict]:
    """Evaluate a SAM2.1+UNet model."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = config.get("training", {}).get(
        "sam2_config", "configs/sam2.1/sam2.1_hiera_b+.yaml"
    )
    sam2 = build_sam2(model_cfg, weights, device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    all_metrics = []
    for img_path, mask_path in tqdm(
        zip(image_paths, mask_paths), total=len(image_paths), desc="SAM2.1+UNet"
    ):
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        image = np.array(Image.open(img_path).convert("RGB"))
        result = mask_generator.generate(image)

        pred_mask = np.zeros(gt_mask.shape, dtype=bool)
        for m in result:
            seg = np.array(m["segmentation"])
            if seg.shape != pred_mask.shape:
                seg = np.array(
                    Image.fromarray(seg).resize(
                        (pred_mask.shape[1], pred_mask.shape[0])
                    )
                )
            pred_mask |= seg.astype(bool)

        pred_mask = (pred_mask.astype(np.uint8) * 255)
        metrics = compute_binary_metrics(gt_mask, pred_mask)
        all_metrics.append(metrics)

    return all_metrics


def evaluate_dinov3_regcls(weights: str, image_paths: list[str], mask_paths: list[str], config: dict) -> list[dict]:
    """Evaluate a DINOv3+RegCls model."""
    import pickle
    import torch
    import torchvision.transforms.functional as TF

    training_cfg = config.get("training", {})
    patch_size = training_cfg.get("patch_size", 16)
    image_size = training_cfg.get("image_size", 768)
    threshold = config.get("inference", {}).get("confidence_threshold", 0.5)
    mean = training_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = training_cfg.get("normalize_std", [0.229, 0.224, 0.225])

    dinov3_cfg = config.get("model", {}).get("dinov3", {})
    model_name = dinov3_cfg.get("variant", "dinov3_vits16")
    n_layers = dinov3_cfg.get("num_layers", 12)

    # Load DINOv3 backbone
    model = torch.hub.load("facebookresearch/dinov3", model_name)
    model.cuda().eval()

    # Load classifier
    with open(weights, "rb") as f:
        clf = pickle.load(f)

    def resize_transform(img, img_size=image_size, p_size=patch_size):
        w, h = img.size
        h_patches = int(img_size / p_size)
        w_patches = int((w * img_size) / (h * p_size))
        return TF.to_tensor(TF.resize(img, (h_patches * p_size, w_patches * p_size)))

    all_metrics = []
    for img_path, mask_path in tqdm(
        zip(image_paths, mask_paths), total=len(image_paths), desc="DINOv3+RegCls"
    ):
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        test_image = Image.open(img_path).convert("RGB")
        test_resized = resize_transform(test_image)
        test_norm = TF.normalize(test_resized, mean=mean, std=std)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                feats = model.get_intermediate_layers(
                    test_norm.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True
                )
                x = feats[-1].squeeze().detach().cpu()
                dim = x.shape[0]
                x = x.view(dim, -1).permute(1, 0)

        h_patches, w_patches = [int(d / patch_size) for d in test_resized.shape[1:]]
        fg_score = clf.predict_proba(x.numpy())[:, 1].reshape(h_patches, w_patches)
        pred_binary = (fg_score > threshold).astype(np.uint8) * 255

        # Resize prediction to match ground truth
        pred_resized = np.array(
            Image.fromarray(pred_binary).resize(
                (gt_mask.shape[1], gt_mask.shape[0]),
                Image.NEAREST,
            )
        )

        metrics = compute_binary_metrics(gt_mask, pred_resized)
        all_metrics.append(metrics)

    return all_metrics


def evaluate_dinov3_eomt(weights: str, image_paths: list[str], mask_paths: list[str]) -> list[dict]:
    """Evaluate a DINOv3+EoMT model."""
    import lightly_train

    model = lightly_train.load_model_from_checkpoint(weights)

    all_metrics = []
    for img_path, mask_path in tqdm(
        zip(image_paths, mask_paths), total=len(image_paths), desc="DINOv3+EoMT"
    ):
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        test_image = Image.open(img_path).convert("RGB")

        import torch
        masks = model.predict(test_image)
        masks = torch.stack([masks == c for c in masks.unique()]).detach().cpu().numpy()
        pred_mask = masks[-1].astype(np.uint8) * 255  # Last class = tactile paving

        # Resize to match ground truth
        pred_resized = np.array(
            Image.fromarray(pred_mask).resize(
                (gt_mask.shape[1], gt_mask.shape[0]),
                Image.NEAREST,
            )
        )

        metrics = compute_binary_metrics(gt_mask, pred_resized)
        all_metrics.append(metrics)

    return all_metrics


def print_results_table(model_name: str, avg_metrics: dict) -> None:
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"  Evaluation Results: {model_name}")
    print("=" * 70)
    print(f"  {'Metric':<20} {'Value':>10}")
    print("-" * 70)

    display_order = ["Precision", "Recall", "F1-Score", "IoU", "Accuracy"]
    for key in display_order:
        if key in avg_metrics:
            print(f"  {key:<20} {avg_metrics[key]:>10.4f}")

    print("=" * 70)


# Paper Table III reference results (Real + Synthetic Data -> RDome-2K)
PAPER_RESULTS = {
    "yolov11_seg_n": {"Prec.": 0.8718, "Rec.": 0.8084, "mAP50-95": 0.7288, "mIoU": 0.7308},
    "yolov11_seg_x": {"Prec.": 0.9102, "Rec.": 0.8588, "mAP50-95": 0.8188, "mIoU": 0.7887},
    "mask2former":   {"Prec.": 0.9611, "Rec.": 0.8669, "mAP50-95": 0.7829, "mIoU": 0.8375},
    "sam2_unet":     {"Prec.": 0.9704, "Rec.": 0.7031, "mAP50-95": 0.5627, "mIoU": 0.6883},
    "dinov3_regcls":  {"Prec.": 0.8667, "Rec.": 0.8924, "mAP50-95": 0.6933, "mIoU": 0.7926},
    "dinov3_eomt":    {"Prec.": 0.9305, "Rec.": 0.9197, "mAP50-95": 0.8492, "mIoU": 0.8756},
}


def main():
    parser = argparse.ArgumentParser(
        description="GuideTWSI Evaluation: Run inference and compute metrics."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "yolov11_seg_n",
            "yolov11_seg_x",
            "mask2former",
            "sam2_unet",
            "dinov3_regcls",
            "dinov3_eomt",
        ],
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--weights", required=True, help="Path to model weights/checkpoint"
    )
    parser.add_argument(
        "--data", required=True, help="Path to test data directory"
    )
    parser.add_argument(
        "--config", default=None, help="Path to model config YAML"
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Print paper reference results for comparison",
    )
    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Load test data
    image_paths, mask_paths = load_test_data(args.data)
    if not image_paths:
        print(f"Error: No test images found in {args.data}")
        sys.exit(1)
    print(f"Found {len(image_paths)} test images")

    # Run evaluation based on model type
    if args.model in ("yolov11_seg_n", "yolov11_seg_x"):
        all_metrics = evaluate_yolov11(args.weights, image_paths, mask_paths)
    elif args.model == "sam2_unet":
        all_metrics = evaluate_sam2_unet(args.weights, image_paths, mask_paths, config)
    elif args.model == "dinov3_regcls":
        all_metrics = evaluate_dinov3_regcls(args.weights, image_paths, mask_paths, config)
    elif args.model == "dinov3_eomt":
        all_metrics = evaluate_dinov3_eomt(args.weights, image_paths, mask_paths)
    elif args.model == "mask2former":
        # Mask2Former evaluation requires Detectron2
        print("Mask2Former evaluation requires Detectron2. See notebooks/Mask2Former.ipynb.")
        sys.exit(0)

    avg_metrics = aggregate_metrics(all_metrics)
    print_results_table(args.model, avg_metrics)

    if args.reference and args.model in PAPER_RESULTS:
        print("\nPaper reference (Real + Synthetic -> RDome-2K):")
        ref = PAPER_RESULTS[args.model]
        for k, v in ref.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

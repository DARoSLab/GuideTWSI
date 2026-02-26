"""
Evaluation metrics for GuideTWSI segmentation models.

Provides functions for computing binary segmentation metrics including
precision, recall, F1-score, IoU, mIoU, and mAP at various IoU thresholds.
"""

import numpy as np


def compute_binary_metrics(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> dict:
    """Compute binary segmentation metrics between ground truth and prediction masks.

    Args:
        ground_truth: Binary ground truth mask (H x W), values in {0, 1} or {0, 255}.
        prediction: Binary prediction mask (H x W), values in {0, 1} or {0, 255}.

    Returns:
        Dictionary with TP, FP, TN, FN, Accuracy, Precision, Recall, F1-Score, IoU.
    """
    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    tp = int(np.sum(np.logical_and(gt, pred)))
    tn = int(np.sum(np.logical_and(~gt, ~pred)))
    fp = int(np.sum(np.logical_and(~gt, pred)))
    fn = int(np.sum(np.logical_and(gt, ~pred)))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "IoU": iou,
    }


def compute_miou(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    num_classes: int = 2,
) -> float:
    """Compute mean Intersection-over-Union across all classes.

    Args:
        gt_masks: List of ground truth masks (H x W) with class IDs as pixel values.
        pred_masks: List of predicted masks (H x W) with class IDs as pixel values.
        num_classes: Number of classes (including background).

    Returns:
        Mean IoU across all classes.
    """
    class_ious = []

    for cls in range(num_classes):
        intersection = 0
        union = 0

        for gt, pred in zip(gt_masks, pred_masks):
            gt_cls = (gt == cls) if gt.max() > 1 else (gt > 0) if cls == 1 else (gt == 0)
            pred_cls = (pred == cls) if pred.max() > 1 else (pred > 0) if cls == 1 else (pred == 0)

            intersection += np.sum(np.logical_and(gt_cls, pred_cls))
            union += np.sum(np.logical_or(gt_cls, pred_cls))

        if union > 0:
            class_ious.append(intersection / union)

    return float(np.mean(class_ious)) if class_ious else 0.0


def compute_iou_single(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    intersection = np.sum(np.logical_and(m1, m2))
    union = np.sum(np.logical_or(m1, m2))
    return float(intersection / union) if union > 0 else 0.0


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_thresholds: list[float] = None,
) -> dict:
    """Compute mean Average Precision at multiple IoU thresholds (mAP50-95).

    Args:
        predictions: List of prediction dicts with keys:
            - "masks": list of binary masks (np.ndarray)
            - "scores": list of confidence scores
        ground_truths: List of ground truth dicts with keys:
            - "masks": list of binary masks (np.ndarray)
        iou_thresholds: IoU thresholds for AP computation.
            Defaults to [0.50, 0.55, ..., 0.95].

    Returns:
        Dictionary with per-threshold AP and mAP50-95.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.50 + 0.05 * i for i in range(10)]

    results = {}
    aps_per_threshold = []

    for thresh in iou_thresholds:
        all_tp = []
        all_scores = []
        total_gt = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_masks = pred["masks"]
            pred_scores = pred["scores"]
            gt_masks = gt["masks"]
            total_gt += len(gt_masks)

            # Sort predictions by score (descending)
            sorted_indices = np.argsort(pred_scores)[::-1]
            matched_gt = set()

            for idx in sorted_indices:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt_mask in enumerate(gt_masks):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou_single(pred_masks[idx], gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= thresh and best_gt_idx >= 0:
                    all_tp.append(1)
                    matched_gt.add(best_gt_idx)
                else:
                    all_tp.append(0)
                all_scores.append(pred_scores[idx])

        # Compute AP using precision-recall curve
        if total_gt == 0:
            ap = 0.0
        else:
            sorted_indices = np.argsort(all_scores)[::-1]
            tp_cumsum = np.cumsum([all_tp[i] for i in sorted_indices])
            fp_cumsum = np.cumsum([1 - all_tp[i] for i in sorted_indices])

            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            recalls = tp_cumsum / total_gt

            # Append sentinel values
            precisions = np.concatenate([[1.0], precisions])
            recalls = np.concatenate([[0.0], recalls])

            # Make precision monotonically decreasing
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])

            # Compute area under curve
            indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
            ap = float(np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices]))

        results[f"AP@{thresh:.2f}"] = ap
        aps_per_threshold.append(ap)

    results["mAP50-95"] = float(np.mean(aps_per_threshold))
    results["mAP50"] = aps_per_threshold[0] if aps_per_threshold else 0.0

    return results


def aggregate_metrics(
    metrics_list: list[dict],
) -> dict:
    """Aggregate metrics across multiple samples by averaging.

    Args:
        metrics_list: List of metric dictionaries from compute_binary_metrics.

    Returns:
        Dictionary with averaged metrics.
    """
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    return {
        key: float(np.mean([m[key] for m in metrics_list]))
        for key in keys
    }

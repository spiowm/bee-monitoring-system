"""Per-frame aggregate behavior evaluation.

Порівнює GT behavior labels з predicted behaviors на рівні кадрів.
Для кожного кадру рахуємо кількість бджіл з кожним GT/pred label,
потім обчислюємо per-class TP/FP/FN та confusion matrix.

Підхід: per-frame aggregate (не per-track) — стійкіший до помилок трекера.
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# GT behavior column → our behavior name
_GT_TO_PRED = {
    "arrival": "foraging",
    "defensive": "defense",
    "fanning": "fanning",
    "washboarding": "washboarding",
}


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """(N,4) × (M,4) → (N,M) IoU."""
    x_left = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y_top = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x_right = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y_bottom = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_behavior_metrics(
    gt_df: pd.DataFrame,
    pred_per_frame: dict[int, dict[int, dict]],
    eval_classes: list[str],
    iou_threshold: float = 0.3,
) -> dict:
    """Compare GT behaviors with predicted behaviors frame-by-frame.

    Parameters
    ----------
    gt_df : DataFrame with columns: frame, track_id, x1, y1, x2, y2, gt_behavior
            (after denormalize + load_gt_behaviors)
    pred_per_frame : {frame_num: {track_id: {"bbox": [x1,y1,x2,y2], "behavior": str}}}
    eval_classes : list of behavior classes to evaluate (e.g. ["foraging", "defense", "fanning"])
    iou_threshold : minimum IoU to consider a GT↔pred match

    Returns
    -------
    dict with keys:
      - per_class: {class_name: {tp, fp, fn, precision, recall, f1, gt_count, pred_count}}
      - confusion_matrix: {gt_class: {pred_class: count}}
      - overall_accuracy: float
      - total_gt_labeled: int (GT frames×tracks with a behavior != idle)
      - total_matched: int (successfully matched GT↔pred pairs)
    """
    # Per-class counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # Confusion matrix: cm[gt_class][pred_class] = count
    cm = defaultdict(lambda: defaultdict(int))
    total_matched = 0
    total_gt_labeled = 0

    frames = sorted(gt_df["frame"].unique())

    for frame_num in frames:
        gt_frame = gt_df[gt_df["frame"] == frame_num]
        pred_frame = pred_per_frame.get(frame_num, {})

        # Filter GT to only rows with a behavior we're evaluating
        gt_labeled = gt_frame[gt_frame["gt_behavior"].isin(eval_classes)]
        total_gt_labeled += len(gt_labeled)

        if len(gt_labeled) == 0:
            # Count any predicted behaviors in eval_classes as FP
            for tid, pdata in pred_frame.items():
                pred_b = pdata.get("behavior")
                if pred_b in eval_classes:
                    fp[pred_b] += 1
            continue

        if len(pred_frame) == 0:
            # All GT labeled bees are FN
            for _, row in gt_labeled.iterrows():
                fn[row["gt_behavior"]] += 1
            continue

        # Build bbox arrays for IoU matching
        gt_boxes = gt_labeled[["x1", "y1", "x2", "y2"]].to_numpy()
        pred_items = list(pred_frame.values())
        pred_boxes = np.array([p["bbox"] for p in pred_items])

        iou = _iou_matrix(gt_boxes, pred_boxes)

        # Greedy matching: for each GT bee, find best pred match
        pred_used = set()
        gt_behaviors = gt_labeled["gt_behavior"].to_list()

        for gi in range(len(gt_boxes)):
            gt_b = gt_behaviors[gi]
            best_pi = -1
            best_iou = iou_threshold
            for pi in range(len(pred_boxes)):
                if pi in pred_used:
                    continue
                if iou[gi, pi] > best_iou:
                    best_iou = iou[gi, pi]
                    best_pi = pi

            if best_pi >= 0:
                pred_used.add(best_pi)
                pred_b = pred_items[best_pi].get("behavior") or "unknown"
                total_matched += 1
                cm[gt_b][pred_b] += 1

                if pred_b == gt_b:
                    tp[gt_b] += 1
                else:
                    fn[gt_b] += 1
                    if pred_b in eval_classes:
                        fp[pred_b] += 1
            else:
                # GT bee not matched → FN
                fn[gt_b] += 1

        # Unmatched predictions → FP
        for pi in range(len(pred_items)):
            if pi not in pred_used:
                pred_b = pred_items[pi].get("behavior") or "unknown"
                if pred_b in eval_classes:
                    fp[pred_b] += 1

    # Build per-class metrics
    per_class = {}
    for cls in eval_classes:
        t, f_p, f_n = tp[cls], fp[cls], fn[cls]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        gt_count = t + f_n
        pred_count = t + f_p
        per_class[cls] = {
            "tp": t, "fp": f_p, "fn": f_n,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "gt_count": gt_count,
            "pred_count": pred_count,
        }

    total_tp = sum(tp.values())
    total_decisions = total_tp + sum(fp.values()) + sum(fn.values())
    overall_acc = total_tp / total_decisions if total_decisions > 0 else 0.0

    # Format confusion matrix
    cm_dict = {}
    all_labels = sorted(set(eval_classes) | set(
        pred_b for gt_map in cm.values() for pred_b in gt_map.keys()
    ))
    for gt_cls in eval_classes:
        cm_dict[gt_cls] = {pred_cls: cm[gt_cls][pred_cls] for pred_cls in all_labels}

    return {
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "overall_accuracy": round(overall_acc, 4),
        "total_gt_labeled": total_gt_labeled,
        "total_matched": total_matched,
        "eval_classes": eval_classes,
    }

"""Перерахунок кутової похибки орієнтації на валідаційних вуликах.

Потрібен, бо MLflow зберігає лише агрегати (mean/median/within), а для гістограми
статті потрібен масив індивідуальних похибок. Логіка зіставлення та формула
кругової різниці — ті самі, що в `run_experiment.py`.

Запуск з research/:
    uv run python src/angular_error_eval.py \
        --weights ../backend/data/models/bee_pose/best.pt --imgsz 1920
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
POSE = ROOT / "datasets" / "raw" / "pose"
VAL_HIVES = ("20230711b", "20230609e")  # hive-based split, hold-out


def load_gt(label_path: Path):
    """→ (boxes_xyxy_norm, kpts_norm) у нормалізованих координатах."""
    boxes, kpts = [], []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 9:
            continue
        cx, cy, w, h = map(float, p[1:5])
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        kpts.append([[float(p[5]), float(p[6])], [float(p[7]), float(p[8])]])
    return np.array(boxes, dtype=np.float32), np.array(kpts, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="angular_errors.json")
    args = ap.parse_args()

    images = sorted(p for p in (POSE / "images").glob("*.jpg")
                    if p.stem[:9] in VAL_HIVES)
    print(f"валідаційних зображень: {len(images)} (вулики {', '.join(VAL_HIVES)})")

    model = YOLO(args.weights)
    errors, nme_vals, total_gt = [], [], 0

    for img in images:
        lbl = POSE / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        gt_boxes, gt_kpts = load_gt(lbl)
        if len(gt_boxes) == 0:
            continue
        total_gt += len(gt_boxes)

        res = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        h, w = res.orig_shape
        pb = res.boxes.xyxy.cpu().numpy() / np.array([w, h, w, h], dtype=np.float32)
        pk = res.keypoints.xy.cpu().numpy() / np.array([w, h], dtype=np.float32)

        available = list(range(len(pb)))
        for gb, gk in zip(gt_boxes, gt_kpts):
            if not available:
                break
            cand = pb[available]
            ix1 = np.maximum(gb[0], cand[:, 0]); iy1 = np.maximum(gb[1], cand[:, 1])
            ix2 = np.minimum(gb[2], cand[:, 2]); iy2 = np.minimum(gb[3], cand[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            ga = (gb[2] - gb[0]) * (gb[3] - gb[1])
            pa = (cand[:, 2] - cand[:, 0]) * (cand[:, 3] - cand[:, 1])
            iou = inter / (ga + pa - inter + 1e-6)
            best = int(np.argmax(iou))
            if iou[best] < 0.5:
                continue
            k = pk[available.pop(best)]
            if np.any(np.all(k == 0, axis=1)):
                continue

            diag = float(np.sqrt((gb[2] - gb[0]) ** 2 + (gb[3] - gb[1]) ** 2))
            if diag < 1e-6:
                continue
            nme_vals.append(float(np.mean(np.linalg.norm(k - gk, axis=1)) / diag))

            gv, pv = gk[0] - gk[1], k[0] - k[1]
            if np.linalg.norm(gv) < 1e-6 or np.linalg.norm(pv) < 1e-6:
                continue
            d = abs(np.arctan2(gv[1], gv[0]) - np.arctan2(pv[1], pv[0]))
            errors.append(float(np.degrees(min(d, 2 * np.pi - d))))

    a = np.array(errors)
    summary = {
        "n_matched": len(nme_vals), "n_total": total_gt,
        "coverage_pct": round(len(nme_vals) / max(total_gt, 1) * 100, 1),
        "angular_mean": round(float(a.mean()), 2),
        "angular_median": round(float(np.median(a)), 2),
        "within_15": round(float((a < 15).mean() * 100), 1),
        "within_30": round(float((a < 30).mean() * 100), 1),
        "nme_mean": round(float(np.mean(nme_vals)), 4),
        "nme_median": round(float(np.median(nme_vals)), 4),
        "errors": errors,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False))
    for k, v in summary.items():
        if k != "errors":
            print(f"  {k:<16} {v}")
    print(f"\nЗбережено {len(errors)} значень → {args.out}")


if __name__ == "__main__":
    main()

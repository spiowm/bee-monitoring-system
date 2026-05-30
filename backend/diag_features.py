"""Діагностика розподілів ознак behavior-класифікації (TP vs FP vs FN).

Прогоняє пайплайн із кешу, збирає per-track фічі (zcr, avg_motion, max_disp,
avg_speed, duration, body_aligned), матчить треки до GT за центром і друкує
розподіли по групах (gt_behavior × pred_behavior). Допомагає зрозуміти, який
ознак-поріг реально розділяє класи.

    uv run python diag_features.py --pair 20230609b-def
    uv run python diag_features.py --pair 20230711a-fan --frames 3000
"""
import argparse
import logging
from collections import defaultdict

import cv2
import numpy as np

logging.basicConfig(level=logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

FEATS = ["zcr", "avg_motion", "max_disp", "avg_speed", "duration"]


def _pct(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else 0.0


def _summary(records, feats=FEATS):
    if not records:
        return "  (порожньо)"
    lines = []
    for f in feats:
        xs = np.array([r[f] for r in records])
        lines.append(f"{f:>11}: med={np.median(xs):7.2f}  p25={_pct(xs,25):7.2f}  "
                     f"p75={_pct(xs,75):7.2f}  max={xs.max():8.2f}")
    ba = np.mean([r["body_aligned"] for r in records]) * 100
    hb = np.mean([r["has_body"] for r in records]) * 100
    lines.append(f"  body_aligned={ba:.0f}%  has_body={hb:.0f}%  n={len(records)}")
    return "\n".join("    " + l for l in lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--match-dist", type=float, default=30.0)
    ap.add_argument("--config", default="config/eval_config.yaml")
    args = ap.parse_args()

    from config import settings
    from services.evaluation.gt_loader import gt_paths, load_gt_behaviors, denormalize, load_entrance_zone
    from services.pipeline import VideoPipeline
    from services.pipeline_stages import BehaviorStage
    import detection_cache as dc
    from eval_fast import base_config, load_yaml_config

    paths = gt_paths(args.pair)
    video_path = str(paths["video"])
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    limit = args.frames if args.frames and args.frames > 0 else total
    limit = min(limit, total)

    config = base_config(80)
    config.update(load_yaml_config(args.config))

    if not dc.is_valid(args.pair, config["conf_threshold"], config["iou_threshold"],
                       config["max_detections"], config.get("imgsz") or 640, limit,
                       settings.MODEL_PATH):
        raise SystemExit("кеш відсутній/невалідний — спершу запусти eval_fast.py --rebuild")
    cached = dc.load_cache(args.pair)

    zone = load_entrance_zone(paths["entrance_zone"])
    gt_df = denormalize(load_gt_behaviors(paths["tracks"]), width, height)
    gt_df = gt_df[gt_df["frame"] <= limit]
    # frame -> (centers[N,2], behaviors[N])
    gt_by_frame = {}
    for fr, grp in gt_df.groupby("frame"):
        gt_by_frame[int(fr)] = (grp[["cx_px", "cy_px"]].to_numpy(),
                                grp["gt_behavior"].to_list())

    pipeline = VideoPipeline(None, config, {}, gt_entrance_zone=zone)
    strat = None
    for st in pipeline.stages:
        if isinstance(st, BehaviorStage):
            strat = st.analyzer.strategy
    strat.debug_enabled = True

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= limit:
            break
        det = cached[frame_num] if frame_num < len(cached) else None
        frame_num += 1
        strat.debug_frame = frame_num
        pipeline.process_frame(frame, frame_num, fps, detection_result=det)
    cap.release()

    # Match each debug record to nearest GT center at that frame
    buckets = defaultdict(list)  # (gt, pred) -> [records]
    for r in strat.debug_records:
        gt_label = "none"
        gd = gt_by_frame.get(r["frame"])
        if gd is not None:
            centers, labels = gd
            d = np.sqrt((centers[:, 0] - r["cx"])**2 + (centers[:, 1] - r["cy"])**2)
            j = int(np.argmin(d))
            if d[j] <= args.match_dist:
                gt_label = labels[j]
        buckets[(gt_label, r["behavior"])].append(r)

    print(f"\n=== {args.pair}  frames={limit}  records={len(strat.debug_records)} ===")

    # Аналіз FANNING: TP (gt=fan,pred=fan) vs FP (gt≠fan,pred=fan) vs FN (gt=fan,pred≠fan)
    print("\n########## FANNING precision (pred=fanning) ##########")
    tp_fan = buckets[("fanning", "fanning")]
    print(f"\n[TP]  gt=fanning, pred=fanning  (n={len(tp_fan)})")
    print(_summary(tp_fan))
    fp_sources = defaultdict(list)
    for (gt, pred), recs in buckets.items():
        if pred == "fanning" and gt != "fanning":
            fp_sources[gt].extend(recs)
    for gt in sorted(fp_sources, key=lambda g: -len(fp_sources[g])):
        recs = fp_sources[gt]
        print(f"\n[FP]  gt={gt}, pred=fanning  (n={len(recs)})")
        print(_summary(recs))

    print("\n########## FANNING recall (gt=fanning) ##########")
    fn_targets = defaultdict(list)
    for (gt, pred), recs in buckets.items():
        if gt == "fanning" and pred != "fanning":
            fn_targets[pred].extend(recs)
    for pred in sorted(fn_targets, key=lambda p: -len(fn_targets[p])):
        recs = fn_targets[pred]
        print(f"\n[FN]  gt=fanning, pred={pred}  (n={len(recs)})")
        print(_summary(recs))

    # Загальний розподіл pred по gt
    print("\n########## Розподіл pred по gt-класах ##########")
    gt_totals = defaultdict(lambda: defaultdict(int))
    for (gt, pred), recs in buckets.items():
        gt_totals[gt][pred] += len(recs)
    for gt in sorted(gt_totals):
        total_g = sum(gt_totals[gt].values())
        dist = "  ".join(f"{p}={n}({100*n/total_g:.0f}%)"
                         for p, n in sorted(gt_totals[gt].items(), key=lambda x: -x[1]))
        print(f"  gt={gt:<13} (n={total_g}): {dist}")


if __name__ == "__main__":
    main()

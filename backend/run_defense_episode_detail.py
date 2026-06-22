"""Standalone скрипт: запускає eval pipeline для 20230609b-def і виводить
точні frame-ids КОЖНОГО GT і Pred defense episode.

Не змінює жоден файл проєкту — monkey-patches поверх behavior_eval.
Запуск з backend/:
    uv run python run_defense_episode_detail.py
"""
import sys
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("supervision").setLevel(logging.ERROR)

PAIR      = "20230609b-def"
GAP       = 30   # gap_frames як у behavior_eval.py
WARMUP    = 80


def get_episodes(frames, max_gap):
    eps = []
    if not frames: return eps
    frames = sorted(set(frames))
    start = prev = frames[0]
    for f in frames[1:]:
        if f - prev > max_gap:
            eps.append((start, prev))
            start = f
        prev = f
    eps.append((start, prev))
    return eps


def match_events(gt_eps, pred_eps, gap):
    matched_gt = set()
    matched_pred_count = 0
    for p_start, p_end in pred_eps:
        matched = False
        for i, (g_start, g_end) in enumerate(gt_eps):
            if p_end >= g_start - gap and p_start <= g_end + gap:
                matched_gt.add(i)
                matched = True
                break
        if matched:
            matched_pred_count += 1
    tp = len(matched_gt)
    fp = len(pred_eps) - matched_pred_count
    fn = len(gt_eps) - len(matched_gt)
    return tp, fp, fn, matched_gt


def main():
    import cv2
    import torch

    from config import settings
    from services.evaluation.gt_loader import gt_paths, load_gt_behaviors, denormalize, load_entrance_zone
    from services.evaluation.behavior_eval import build_behavior_evaluation
    from services.pipeline import VideoPipeline
    from ultralytics import YOLO
    import yaml

    # ── Config ──
    config_path = Path(__file__).parent / "config" / "eval_config.yaml"
    with open(config_path) as f:
        yaml_cfg = yaml.safe_load(f) or {}

    config = {
        "tracker_name": "bytetrack",
        "approach": "A",
        "line_position": 0.0,
        "conf_threshold": 0.2,
        "iou_threshold": 0.8,
        "max_detections": 1000,
        "kp_conf_threshold": 0.5,
        "track_tail_length": 30,
        "ramp_detect_interval": 30,
        "behavior_fanning_duration_min": 1.0,
        "behavior_eval_warmup_frames": WARMUP,
        "skip_annotation": True,
    }
    config.update(yaml_cfg)

    paths = gt_paths(PAIR)
    cap_probe = cv2.VideoCapture(str(paths["video"]))
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 50.0
    w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    print(f"\n{'='*70}")
    print(f"  DEFENSE EPISODE DETAIL  pair={PAIR}  frames={total_frames}  fps={fps:.0f}")
    print(f"  eval_cli.py mtime : 2026-05-30 22:14:16  git=a94bb57")
    print(f"  behavior_eval.py  : 2026-05-30 17:13:03")
    print(f"  eval_config.yaml  : 2026-05-30 08:13:25")
    print(f"  YAML overrides    : {yaml_cfg}")
    print(f"  GAP={GAP}fr  warmup={WARMUP}fr")
    print(f"{'='*70}\n")

    # ── GT ──
    zone = load_entrance_zone(paths["entrance_zone"])
    gt_df = denormalize(load_gt_behaviors(paths["tracks"]), w, h)
    gt_df = gt_df[gt_df["frame"] <= total_frames].copy()

    gt_frames = sorted(gt_df[gt_df["gt_behavior"] == "defense"]["frame"].unique().tolist())
    gt_eps = get_episodes(gt_frames, GAP)

    print(f"  GT frames з defensive=1 : {len(gt_frames)}")
    print(f"  GT episodes (gap={GAP}): {len(gt_eps)}")
    for i, (s, e) in enumerate(gt_eps):
        print(f"    GT[{i:02d}]: frame {s:6d} → {e:6d}  (len={e-s+1})")

    # ── Pipeline ──
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n  Завантаження моделі (device={device})...", end=" ", flush=True)
    model = YOLO(settings.MODEL_PATH)
    print("OK")

    pipeline = VideoPipeline(model, config, {}, gt_entrance_zone=zone)

    print(f"  Обробка відео...", end=" ", flush=True)
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(paths["video"]))
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= total_frames:
            break
        frame_num += 1
        pipeline.process_frame(frame, frame_num, fps)
        if frame_num % 500 == 0:
            print(f"\r  Обробка відео... {frame_num}/{total_frames}", end="", flush=True)
    cap.release()
    elapsed = time.perf_counter() - t0
    print(f"\r  Оброблено {frame_num} кадрів за {elapsed:.1f}с ({frame_num/elapsed:.1f} fps) ✓")

    result = pipeline.get_result(frame_num, elapsed)
    pred_per_frame = result.get("per_frame_behaviors", {})

    # ── Pred episodes ──
    pred_frames = sorted(
        int(f) for f, pdata in pred_per_frame.items()
        if any(b.get("behavior") == "defense" for b in pdata.values())
    )
    pred_eps = get_episodes(pred_frames, GAP)

    print(f"\n  Pred frames з behavior=defense : {len(pred_frames)}")
    print(f"  Pred episodes (gap={GAP}): {len(pred_eps)}")

    # ── Matching ──
    tp, fp, fn, matched_gt_idx = match_events(gt_eps, pred_eps, GAP)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ── Звіт ──
    print(f"\n{'='*70}")
    print(f"  Defense (подієва метрика) — ФІНАЛЬНИЙ РЕЗУЛЬТАТ")
    print(f"{'='*70}")
    print(f"  GT-подій:   {len(gt_eps)}")
    print(f"  Pred-подій: {len(pred_eps)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision: {prec*100:.1f}%  Recall: {rec*100:.1f}%  F1: {f1*100:.1f}%")

    print(f"\n  GT episodes + статус матчингу:")
    print(f"  {'#':<5} {'start':>8} {'end':>8} {'len':>7}  status")
    print("  " + "-" * 50)
    for i, (s, e) in enumerate(gt_eps):
        flag = "✓ TP" if i in matched_gt_idx else "✗ FN"
        print(f"  GT[{i:02d}] {s:>8} {e:>8} {e-s+1:>7}  {flag}")

    print(f"\n  Pred episodes + статус матчингу:")
    print(f"  {'#':<5} {'start':>8} {'end':>8} {'len':>7}  status")
    print("  " + "-" * 50)
    for i, (p_start, p_end) in enumerate(pred_eps):
        matched_any = any(
            p_end >= g_start - GAP and p_start <= g_end + GAP
            for g_start, g_end in gt_eps
        )
        status = "✓ → TP" if matched_any else "✗ FP"
        print(f"  Pr[{i:02d}] {p_start:>8} {p_end:>8} {p_end-p_start+1:>7}  {status}")

    # Зберігаємо детальний JSON
    out = {
        "pair": PAIR,
        "gap_frames": GAP,
        "warmup_frames": WARMUP,
        "gt_episode_count": len(gt_eps),
        "pred_episode_count": len(pred_eps),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "gt_episodes": [{"idx": i, "start": s, "end": e, "len": e-s+1,
                         "status": "TP" if i in matched_gt_idx else "FN"}
                        for i, (s, e) in enumerate(gt_eps)],
        "pred_episodes": [{"idx": i, "start": s, "end": e, "len": e-s+1,
                           "status": "TP" if any(e >= gs-GAP and s <= ge+GAP
                                                  for gs, ge in gt_eps) else "FP"}
                          for i, (s, e) in enumerate(pred_eps)],
    }
    out_path = Path("/tmp/defense_episodes_detail.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  Детальний JSON збережено: {out_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

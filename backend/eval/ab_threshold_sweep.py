"""Чи є у pose-фільтра взагалі робоча точка, і на чиєму боці розсинхрон.

Два питання, на які відповідає скрипт:

1. Розгортка порогу. Поріг 60° узятий апріорно. Збираємо кожного кандидата на
   перетин ОДИН раз, далі фільтруємо офлайн за різними порогами. Якщо F1 росте
   монотонно до 180° (= Підхід A), робочої точки не існує в принципі.

2. Декомпозиція. Фільтр порівнює вектор РУХУ (EMA по трекеру) з вектором ПОЗИ
   (жало→голова). Розсинхрон може давати будь-яка зі сторін. Тому для кожного
   кандидата рахуємо кут не лише з EMA, а й з переміщенням на довшому вікні —
   і дивимось на залежність від довжини треку. Якщо на довгих чистих треках
   узгодження добре, винен трекер, а не модель пози.

Запуск з backend/:
    uv run python -m eval.ab_threshold_sweep --cache-dir data/eval_cache_1920
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import detection_cache as dc  # noqa: E402
from eval.ab_counting_eval import (FPS, PAIRS, gt_events, line_y_at,  # noqa: E402
                                   load_zone, match)
from services.orientation import get_orientation_vector  # noqa: E402
from services.track_history import TrackHistory  # noqa: E402

DEBOUNCE = 45  # settings.counting_debounce_frames


def angle_between(u, v) -> float | None:
    if u is None or v is None:
        return None
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u / nu, v / nv), -1, 1))))


def collect(pair: str, zone: np.ndarray, shuffle: bool = False) -> list[dict]:
    """Усі кандидати на перетин з набором кутів і контекстом треку."""
    cached = dc.load_cache(pair)
    tracker = sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=100,
                           minimum_matching_threshold=0.9,
                           minimum_consecutive_frames=1, frame_rate=int(FPS))
    history = TrackHistory()
    ramp = [zone[0], zone[1]]
    last_counted: dict[int, int] = {}
    seen: dict[int, int] = {}
    cands = []

    for i in range(len(cached)):
        frame_num = i + 1
        res = cached[i]
        boxes = res.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            continue
        kpts = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
        det = sv.Detections(xyxy=boxes, confidence=res.boxes.conf.cpu().numpy(),
                            class_id=np.zeros(len(boxes), dtype=int))
        if kpts is not None and len(kpts) == len(boxes):
            det.data["kpt_idx"] = np.arange(len(boxes))
        tracked = tracker.update_with_detections(det)
        if tracked.tracker_id is None or len(tracked) == 0:
            continue
        idx = tracked.data.get("kpt_idx") if "kpt_idx" in tracked.data else None
        if shuffle and idx is not None:
            idx = np.random.permutation(np.asarray(idx))

        for j, tid in enumerate(tracked.tracker_id):
            tid = int(tid)
            x1, y1, x2, y2 = tracked.xyxy[j]
            cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
            history.update(tid, cx, cy, frame_num)
            seen[tid] = seen.get(tid, 0) + 1

            entry = history.get(tid)
            if entry is None or len(entry.positions) < 2:
                continue
            pos = entry.positions
            prev_y, curr_y = pos[-2][1], pos[-1][1]
            ly = line_y_at(zone, cx)
            if not ((prev_y < ly <= curr_y) or (prev_y > ly >= curr_y)):
                continue
            if frame_num - last_counted.get(tid, -10 ** 9) < DEBOUNCE:
                continue
            last_counted[tid] = frame_num

            m = entry.compute_metrics(FPS)
            ema = np.array(m["ema_dir_vec"], dtype=np.float32)
            k = min(10, len(pos))
            longvec = np.array([pos[-1][0] - pos[-k][0], pos[-1][1] - pos[-k][1]],
                               dtype=np.float32)
            kp = kpts[idx[j]] if (kpts is not None and idx is not None) else None
            orient = get_orientation_vector(kp)

            cands.append({
                "frame": frame_num, "track_id": tid,
                "direction": "OUT" if curr_y > prev_y else "IN",
                "angle_ema": angle_between(ema, orient),
                "angle_long": angle_between(longvec, orient),
                "track_len": seen[tid],
                "speed": float(m["current_speed"]),
                "disp": float(np.linalg.norm(longvec)),
            })
    return cands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--out", default="ab_threshold_sweep.json")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()
    if args.cache_dir:
        dc.CACHE_DIR = Path(args.cache_dir)
    pairs = args.pair or PAIRS
    THRESHOLDS = [15, 30, 45, 60, 90, 120, 150, 180]
    report = {}

    for pair in pairs:
        zone = load_zone(pair)
        cands = collect(pair, zone, shuffle=args.shuffle)
        gt = gt_events(pair, zone, 10 ** 9)
        print(f"\n{'='*78}\n  {pair}   GT={len(gt)}   кандидатів={len(cands)}\n{'='*78}")

        print(f"  {'поріг':>7}{'подій':>8}{'P':>9}{'R':>9}{'F1':>9}   (кут EMA↔поза)")
        rows = {}
        for t in THRESHOLDS:
            kept = [c for c in cands if c["angle_ema"] is None or c["angle_ema"] <= t]
            m = match(gt, kept, args.window)
            rows[t] = m
            star = "  ← Підхід A" if t == 180 else ("  ← поточний" if t == 60 else "")
            print(f"  {t:>6}°{m['pred']:>8}{m['precision']*100:>8.1f}%"
                  f"{m['recall']*100:>8.1f}%{m['f1']*100:>8.1f}%{star}")

        best = max(rows, key=lambda t: rows[t]["f1"])
        print(f"  найкращий поріг за F1: {best}° (F1={rows[best]['f1']*100:.1f}%)")

        ang = np.array([c["angle_ema"] for c in cands if c["angle_ema"] is not None])
        print(f"\n  Розподіл кута EMA↔поза: медіана={np.median(ang):.0f}°  "
              f"частка ≤60°={np.mean(ang <= 60)*100:.0f}%   "
              f"(при випадковій позі очікувано ≈33%)")

        al = np.array([c["angle_long"] for c in cands if c["angle_long"] is not None])
        print(f"  Кут на довгому вікні (10 кадрів): медіана={np.median(al):.0f}°  "
              f"частка ≤60°={np.mean(al <= 60)*100:.0f}%")

        print(f"\n  {'довжина треку':<18}{'n':>6}{'медіана кута':>15}{'≤60°':>8}")
        for lo, hi, name in [(0, 10, "1-10 кадрів"), (10, 30, "10-30"),
                             (30, 100, "30-100"), (100, 10**9, "100+")]:
            sub = [c["angle_ema"] for c in cands
                   if c["angle_ema"] is not None and lo < c["track_len"] <= hi]
            if sub:
                s = np.array(sub)
                print(f"  {name:<18}{len(s):>6}{np.median(s):>14.0f}°{np.mean(s<=60)*100:>7.0f}%")

        report[pair] = {"gt": len(gt), "candidates": len(cands),
                        "sweep": {str(t): rows[t] for t in THRESHOLDS},
                        "best_threshold": best,
                        "frames": [c["frame"] for c in cands],
                        "directions": [c["direction"] for c in cands],
                        "angles_ema": [c["angle_ema"] for c in cands],
                        "angles_long": [c["angle_long"] for c in cands],
                        "track_len": [c["track_len"] for c in cands]}

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False))
    print(f"\nЗбережено: {args.out}")


if __name__ == "__main__":
    np.random.seed(42)
    main()

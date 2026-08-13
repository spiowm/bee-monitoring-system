"""Парне порівняння підходів підрахунку A (траєкторія) і B (pose-верифікація).

GT подій перетину в датасеті немає — `tracks_and_behavior.txt` містить поведінкові
мітки, а не події. Тому GT подій виводиться з САМИХ розмічених треків: беремо
покадрові позиції кожного GT-треку і фіксуємо стійкі перетини лінії льотка
(≥ STABLE кадрів по обидва боки + дебаунс). Розмітка людська, ID стабільні —
це надійніший орієнтир, ніж будь-яка похідна від нашого ж трекера.

Порівняння парне: один прохід трекінгу, ті самі детекції й ті самі треки
подаються обом лічильникам. Різниця між A і B — виключно pose-фільтр,
тож роздільна здатність кешу впливає на абсолютний рівень, але не на дельту.

Запуск з backend/:
    uv run python -m eval.ab_counting_eval
    uv run python -m eval.ab_counting_eval --pair 20230711b-fan --window 15
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import detection_cache as dc  # noqa: E402
from services.counter import TrafficCounter  # noqa: E402
from services.track_history import TrackHistory  # noqa: E402

GT_ROOT = Path(__file__).resolve().parent.parent.parent / \
    "research" / "datasets" / "raw" / "tracking_and_behavior"
GT_COLUMNS = ["frame", "track_id", "cx", "cy", "w", "h",
              "arrival", "defensive", "fanning", "washboarding"]

PAIRS = ["20230711a-fan", "20230711b-fan", "20230609b-def"]
WIDTH, HEIGHT, FPS = 1920, 1080, 50.0
SHIFT_DOWN = 8      # як у counter.get_line_y
STABLE = 5          # кадрів по обидва боки лінії, щоб перетин вважався справжнім
GT_DEBOUNCE = 30    # кадрів між двома подіями одного GT-треку


# ── GT ──────────────────────────────────────────────────────────────────────

def load_zone(pair: str) -> np.ndarray:
    nums = re.findall(r"-?\d+\.?\d*", (GT_ROOT / pair / "entrance_zone.txt").read_text())
    return np.array([float(n) for n in nums[:8]], dtype=np.float32).reshape(4, 2)


def line_y_at(zone: np.ndarray, cx: float) -> float:
    (x1, y1), (x2, y2) = zone[0], zone[1]
    if abs(x2 - x1) < 1e-3:
        return float(y1 + SHIFT_DOWN)
    return float(y1 + (y2 - y1) / (x2 - x1) * (cx - x1) + SHIFT_DOWN)


def gt_events(pair: str, zone: np.ndarray, limit: int) -> list[dict]:
    """Стійкі перетини лінії, виведені з розмічених GT-треків."""
    df = pd.read_csv(GT_ROOT / pair / "tracks_and_behavior.txt", header=None, names=GT_COLUMNS)
    df = df[df.frame <= limit]
    events = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        cx = g.cx.values * WIDTH
        cy = g.cy.values * HEIGHT
        fr = g.frame.values
        side = np.sign(cy - np.array([line_y_at(zone, x) for x in cx]))
        keep = side != 0
        s, f = side[keep], fr[keep]
        if len(s) < 2:
            continue
        last = -10 ** 9
        for k in np.where(np.diff(s) != 0)[0]:
            if f[k + 1] - last < GT_DEBOUNCE:
                continue
            before = int(np.sum(s[max(0, k - STABLE + 1):k + 1] == s[k]))
            after = int(np.sum(s[k + 1:k + 1 + STABLE] == s[k + 1]))
            if before < STABLE or after < STABLE:
                continue
            last = int(f[k + 1])
            events.append({"frame": int(f[k + 1]), "track_id": int(tid),
                           "direction": "OUT" if s[k + 1] > 0 else "IN"})
    return sorted(events, key=lambda e: e["frame"])


# ── Прогін підрахунку ───────────────────────────────────────────────────────

def run_counters(pair: str, zone: np.ndarray, limit: int, shuffle_kpts: bool = False) -> dict:
    """Один прохід трекінгу, обидва лічильники на тих самих треках."""
    cached = dc.load_cache(pair)
    tracker = sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=100,
                           minimum_matching_threshold=0.9,
                           minimum_consecutive_frames=1, frame_rate=int(FPS))
    history = TrackHistory()
    counters = {"A": TrafficCounter(approach="A"), "B": TrafficCounter(approach="B")}
    out = {"A": [], "B": [], "rejected": []}
    ramp_kpts = [zone[0], zone[1]]

    n = min(len(cached), limit)
    for i in range(n):
        frame_num = i + 1
        res = cached[i]
        boxes = res.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            continue
        conf = res.boxes.conf.cpu().numpy()
        kpts = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None

        det = sv.Detections(xyxy=boxes, confidence=conf,
                            class_id=np.zeros(len(boxes), dtype=int))
        if kpts is not None and len(kpts) == len(boxes):
            det.data["kpt_idx"] = np.arange(len(boxes))
        tracked = tracker.update_with_detections(det)
        if tracked.tracker_id is None or len(tracked) == 0:
            continue

        idx = tracked.data.get("kpt_idx") if "kpt_idx" in tracked.data else None
        if kpts is not None and idx is not None:
            if shuffle_kpts:  # контроль: руйнуємо звʼязок keypoints↔трек
                idx = np.random.permutation(np.asarray(idx))
            tracked_kpts = [kpts[j] for j in idx]
        else:
            tracked_kpts = [None] * len(tracked)

        for j, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[j]
            history.update(int(tid), float((x1 + x2) / 2), float((y1 + y2) / 2), frame_num)

        for name, counter in counters.items():
            ev, rej = counter.update(frame_num, tracked, None, ramp_kpts,
                                     tracked_kpts, history, None, FPS)
            out[name].extend(ev)
            if name == "B":
                out["rejected"].extend(rej)
    return out


# ── Зіставлення подій ───────────────────────────────────────────────────────

def match(gt: list[dict], pred: list[dict], window: int) -> dict:
    """Жадібне зіставлення у часовому вікні; напрямок має збігатись."""
    used = [False] * len(gt)
    tp = 0
    matched_pred = []
    for p in sorted(pred, key=lambda e: e["frame"]):
        best, best_d = -1, window + 1
        for i, g in enumerate(gt):
            if used[i] or g["direction"] != p["direction"]:
                continue
            d = abs(g["frame"] - p["frame"])
            if d <= window and d < best_d:
                best, best_d = i, d
        if best >= 0:
            used[best] = True
            tp += 1
            matched_pred.append(True)
        else:
            matched_pred.append(False)
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"gt": len(gt), "pred": len(pred), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", help="повторюваний; типово всі три")
    ap.add_argument("--window", type=int, default=15, help="вікно зіставлення, кадрів")
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--out", default="ab_counting_results.json")
    ap.add_argument("--cache-dir", default=None, help="альтернативний кеш детекцій")
    ap.add_argument("--shuffle-control", action="store_true",
                    help="контроль: випадкова привʼязка keypoints до треків")
    args = ap.parse_args()
    if args.cache_dir:
        dc.CACHE_DIR = Path(args.cache_dir)
    pairs = args.pair or PAIRS

    results = {}
    for pair in pairs:
        zone = load_zone(pair)
        preds = run_counters(pair, zone, args.limit, shuffle_kpts=args.shuffle_control)
        limit = max([e["frame"] for e in preds["A"]] + [1])
        gt = gt_events(pair, zone, args.limit)
        row = {k: match(gt, preds[k], args.window) for k in ("A", "B")}
        rej = preds["rejected"]
        row["rejected_count"] = len(rej)
        row["rejected_share"] = len(rej) / max(1, len(preds["A"]))
        row["angles_rejected"] = [e["angle_deg"] for e in rej if e.get("angle_deg") is not None]
        results[pair] = row

        print(f"\n{'='*74}\n  {pair}   GT-подій: {row['A']['gt']}\n{'='*74}")
        print(f"  {'підхід':<10}{'pred':>7}{'TP':>6}{'FP':>6}{'FN':>6}"
              f"{'P':>9}{'R':>9}{'F1':>9}")
        for k in ("A", "B"):
            m = row[k]
            print(f"  {k:<10}{m['pred']:>7}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}"
                  f"{m['precision']*100:>8.1f}%{m['recall']*100:>8.1f}%{m['f1']*100:>8.1f}%")
        d = row["B"]["f1"] - row["A"]["f1"]
        print(f"  Δ F1 (B − A): {d*100:+.2f} в.п.   відхилено подій: {len(rej)} "
              f"({row['rejected_share']*100:.1f}% від A)")

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nЗбережено: {args.out}")


if __name__ == "__main__":
    np.random.seed(42)
    main()

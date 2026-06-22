"""Повний звіт defense episode-level: GT episodes + Pred episodes з frame-ids.
Відтворює рівно ту ж логіку що в _compute_event_metrics_for_class.

Запуск (з backend/):
    uv run python extract_defense_episodes_full.py /tmp/eval_def_result.json
"""
import json
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────── Константи ───────────────
PAIR      = "20230609b-def"
GAP       = 30          # gap_frames як у behavior_eval.py
RESULT_F  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/eval_def_result.json")


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


# ─────────────── GT episodes ───────────────
def load_gt_defense_frames(pair: str):
    """Читає tracks_and_behavior.txt і повертає відсортовані frame-номери з defensive=1."""
    from config import settings
    gt_root = Path(settings.GT_DATASET_PATH)
    tracks_path = gt_root / pair / "tracks_and_behavior.txt"
    cols = ["frame","track_id","cx","cy","w","h","arrival","defensive","fanning","washboarding"]
    df = pd.read_csv(tracks_path, header=None, names=cols)
    defense_frames = df[df["defensive"] == 1]["frame"].astype(int).unique()
    defense_frames.sort()
    return defense_frames.tolist()


# ─────────────── Pred episodes з JSON ───────────────
def load_pred_defense_frames(result_path: Path):
    doc = json.loads(result_path.read_text())
    ppf = doc.get("pred_per_frame", {})
    frames = []
    for f_str, pdata in ppf.items():
        if any(b.get("behavior") == "defense" for b in pdata.values()):
            frames.append(int(f_str))
    return sorted(frames), doc


# ─────────────── Matching (як у _compute_event_metrics_for_class) ───────────────
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


# ─────────────── Main ───────────────
def main():
    print("=" * 70)
    print("  DEFENSE EVENT-LEVEL — повний звіт (GT + Pred episodes)")
    print(f"  Пара: {PAIR}  |  GAP={GAP} frames  |  JSON: {RESULT_F}")
    print("=" * 70)

    gt_frames = load_gt_defense_frames(PAIR)
    pred_frames, doc = load_pred_defense_frames(RESULT_F)

    gt_eps   = get_episodes(gt_frames,   GAP)
    pred_eps = get_episodes(pred_frames, GAP)

    tp, fp, fn, matched_gt_idx = match_events(gt_eps, pred_eps, GAP)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n  GT-подій:   {len(gt_eps)}")
    print(f"  Pred-подій: {len(pred_eps)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision: {prec*100:.1f}%  Recall: {rec*100:.1f}%  F1: {f1*100:.1f}%")

    # Звіряємо з doc
    de = doc.get("defense_events", {})
    if de:
        print(f"\n  [Перевірка з eval_cli JSON]")
        print(f"  JSON:  TP={de['tp']} FP={de['fp']} FN={de['fn']} "
              f"P={de['precision']*100:.1f}% R={de['recall']*100:.1f}% F1={de['f1']*100:.1f}%")
        match = (tp == de['tp'] and fp == de['fp'] and fn == de['fn'])
        print(f"  Збіг: {'✓ ПОВНИЙ ЗБІГ' if match else '✗ РОЗБІЖНІСТЬ — різні pred_per_frame?'}")

    print(f"\n  GT episodes (defensive=1, gap={GAP}fr):")
    print(f"  {'#':<5} {'start':>8} {'end':>8} {'len':>8}  {'matched?'}")
    print("  " + "-" * 45)
    for i, (s, e) in enumerate(gt_eps):
        flag = "✓ TP" if i in matched_gt_idx else "✗ FN"
        print(f"  GT[{i:02d}] {s:>8} {e:>8} {e-s+1:>8}  {flag}")

    print(f"\n  Pred episodes (behavior=defense, gap={GAP}fr):")
    print(f"  {'#':<5} {'start':>8} {'end':>8} {'len':>8}  {'status'}")
    print("  " + "-" * 45)
    # перевіряємо кожен pred
    for i, (p_start, p_end) in enumerate(pred_eps):
        matched_any = False
        for g_start, g_end in gt_eps:
            if p_end >= g_start - GAP and p_start <= g_end + GAP:
                matched_any = True
                break
        status = "✓ (→TP)" if matched_any else "✗ FP"
        print(f"  Pr[{i:02d}] {p_start:>8} {p_end:>8} {p_end-p_start+1:>8}  {status}")

    print(f"\n  GT frames з defense: {len(gt_frames)}")
    print(f"  Pred frames з defense: {len(pred_frames)}")
    if pred_frames:
        print(f"  Перші 10 pred-frames: {pred_frames[:10]}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

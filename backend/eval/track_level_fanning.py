"""Track-level recall для fanning-класифікатора BuzzTrack.

Мотивація:
  Per-frame метрика несправедлива для темпорального класифікатора (30+ кадрів warm-up),
  бо систематично штрафує за FN на початку кожного GT-треку. Track-level recall
  вимірює: «чи виявив BuzzTrack fanning у цій бджолі протягом її GT-треку?»

Два режими матчингу:
  single-track:  матчить GT-трек тільки з одним найкращим pred-треком.
                 Виявляє track fragmentation — занижені числа якщо tracker
                 часто губить і перезапускає треки.
  multi-track:   агрегує fanning-coverage по ВСІХ pred-треках що
                 перетинають GT-трек (IoU ≥ threshold). Правильніший для
                 BuzzTrack де stitching не завжди об'єднує всі фрагменти.

Діагностика (20230711a-fan, GT tid=1, 5994к, fan=100%):
  Найкращий pred-трек: tid=3622, 760 кадрів, single-track overlap=0.097 (FN)
  Multi-track overlap (15 pred-треків разом): 5908/5994 = 0.986 (TP!)
  → BuzzTrack ПРАВИЛЬНО детектує fanning, але tracker fragmentation занижує single-track

Запуск (з теки backend/):
    uv run python -m eval.track_level_fanning
    uv run python -m eval.track_level_fanning --min-len 30 --overlap 0.5
    uv run python -m eval.track_level_fanning --json /tmp/track_result.json
"""
import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("supervision").setLevel(logging.ERROR)

DEFAULT_PAIRS = ["20230711a-fan", "20230711b-fan", "20230609b-def"]

# ─── Per-frame числа з попереднього eval_fast (для порівняльної таблиці) ───────
PERFRAME_BASELINE = {
    "20230711a-fan": {"fan_p": 0.605, "fan_r": 0.559, "fan_f1": 0.581},
    "20230711b-fan": {"fan_p": 0.386, "fan_r": 0.203, "fan_f1": 0.266},
    "20230609b-def": {"fan_p": 0.042, "fan_r": 0.430, "fan_f1": 0.076},
}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _median_iou_between_tracks(
    gt_frames_set: set[int],
    gt_bboxes_by_frame: dict[int, list[float]],
    pred_frames_set: set[int],
    pred_bboxes_by_frame: dict[int, list[float]],
) -> float:
    """Медіанний IoU між GT-треком та pred-треком по їх спільних кадрах."""
    common = gt_frames_set & pred_frames_set
    if not common:
        return 0.0
    ious = []
    for f in common:
        g = gt_bboxes_by_frame.get(f)
        p = pred_bboxes_by_frame.get(f)
        if g is None or p is None:
            continue
        # g, p — [x1,y1,x2,y2]
        xi = max(g[0], p[0]); yi = max(g[1], p[1])
        xa = min(g[2], p[2]); ya = min(g[3], p[3])
        inter = max(0.0, xa - xi) * max(0.0, ya - yi)
        ag = (g[2] - g[0]) * (g[3] - g[1])
        ap = (p[2] - p[0]) * (p[3] - p[1])
        union = ag + ap - inter
        ious.append(inter / union if union > 0 else 0.0)
    return float(np.median(ious)) if ious else 0.0


def build_gt_tracks(gt_df, width: int, height: int) -> dict:
    """
    Повертає словник {gt_track_id -> track_info} де:
    track_info = {
        'frames': sorted list of frame nums,
        'fan_frames': set of frames with fanning=1,
        'bboxes': {frame: [x1,y1,x2,y2]},
        'fanning_ratio': float,
        'length': int,
    }
    Координати в пікселях (вже денормалізовані, gt_df вже містить x1,y1,x2,y2).
    """
    tracks = {}
    for tid, grp in gt_df.groupby("track_id"):
        grp_sorted = grp.sort_values("frame")
        frames = grp_sorted["frame"].tolist()
        fan_frames = set(grp_sorted[grp_sorted["fanning"] == 1]["frame"].tolist())
        bboxes = {
            int(row.frame): [float(row.x1), float(row.y1), float(row.x2), float(row.y2)]
            for _, row in grp_sorted.iterrows()
        }
        length = len(frames)
        fan_ratio = len(fan_frames) / length if length > 0 else 0.0
        tracks[int(tid)] = {
            "frames": frames,
            "frames_set": set(frames),
            "fan_frames": fan_frames,
            "bboxes": bboxes,
            "fanning_ratio": fan_ratio,
            "length": length,
        }
    return tracks


def build_pred_tracks(pred_per_frame: dict) -> dict:
    """
    З per_frame_behaviors {frame_num: {track_id: {bbox, behavior}}} будує
    pred_tracks {track_id -> track_info} де:
    track_info = {
        'frames': sorted list,
        'fan_frames': set of frames where behavior=='fanning',
        'bboxes': {frame: [x1,y1,x2,y2]},
        'fanning_ratio': float,
        'length': int,
    }
    """
    # Зібрати всі кадри та bbox по track_id
    by_tid: dict[int, dict] = defaultdict(lambda: {
        "frames": [], "fan_frames": set(), "bboxes": {}
    })
    for f_num, bees in pred_per_frame.items():
        for tid, info in bees.items():
            tid = int(tid)
            f = int(f_num)
            by_tid[tid]["frames"].append(f)
            by_tid[tid]["bboxes"][f] = info["bbox"]
            if info.get("behavior") == "fanning":
                by_tid[tid]["fan_frames"].add(f)

    tracks = {}
    for tid, data in by_tid.items():
        frames = sorted(data["frames"])
        length = len(frames)
        fan_ratio = len(data["fan_frames"]) / length if length > 0 else 0.0
        tracks[tid] = {
            "frames": frames,
            "frames_set": set(frames),
            "fan_frames": data["fan_frames"],
            "bboxes": data["bboxes"],
            "fanning_ratio": fan_ratio,
            "length": length,
        }
    return tracks


def match_gt_to_pred(
    gt_track: dict,
    pred_tracks: dict,
    iou_threshold: float = 0.3,
) -> tuple[int | None, float]:
    """
    Знаходить найкращий pred-трек для GT-треку через медіанний IoU.
    Повертає (best_pred_tid, best_iou) або (None, 0.0).
    """
    best_tid = None
    best_iou = iou_threshold  # мінімальний поріг

    gt_frames_set = gt_track["frames_set"]
    gt_bboxes = gt_track["bboxes"]

    for pred_tid, pred_track in pred_tracks.items():
        if not (gt_frames_set & pred_track["frames_set"]):
            continue
        med_iou = _median_iou_between_tracks(
            gt_frames_set, gt_bboxes,
            pred_track["frames_set"], pred_track["bboxes"],
        )
        if med_iou > best_iou:
            best_iou = med_iou
            best_tid = pred_tid

    return best_tid, best_iou


def compute_multitrack_overlap(
    gt_track: dict,
    pred_tracks: dict,
    iou_threshold: float = 0.3,
) -> tuple[float, list[int], dict]:
    """
    Агрегований multi-track temporal overlap.

    Для кожного GT-треку знаходить ВСІ pred-треки з медіанним IoU ≥ iou_threshold,
    об'єднує їх fan_frames ∩ gt_frames і ділить на |gt_frames|.

    Це правильна метрика для систем де tracker fragmentation є нормою —
    BuzzTrack може розбити одну бджолу на 10–20 pred-треків, але сумарно
    правильно класифікує майже всі її кадри.

    Повертає (overlap, [matched_pred_tids], {pred_tid: {iou, frames, fan_ov}})
    """
    gt_frames_set = gt_track["frames_set"]
    gt_bboxes = gt_track["bboxes"]
    n_gt = len(gt_frames_set)
    if n_gt == 0:
        return 0.0, [], {}

    matched_pred: dict[int, dict] = {}
    for pred_tid, pred_track in pred_tracks.items():
        if not (gt_frames_set & pred_track["frames_set"]):
            continue
        med_iou = _median_iou_between_tracks(
            gt_frames_set, gt_bboxes,
            pred_track["frames_set"], pred_track["bboxes"],
        )
        if med_iou >= iou_threshold:
            fan_frames_in_gt = gt_frames_set & pred_track["fan_frames"]
            matched_pred[pred_tid] = {
                "iou": round(med_iou, 3),
                "common_frames": len(gt_frames_set & pred_track["frames_set"]),
                "fan_frames_in_gt": len(fan_frames_in_gt),
                "partial_overlap": round(len(fan_frames_in_gt) / n_gt, 4),
                "pred_fan_ratio": round(pred_track["fanning_ratio"], 3),
                "pred_len": pred_track["length"],
            }

    # Union всіх fan frames по всіх matched pred-треках
    union_fan: set[int] = set()
    for pred_tid, pred_track in pred_tracks.items():
        if pred_tid in matched_pred:
            union_fan |= (gt_frames_set & pred_track["fan_frames"])

    overlap = len(union_fan) / n_gt
    return round(overlap, 4), list(matched_pred.keys()), matched_pred


def compute_temporal_overlap(
    gt_track: dict,
    pred_track: dict,
) -> float:
    """
    Частка кадрів GT-треку де pred-трек класифікує fanning.
    = |gt_frames ∩ pred_fan_frames| / |gt_frames|
    """
    n_gt = len(gt_track["frames"])
    if n_gt == 0:
        return 0.0
    overlap_frames = gt_track["frames_set"] & pred_track["fan_frames"]
    return len(overlap_frames) / n_gt


def run_track_eval(
    pair: str,
    config: dict,
    min_track_len: int = 30,
    fan_ratio_thr: float = 0.5,
    iou_threshold: float = 0.3,
    overlap_threshold: float = 0.5,
    multi_track: bool = True,
) -> dict:
    """Головна функція — повертає повний track-level eval doc для одного відео."""
    import cv2
    from services.evaluation.gt_loader import gt_paths, load_gt_behaviors, denormalize, load_entrance_zone
    from services.pipeline import VideoPipeline
    from eval import detection_cache as dc

    paths = gt_paths(pair)
    video_path = str(paths["video"])
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ── GT ──
    cached = dc.load_cache(pair)
    zone = load_entrance_zone(paths["entrance_zone"])
    gt_df = denormalize(load_gt_behaviors(paths["tracks"]), w, h)

    # ── Pipeline (з кешу) ──
    pipeline = VideoPipeline(None, config, {}, gt_entrance_zone=zone)
    cap = cv2.VideoCapture(video_path)
    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret or fn >= total:
            break
        det = cached[fn] if fn < len(cached) else None
        fn += 1
        pipeline.process_frame(frame, fn, fps, detection_result=det)
    cap.release()
    pred_per_frame = pipeline.pipeline_state.get("per_frame_behaviors", {})

    # ── Побудова треків ──
    gt_tracks = build_gt_tracks(gt_df, w, h)
    pred_tracks = build_pred_tracks(pred_per_frame)

    # ── Фільтр GT fanning-треків ──
    gt_fan_tracks = {
        tid: t for tid, t in gt_tracks.items()
        if t["fanning_ratio"] >= fan_ratio_thr and t["length"] >= min_track_len
    }

    # ── Фільтр Pred fanning-треків ──
    pred_fan_tracks = {
        tid: t for tid, t in pred_tracks.items()
        if t["fanning_ratio"] >= fan_ratio_thr and t["length"] >= min_track_len
    }

    # ─────────────────────────────────────────────────────────────────────────
    # RECALL: GT fanning-треки → overlap check
    # ─────────────────────────────────────────────────────────────────────────
    gt_results = []
    matched_pred_ids: set[int] = set()

    for gt_tid, gt_track in gt_fan_tracks.items():
        if multi_track:
            overlap, matched_tids, matched_details = compute_multitrack_overlap(
                gt_track, pred_tracks, iou_threshold
            )
            # Find best tid for display purpose only
            best_tid = None
            if matched_details:
                best_tid = max(matched_details, key=lambda k: matched_details[k]["iou"])
            
            is_tp = overlap >= overlap_threshold
            fn_reason = None if is_tp else ("no_detection" if not matched_tids else "low_overlap")
            if is_tp:
                matched_pred_ids.update(matched_tids)
            gt_results.append({
                "gt_tid": gt_tid,
                "gt_len": gt_track["length"],
                "gt_fan_ratio": round(gt_track["fanning_ratio"], 3),
                "pred_tid": best_tid,
                "n_matched_pred": len(matched_tids),
                "match_iou": matched_details[best_tid]["iou"] if best_tid else 0.0,
                "overlap": overlap,
                "tp": is_tp,
                "fn_reason": fn_reason,
            })
        else:
            # Single-track: тільки найкращий pred-трек
            pred_tid, iou = match_gt_to_pred(gt_track, pred_tracks, iou_threshold)
            if pred_tid is None:
                gt_results.append({
                    "gt_tid": gt_tid,
                    "gt_len": gt_track["length"],
                    "gt_fan_ratio": round(gt_track["fanning_ratio"], 3),
                    "pred_tid": None,
                    "n_matched_pred": 0,
                    "match_iou": 0.0,
                    "overlap": 0.0,
                    "tp": False,
                    "fn_reason": "no_detection",
                })
            else:
                pred_track = pred_tracks[pred_tid]
                overlap = compute_temporal_overlap(gt_track, pred_track)
                is_tp = overlap >= overlap_threshold
                if is_tp:
                    matched_pred_ids.add(pred_tid)
                gt_results.append({
                    "gt_tid": gt_tid,
                    "gt_len": gt_track["length"],
                    "gt_fan_ratio": round(gt_track["fanning_ratio"], 3),
                    "pred_tid": pred_tid,
                    "n_matched_pred": 1,
                    "match_iou": round(iou, 3),
                    "overlap": round(overlap, 3),
                    "tp": is_tp,
                    "fn_reason": None if is_tp else "low_overlap",
                })

    tp_count = sum(1 for r in gt_results if r["tp"])
    fn_count = sum(1 for r in gt_results if not r["tp"])

    # ─────────────────────────────────────────────────────────────────────────
    # PRECISION: pred fanning-треки без GT-матчу = FP
    # ─────────────────────────────────────────────────────────────────────────
    fp_pred_tids = []
    for pred_tid, pred_track in pred_fan_tracks.items():
        if pred_tid in matched_pred_ids:
            continue
        # Перевірити чи є будь-який GT fanning-трек що матчить цей pred
        found = False
        for gt_tid, gt_track in gt_fan_tracks.items():
            med_iou = _median_iou_between_tracks(
                gt_track["frames_set"], gt_track["bboxes"],
                pred_track["frames_set"], pred_track["bboxes"],
            )
            if med_iou >= iou_threshold:
                if multi_track:
                    found = True
                    break
                else:
                    overlap = compute_temporal_overlap(gt_track, pred_track)
                    if overlap >= overlap_threshold:
                        found = True
                        break
        if not found:
            fp_pred_tids.append(pred_tid)

    fp_count = len(fp_pred_tids)
    precision, recall, f1 = _prf(tp_count, fp_count, fn_count)

    # ── Розподіл overlap значень ──
    overlaps = [r["overlap"] for r in gt_results if r["pred_tid"] is not None]
    overlap_hist = {
        "0.0–0.1": sum(1 for o in overlaps if o < 0.10),
        "0.1–0.2": sum(1 for o in overlaps if 0.10 <= o < 0.20),
        "0.2–0.3": sum(1 for o in overlaps if 0.20 <= o < 0.30),
        "0.3–0.5": sum(1 for o in overlaps if 0.30 <= o < 0.50),
        "0.5–0.7": sum(1 for o in overlaps if 0.50 <= o < 0.70),
        "0.7–0.9": sum(1 for o in overlaps if 0.70 <= o < 0.90),
        "0.9–1.0": sum(1 for o in overlaps if o >= 0.90),
    }
    quantiles = {}
    if overlaps:
        for q in [25, 50, 75, 90]:
            quantiles[f"p{q}"] = round(float(np.percentile(overlaps, q)), 3)

    return {
        "pair": pair,
        "total_video_frames": total,
        "multi_track_mode": multi_track,
        # GT stats
        "gt_all_tracks": len(gt_tracks),
        "gt_fan_tracks": len(gt_fan_tracks),
        # Pred stats
        "pred_all_tracks": len(pred_tracks),
        "pred_fan_tracks": len(pred_fan_tracks),
        # Core numbers
        "tp": tp_count,
        "fn": fn_count,
        "fp": fp_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        # Details
        "gt_results": gt_results,
        "fp_pred_tids": fp_pred_tids,
        "overlap_histogram": overlap_hist,
        "overlap_quantiles": quantiles,
        "overlap_mean": round(float(np.mean(overlaps)), 3) if overlaps else 0.0,
        "no_detection_count": sum(1 for r in gt_results if r["fn_reason"] == "no_detection"),
        "low_overlap_count": sum(1 for r in gt_results if r["fn_reason"] == "low_overlap"),
    }


def print_results(doc: dict, baseline: dict | None = None):
    pair = doc["pair"]
    mode = "multi-track" if doc.get("multi_track_mode", True) else "single-track"
    print(f"\n{'─'*68}")
    print(f"  ▶  {pair}  [{mode}]")
    print(f"{'─'*68}")
    print(f"  GT-треки (всього):          {doc['gt_all_tracks']:>5}")
    print(f"  GT fanning-треки (≥50%, ≥30к): {doc['gt_fan_tracks']:>5}")
    print(f"  Pred-треки (всього):        {doc['pred_all_tracks']:>5}")
    print(f"  Pred fanning-треки:         {doc['pred_fan_tracks']:>5}")

    print()
    print(f"  {'':25} {'TP':>5} {'FP':>5} {'FN':>5}   {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'─'*68}")

    p, r, f = doc["precision"], doc["recall"], doc["f1"]
    label = f"Track-level ({mode})"
    print(f"  {label:25} {doc['tp']:>5} {doc['fp']:>5} {doc['fn']:>5}"
          f"   {p*100:6.1f}% {r*100:6.1f}% {f*100:6.1f}%")

    if baseline:
        bp = baseline["fan_p"]; br = baseline["fan_r"]; bf = baseline["fan_f1"]
        print(f"  {'Per-frame (eval_fast)':25} {'—':>5} {'—':>5} {'—':>5}"
              f"   {bp*100:6.1f}% {br*100:6.1f}% {bf*100:6.1f}%")

    # FN breakdown
    nd = doc["no_detection_count"]
    lo = doc["low_overlap_count"]
    if nd + lo > 0:
        print(f"\n  FN breakdown: no-detection={nd}, low-overlap={lo}")

    # Overlap histogram
    if doc["overlap_histogram"]:
        print(f"\n  Розподіл overlap (по GT-треках з pred-матчем, n={sum(doc['overlap_histogram'].values())}):")
        for bucket, cnt in doc["overlap_histogram"].items():
            bar = "█" * cnt + "·" * max(0, 20 - cnt)
            star = " ← TP threshold" if bucket in ("0.5–0.7", "0.7–0.9", "0.9–1.0") else ""
            print(f"    {bucket}: {cnt:>3}  {bar}{star}")

    # Quantiles
    if doc["overlap_quantiles"]:
        q = doc["overlap_quantiles"]
        print(f"  Квантилі overlap: "
              f"p25={q.get('p25', '—')}  p50={q.get('p50', '—')}  "
              f"p75={q.get('p75', '—')}  p90={q.get('p90', '—')}  "
              f"mean={doc['overlap_mean']}")

    # Per-track деталі (перші 20 GT треків для зручності)
    print(f"\n  Per-GT-track деталі (перші 20 з {len(doc['gt_results'])}):")
    print(f"  {'gt_tid':>8} {'len':>5} {'fan%':>6} {'pred_tid':>9} {'IoU':>6} "
          f"{'overlap':>9} {'result':>8}")
    print(f"  {'─'*65}")
    for r_item in sorted(doc["gt_results"], key=lambda x: -x["gt_len"])[:20]:
        tp_str = "✓ TP" if r_item["tp"] else f"✗ FN ({r_item['fn_reason']})"
        pred_str = str(r_item["pred_tid"]) if r_item["pred_tid"] is not None else "—"
        iou_str = f"{r_item['match_iou']:.3f}" if r_item["pred_tid"] else "—"
        ov_str = f"{r_item['overlap']:.3f}" if r_item["pred_tid"] else "—"
        print(f"  {r_item['gt_tid']:>8} {r_item['gt_len']:>5} "
              f"{r_item['gt_fan_ratio']*100:>5.0f}%  {pred_str:>9} {iou_str:>6} "
              f"{ov_str:>9}  {tp_str}")


def print_summary_table(all_docs: list[dict], all_docs_single: list[dict] | None = None):
    """Зведена markdown-таблиця для статті."""
    print(f"\n{'='*72}")
    print(f"  ЗВЕДЕНА ТАБЛИЦЯ: TRACK-LEVEL (multi) vs SINGLE-TRACK vs PER-FRAME")
    print(f"{'='*72}")
    print()
    print("| Відео | Метрика | GT треки | TP | FP | FN | P | R | F1 |")
    print("|-------|---------|:-------:|---:|---:|---:|---:|---:|---:|")

    mean_mt_f1, mean_st_f1, mean_pf_f1 = [], [], []
    single_by_pair = {d["pair"]: d for d in (all_docs_single or [])}

    for doc in all_docs:
        pair = doc["pair"]
        p, r, f = doc["precision"], doc["recall"], doc["f1"]
        mean_mt_f1.append(f)
        print(f"| {pair} | **Track-level (multi)** | {doc['gt_fan_tracks']} | "
              f"{doc['tp']} | {doc['fp']} | {doc['fn']} | "
              f"{p*100:.1f}% | {r*100:.1f}% | **{f*100:.1f}%** |")
        # Single-track для порівняння
        sd = single_by_pair.get(pair)
        if sd:
            sp, sr, sf = sd["precision"], sd["recall"], sd["f1"]
            mean_st_f1.append(sf)
            print(f"| {pair} | Track-level (single) | {sd['gt_fan_tracks']} | "
                  f"{sd['tp']} | {sd['fp']} | {sd['fn']} | "
                  f"{sp*100:.1f}% | {sr*100:.1f}% | {sf*100:.1f}% |")
        # Per-frame baseline
        bl = PERFRAME_BASELINE.get(pair, {})
        if bl:
            bp, br, bf = bl["fan_p"], bl["fan_r"], bl["fan_f1"]
            mean_pf_f1.append(bf)
            print(f"| {pair} | Per-frame (eval_fast) | — | — | — | — | "
                  f"{bp*100:.1f}% | {br*100:.1f}% | {bf*100:.1f}% |")

    if mean_mt_f1:
        avg_mt = sum(mean_mt_f1) / len(mean_mt_f1)
        avg_st = sum(mean_st_f1) / len(mean_st_f1) if mean_st_f1 else 0.0
        avg_pf = sum(mean_pf_f1) / len(mean_pf_f1) if mean_pf_f1 else 0.0
        print(f"| **mean** | **Track-level (multi)** | | | | | | | **{avg_mt*100:.1f}%** |")
        if mean_st_f1:
            print(f"| **mean** | Track-level (single) | | | | | | | {avg_st*100:.1f}% |")
        print(f"| **mean** | Per-frame (eval_fast) | | | | | | | {avg_pf*100:.1f}% |")

    print()
    print(f"*Multi-track: агрегований overlap по ВСІХ IoU-matched pred-треках*")
    print(f"*Single-track: overlap тільки з найкращим одним pred-треком*")
    print(f"*Різниця multi vs single показує вплив tracker fragmentation*")


def main():
    ap = argparse.ArgumentParser(
        description="Track-level fanning recall evaluation for BuzzTrack"
    )
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--min-len", type=int, default=30,
                    help="Мінімальна тривалість треку (кадрів, default=30)")
    ap.add_argument("--fan-ratio", type=float, default=0.5,
                    help="Мінімальна частка fanning кадрів у GT-треку (default=0.5)")
    ap.add_argument("--iou", type=float, default=0.3,
                    help="Поріг IoU для GT↔Pred матчингу (default=0.3)")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Поріг temporal overlap для TP (default=0.5)")
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--single", action="store_true",
                    help="Також запустити single-track режим для порівняння (повільніше)")
    ap.add_argument("--json", default="", help="Зберегти JSON-результат")
    args = ap.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    from eval.eval_fast import base_config, load_yaml_config
    DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "eval_config.yaml")
    config = base_config(args.warmup)
    config.update(load_yaml_config(DEFAULT_CONFIG))

    print(f"\n{'='*72}")
    print(f"  TRACK-LEVEL FANNING RECALL  (BuzzTrack temporal classifier eval)")
    print(f"  pairs={pairs}  min_len={args.min_len}  fan_ratio≥{args.fan_ratio}")
    print(f"  iou_thr={args.iou}  overlap_thr={args.overlap}  "
          f"mode={'multi+single' if args.single else 'multi-track'}")
    print(f"{'='*72}")

    all_docs_multi = []
    all_docs_single = []

    for pair in pairs:
        print(f"\n  [{pair}] Запуск pipeline [multi-track] (з кешу)...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            doc = run_track_eval(
                pair, config,
                min_track_len=args.min_len,
                fan_ratio_thr=args.fan_ratio,
                iou_threshold=args.iou,
                overlap_threshold=args.overlap,
                multi_track=True,
            )
            elapsed = time.perf_counter() - t0
            print(f"OK ({elapsed:.1f}s)")
            all_docs_multi.append(doc)
            print_results(doc, PERFRAME_BASELINE.get(pair))
        except Exception as e:
            import traceback
            elapsed = time.perf_counter() - t0
            print(f"ERROR ({elapsed:.1f}s): {e}")
            traceback.print_exc()

        # Опціонально — single-track для порівняння (показує вплив fragmentation)
        if args.single:
            print(f"  [{pair}] Запуск pipeline [single-track]...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                doc_s = run_track_eval(
                    pair, config,
                    min_track_len=args.min_len,
                    fan_ratio_thr=args.fan_ratio,
                    iou_threshold=args.iou,
                    overlap_threshold=args.overlap,
                    multi_track=False,
                )
                elapsed = time.perf_counter() - t0
                print(f"OK ({elapsed:.1f}s)")
                all_docs_single.append(doc_s)
                print_results(doc_s, None)  # без baseline щоб не дублювати
            except Exception as e:
                import traceback
                elapsed = time.perf_counter() - t0
                print(f"ERROR ({elapsed:.1f}s): {e}")
                traceback.print_exc()

    if all_docs_multi:
        print_summary_table(all_docs_multi, all_docs_single if args.single else None)

    if args.json and all_docs_multi:
        out = {
            "config": {
                "pairs": pairs, "min_len": args.min_len,
                "fan_ratio": args.fan_ratio, "iou": args.iou,
                "overlap": args.overlap,
            },
            "multi_track": all_docs_multi,
            "single_track": all_docs_single,
        }
        # gt_results.matched_details може бути великим — обрізаємо
        Path(args.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n  JSON → {args.json}")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()

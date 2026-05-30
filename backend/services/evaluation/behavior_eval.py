"""Per-frame aggregate behavior evaluation + event-based foraging metric.

- fanning/washboarding/defense: per-frame IoU matching з warm-up виключенням перших кадрів треку
- foraging: подієва метрика (GT line-crossing events vs pred events)
- multi-label кадри (≥2 прапорці одночасно) виключаються з per-frame порівняння
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from services.evaluation.counting_eval import compute_gt_events, match_events

logger = logging.getLogger(__name__)

PERFRAME_CLASSES = ["fanning", "washboarding", "defense"]
PAPER_CLASSES = ["foraging", "fanning", "washboarding", "defense", "background"]
PAPER_BEHAV = {"foraging", "fanning", "washboarding", "defense"}


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


def _compute_perframe_metrics(
    gt_df: pd.DataFrame,
    pred_per_frame: dict,
    eval_classes: list[str],
    iou_threshold: float = 0.3,
    warmup_frames: int | dict = 80,
) -> tuple[dict, dict, int, int, int]:
    """Per-frame TP/FP/FN для fanning/washboarding/defense з warm-up і multi-label виключенням.

    warmup_frames: int (глобальний) або dict {class_name: frames} для per-class warmup.
    Defense використовує маленький warmup (~15f), fanning — великий (~80f).

    Повертає (per_class_metrics, confusion_matrix, total_gt_labeled, total_matched, excluded_multilabel).
    """
    # Нормалізуємо warmup до dict
    if isinstance(warmup_frames, int):
        warmup_map = {cls: warmup_frames for cls in eval_classes}
    else:
        warmup_map = {cls: warmup_frames.get(cls, 80) for cls in eval_classes}

    # Перший кадр кожного GT-треку (для warm-up)
    track_first_frame: dict[int, int] = (
        gt_df.groupby("track_id")["frame"].min().to_dict()
    )

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    detected: dict[str, int] = defaultdict(int)  # GT-бджоли зматчені з будь-яким pred-боксом
    cm: dict = defaultdict(lambda: defaultdict(int))

    total_matched = 0
    total_gt_labeled = 0
    excluded_multilabel = 0

    frames = sorted(gt_df["frame"].unique())

    for frame_num in frames:
        gt_frame = gt_df[gt_df["frame"] == frame_num]
        pred_frame = pred_per_frame.get(frame_num, {})

        # Виключити multi-label рядки
        gt_single = gt_frame[gt_frame["gt_label_count"] <= 1]
        excluded_multilabel += len(gt_frame) - len(gt_single)

        # Warm-up per-class: виключити рядки де бджола ще не набрала достатньо кадрів
        def _past_warmup(row):
            w = warmup_map.get(row["gt_behavior"], 80)
            if w <= 0:
                return True
            first = track_first_frame.get(int(row["track_id"]), 0)
            return row["frame"] >= first + w

        if any(v > 0 for v in warmup_map.values()):
            gt_single = gt_single[gt_single.apply(_past_warmup, axis=1)]

        # Фільтр по класах що оцінюємо
        gt_labeled = gt_single[gt_single["gt_behavior"].isin(eval_classes)]
        total_gt_labeled += len(gt_labeled)

        if len(gt_labeled) == 0:
            for tid, pdata in pred_frame.items():
                pred_b = pdata.get("behavior")
                if pred_b in eval_classes:
                    fp[pred_b] += 1
            continue

        if len(pred_frame) == 0:
            for _, row in gt_labeled.iterrows():
                fn[row["gt_behavior"]] += 1
            continue

        gt_boxes = gt_labeled[["x1", "y1", "x2", "y2"]].to_numpy()
        pred_items = list(pred_frame.values())
        pred_boxes = np.array([p["bbox"] for p in pred_items])

        iou = _iou_matrix(gt_boxes, pred_boxes)

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
                detected[gt_b] += 1
                cm[gt_b][pred_b] += 1
                if pred_b == gt_b:
                    tp[gt_b] += 1
                else:
                    fn[gt_b] += 1
                    if pred_b in eval_classes:
                        fp[pred_b] += 1
            else:
                fn[gt_b] += 1

        for pi in range(len(pred_items)):
            if pi not in pred_used:
                pred_b = pred_items[pi].get("behavior") or "unknown"
                if pred_b in eval_classes:
                    fp[pred_b] += 1

    per_class = {}
    for cls in eval_classes:
        t, f_p, f_n = tp[cls], fp[cls], fn[cls]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        gt_count = t + f_n
        # detection_recall: скільки GT-бджіл цього класу взагалі знайдено (будь-який pred-бокс)
        # classification_recall (== recall): з них правильно класифіковано
        det_recall = detected[cls] / gt_count if gt_count > 0 else 0.0
        per_class[cls] = {
            "tp": t, "fp": f_p, "fn": f_n,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "gt_count": gt_count,
            "pred_count": t + f_p,
            "detected": detected[cls],
            "detection_recall": round(det_recall, 4),
            "classification_recall": round(rec, 4),
        }

    all_labels = sorted(set(eval_classes) | {
        pb for gt_map in cm.values() for pb in gt_map
    })
    cm_dict = {
        gt_cls: {pred_cls: cm[gt_cls][pred_cls] for pred_cls in all_labels}
        for gt_cls in eval_classes
    }

    return per_class, cm_dict, total_gt_labeled, total_matched, excluded_multilabel



def _compute_event_metrics_for_class(
    gt_df,
    pred_per_frame: dict,
    behavior_class: str,
    gap_frames: int = 30,
) -> dict:
    import pandas as pd
    gt_frames = gt_df[gt_df["gt_behavior"] == behavior_class]["frame"].unique()
    gt_frames.sort()
    
    pred_frames = []
    for f_num, pdata in pred_per_frame.items():
        if any(b.get("behavior") == behavior_class for b in pdata.values()):
            pred_frames.append(f_num)
    pred_frames.sort()
    
    def get_episodes(frames, max_gap):
        eps = []
        if len(frames) == 0: return eps
        start = frames[0]
        prev = frames[0]
        for f in frames[1:]:
            if f - prev > max_gap:
                eps.append((start, prev))
                start = f
            prev = f
        eps.append((start, prev))
        return eps
        
    gt_eps = get_episodes(gt_frames, gap_frames)
    pred_eps = get_episodes(pred_frames, gap_frames)
    
    tp = 0
    fp = 0
    
    matched_gt = set()
    matched_pred = 0
    for p_start, p_end in pred_eps:
        matched = False
        for i, (g_start, g_end) in enumerate(gt_eps):
            if p_end >= g_start - gap_frames and p_start <= g_end + gap_frames:
                matched_gt.add(i)
                matched = True
                break
        if matched:
            matched_pred += 1
            
    tp = len(matched_gt)
    fp = len(pred_eps) - matched_pred
    fn = len(gt_eps) - len(matched_gt)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "gt_count": len(gt_eps),
        "pred_count": len(pred_eps),
    }


def _compute_foraging_events(
    gt_df: pd.DataFrame,
    entrance_zone: np.ndarray,
    pred_events: list[dict],
    fps: float,
) -> dict:
    """Подієва метрика для foraging: GT line-crossing events vs pred counting events.

    Обидва напрямки (IN+OUT) рахуємо разом — важливий факт реєстрації, не напрям.
    Виводимо GT-події з УСІХ GT-треків, оскільки будь-який перетин = foraging-подія.
    """
    foraging_tracks = gt_df[gt_df["gt_behavior"] == "foraging"]
    gt_events = compute_gt_events(foraging_tracks, entrance_zone, fps=fps)
    
    # Фільтруємо pred_events: залишаємо тільки ті, де бджола мала стан 'foraging'
    pred_foraging = [e for e in pred_events if e.get("behavior_class") == "foraging"]
    
    if len(gt_events) == 0 and len(pred_foraging) == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "gt_count": 0, "pred_count": 0}

    # Зіставляємо без поділу на напрям (нейтралізуємо направленість)
    pred_neutral = [{"frame": e["frame"]} for e in pred_foraging]
    gt_neutral = [{"frame": e["frame"]} for e in gt_events]

    from services.evaluation.counting_eval import _match_directional, _metrics
    tp, fp, fn = _match_directional(gt_neutral, pred_neutral, frame_window=15)
    return _metrics(tp, fp, fn, len(gt_neutral), len(pred_neutral))


def _compute_frame_presence_metrics(
    gt_df: pd.DataFrame,
    pred_per_frame: dict,
    total_frames: int,
) -> dict:
    """Frame-presence метрика — як у статті Sledevič et al. (PLOS ONE 2025, Fig. 14).

    Оцінює на рівні КАДРУ (не на рівні окремої бджоли):
      - GT: які поведінкові класи присутні у кадрі (multi-label, без IoU)
      - Pred: які класи є серед передбачень у кадрі (multi-label)
      - Клас background: кадри без жодної специфічної поведінки
      - Якщо у кадрі є і fanning, і foraging — обидва TP (multi-label)

    Це поблажливіша метрика: вона не штрафує за класифікацію конкретної бджоли —
    достатньо, щоб клас хоч десь з'явився у кадрі.
    """
    from collections import defaultdict

    # GT: набір класів у кожному кадрі
    gt_present: dict[int, set] = defaultdict(set)
    for fr, grp in gt_df.groupby("frame"):
        s = {b for b in grp["gt_behavior"].unique() if b in PAPER_BEHAV}
        gt_present[int(fr)] = s

    # Pred: набір класів у кожному кадрі
    pred_present: dict[int, set] = defaultdict(set)
    for f_num, bees in pred_per_frame.items():
        s = {b["behavior"] for b in bees.values() if b.get("behavior") in PAPER_BEHAV}
        pred_present[int(f_num)] = s

    stat: dict[str, dict] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in PAPER_CLASSES}

    for f_num in range(1, total_frames + 1):
        g = gt_present.get(f_num, set())
        p = pred_present.get(f_num, set())
        # background = порожній набір специфічних поведінок
        g_eff = g if g else {"background"}
        p_eff = p if p else {"background"}
        for c in PAPER_CLASSES:
            in_g = c in g_eff
            in_p = c in p_eff
            if in_g and in_p:
                stat[c]["tp"] += 1
            elif in_p and not in_g:
                stat[c]["fp"] += 1
            elif in_g and not in_p:
                stat[c]["fn"] += 1

    per_class_fp: dict[str, dict] = {}
    for c in PAPER_CLASSES:
        t, fp, fn = stat[c]["tp"], stat[c]["fp"], stat[c]["fn"]
        prec = t / (t + fp) if (t + fp) > 0 else 0.0
        rec = t / (t + fn) if (t + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_fp[c] = {
            "tp": t, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    # Загальна accuracy = (TP всіх класів) / (TP + FP + FN) — як у їхній матриці
    total_tp = sum(v["tp"] for v in per_class_fp.values())
    total_all = sum(v["tp"] + v["fp"] + v["fn"] for v in per_class_fp.values())
    # Macro-F1 по 4 поведінкових класах (без background) — найближче до їхніх 87%
    behav_f1s = [per_class_fp[c]["f1"] for c in PAPER_BEHAV if c in per_class_fp]
    macro_f1_behav = sum(behav_f1s) / len(behav_f1s) if behav_f1s else 0.0
    macro_f1_all = sum(v["f1"] for v in per_class_fp.values()) / len(per_class_fp)

    return {
        "per_class": per_class_fp,
        "macro_f1_behavior": round(macro_f1_behav, 4),
        "macro_f1_all": round(macro_f1_all, 4),
        "total_frames": total_frames,
    }


def build_behavior_evaluation(
    gt_df: pd.DataFrame,
    pred_per_frame: dict,
    pred_events: list[dict],
    entrance_zone: np.ndarray,
    fps: float,
    *,
    warmup_frames: int | dict = 80,
    iou_threshold: float = 0.3,
    total_frames: int = 0,
) -> dict:
    """Головна функція: повертає повний evaluation_doc без Mongo.

    Переюзається і з evaluator.py, і з eval_cli.py.
    """
    # Визначаємо які per-frame класи є у цьому GT-файлі
    present_perframe = [
        cls for cls in PERFRAME_CLASSES
        if cls in gt_df["gt_behavior"].values
    ]

    # foraging підрахунок, якщо є в GT
    has_foraging = "foraging" in gt_df["gt_behavior"].values

    # Per-class warmup: defense не потребує довгого warm-up (детектується миттєво)
    if isinstance(warmup_frames, int):
        warmup_map: dict = {
            "fanning": warmup_frames,
            "washboarding": warmup_frames,
            "defense": min(15, warmup_frames),
        }
    else:
        warmup_map = warmup_frames

    per_class, cm_dict, total_gt_labeled, total_matched, excluded_multilabel = (
        _compute_perframe_metrics(
            gt_df, pred_per_frame, present_perframe,
            iou_threshold=iou_threshold,
            warmup_frames=warmup_map,
        )
    )

    total_tp = sum(m["tp"] for m in per_class.values())
    total_decisions = total_tp + sum(m["fp"] for m in per_class.values()) + sum(m["fn"] for m in per_class.values())
    overall_acc = total_tp / total_decisions if total_decisions > 0 else 0.0


    foraging_events = None
    if has_foraging:
        foraging_events = _compute_foraging_events(gt_df, entrance_zone, pred_events, fps)
        
    defense_events = None
    if "defense" in gt_df["gt_behavior"].values:
        defense_events = _compute_event_metrics_for_class(gt_df, pred_per_frame, "defense", gap_frames=30)

    # Frame-presence метрика (як у Sledevič et al. PLOS ONE 2025)
    n_frames = total_frames if total_frames > 0 else int(gt_df["frame"].max())
    frame_presence = _compute_frame_presence_metrics(gt_df, pred_per_frame, n_frames)

    return {
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "overall_accuracy": round(overall_acc, 4),
        "total_gt_labeled": total_gt_labeled,
        "total_matched": total_matched,
        "eval_classes": present_perframe,
        "foraging_events": foraging_events,
        "defense_events": defense_events,
        "excluded_multilabel": excluded_multilabel,
        "warmup_frames": warmup_frames,
        "frame_presence": frame_presence,
    }


# Залишаємо для зворотної сумісності (evaluator старий міг імпортувати)
def compute_behavior_metrics(
    gt_df: pd.DataFrame,
    pred_per_frame: dict,
    eval_classes: list[str],
    iou_threshold: float = 0.3,
) -> dict:
    per_class, cm_dict, total_gt_labeled, total_matched, _ = _compute_perframe_metrics(
        gt_df, pred_per_frame, eval_classes, iou_threshold=iou_threshold, warmup_frames=0
    )
    total_tp = sum(m["tp"] for m in per_class.values())
    total_decisions = total_tp + sum(m["fp"] for m in per_class.values()) + sum(m["fn"] for m in per_class.values())
    overall_acc = total_tp / total_decisions if total_decisions > 0 else 0.0
    return {
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "overall_accuracy": round(overall_acc, 4),
        "total_gt_labeled": total_gt_labeled,
        "total_matched": total_matched,
        "eval_classes": eval_classes,
    }

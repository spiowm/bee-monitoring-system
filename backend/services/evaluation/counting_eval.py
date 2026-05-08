"""Обчислення GT-подій вліт/виліт і зіставлення з передбаченими подіями."""
import numpy as np
import pandas as pd


def get_line_y_for_cx(entrance_zone: np.ndarray, cx: float, shift_down: int = 8) -> float:
    """
    Інтерполяція Y по верхньому краю entrance_zone (перші дві точки полігону)
    для заданої X-координати бджоли. Збігається з логікою TrafficCounter.get_line_y().
    
    entrance_zone: (4, 2) array — [top-left, top-right, bottom-right, bottom-left].
    shift_down: зміщення в пікселях вниз (як у counter.py).
    """
    x1, y1 = float(entrance_zone[0][0]), float(entrance_zone[0][1])
    x2, y2 = float(entrance_zone[1][0]), float(entrance_zone[1][1])
    if abs(x2 - x1) > 1e-3:
        return y1 + (y2 - y1) / (x2 - x1) * (cx - x1) + shift_down
    return y1 + shift_down


def get_line_y(entrance_zone: np.ndarray, line_position: float) -> float:
    """Y-координата лінії підрахунку (для GT-відео анотації — горизонтальна лінія)."""
    x1, y1 = float(entrance_zone[0][0]), float(entrance_zone[0][1])
    x2, y2 = float(entrance_zone[1][0]), float(entrance_zone[1][1])
    return (y1 + y2) / 2 + 8  # середина верхнього краю + shift


def compute_gt_events(
    gt_df: pd.DataFrame,
    entrance_zone: np.ndarray,
    fps: float,
    line_position: float = 0.5,
) -> list[dict]:
    """
    Реплеїть GT-треки кадр за кадром і фіксує перетини лінії.

    Лінія підрахунку — верхній край entrance_zone (як у detection pipeline),
    інтерпольована по X-координаті кожної бджоли.

    Очікує df після denormalize() (з cx_px, cy_px у пікселях).
    """
    events: list[dict] = []

    for track_id, track_df in gt_df.groupby("track_id", sort=False):
        track_df = track_df.sort_values("frame")
        ys = track_df["cy_px"].to_numpy()
        xs = track_df["cx_px"].to_numpy()
        frames = track_df["frame"].to_numpy()
        if len(ys) < 2:
            continue

        last_event_frame = -9999

        for i in range(1, len(ys)):
            prev_y, curr_y = ys[i - 1], ys[i]
            curr_x = xs[i]
            line_y = get_line_y_for_cx(entrance_zone, curr_x)

            crossed_down = prev_y < line_y <= curr_y
            crossed_up = prev_y > line_y >= curr_y
            if not (crossed_down or crossed_up):
                continue

            direction = "OUT" if crossed_down else "IN"
            frame = int(frames[i])
            
            # Додаємо такий самий debounce (45 кадрів), як у counter.py,
            # щоб не рахувати джиттер (коли бджола сидить на лінії і коливається на 1 піксель)
            if frame - last_event_frame >= 45:
                events.append({
                    "frame": frame,
                    "timestamp_sec": frame / fps,
                    "track_id": int(track_id),
                    "direction": direction,
                })
                last_event_frame = frame

    events.sort(key=lambda e: e["frame"])
    return events


def _match_directional(
    gt: list[dict], pred: list[dict], frame_window: int
) -> tuple[int, int, int]:
    """Жадібне зіставлення подій одного напрямку. Повертає (TP, FP, FN)."""
    gt_used = [False] * len(gt)
    tp = 0
    fp = 0

    for p in pred:
        best_idx = -1
        best_dist = frame_window + 1
        for i, g in enumerate(gt):
            if gt_used[i]:
                continue
            d = abs(p["frame"] - g["frame"])
            if d <= frame_window and d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            gt_used[best_idx] = True
            tp += 1
        else:
            fp += 1

    fn = gt_used.count(False)
    return tp, fp, fn


def _metrics(tp: int, fp: int, fn: int, gt_count: int, pred_count: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mae": abs(pred_count - gt_count),
        "gt_count": gt_count,
        "pred_count": pred_count,
    }


def match_events(
    gt_events: list[dict],
    pred_events: list[dict],
    frame_window: int = 15,
) -> dict:
    """
    Окремо для IN та OUT: TP/FP/FN/precision/recall/F1/MAE.
    Підсумкова accuracy = (TP_in + TP_out) / total_decisions.
    """
    gt_in = [e for e in gt_events if e["direction"] == "IN"]
    gt_out = [e for e in gt_events if e["direction"] == "OUT"]
    pred_in = [e for e in pred_events if e["direction"] == "IN"]
    pred_out = [e for e in pred_events if e["direction"] == "OUT"]

    tp_in, fp_in, fn_in = _match_directional(gt_in, pred_in, frame_window)
    tp_out, fp_out, fn_out = _match_directional(gt_out, pred_out, frame_window)

    in_metrics = _metrics(tp_in, fp_in, fn_in, len(gt_in), len(pred_in))
    out_metrics = _metrics(tp_out, fp_out, fn_out, len(gt_out), len(pred_out))

    total_tp = tp_in + tp_out
    total_decisions = total_tp + fp_in + fp_out + fn_in + fn_out
    accuracy = total_tp / total_decisions if total_decisions > 0 else 0.0

    return {
        "in": in_metrics,
        "out": out_metrics,
        "accuracy": round(accuracy, 4),
        "gt_total_in": len(gt_in),
        "gt_total_out": len(gt_out),
        "pred_total_in": len(pred_in),
        "pred_total_out": len(pred_out),
        "frame_window": frame_window,
    }

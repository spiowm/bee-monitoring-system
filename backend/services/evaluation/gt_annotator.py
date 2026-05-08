"""Рендер GT-анотованого відео (жовті bbox + полігон + лінія підрахунку)."""
import cv2
import logging
import numpy as np
import pandas as pd

from services.evaluation.counting_eval import get_line_y_for_cx

logger = logging.getLogger(__name__)

GT_BOX_COLOR = (0, 255, 255)       # жовтий BGR
GT_POLYGON_COLOR = (0, 100, 255)   # помаранчево-червоний
GT_LINE_COLOR = (0, 255, 0)        # зелений
TEXT_COLOR = (255, 255, 255)


def render_gt_video(
    video_path: str,
    gt_df: pd.DataFrame,
    entrance_zone: np.ndarray,
    line_position: float,
    output_path: str,
    progress_cb=None,
    gt_events: list = None,
) -> dict:
    """
    Малює GT-bbox на оригінальному відео. gt_df має бути після denormalize().
    Лінія підрахунку — верхній край entrance_zone (як у detection pipeline).
    Повертає {fps, width, height, total_frames}.
    """
    if gt_events is None:
        gt_events = []
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не можу відкрити відео {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не можу створити writer для {output_path}")

    by_frame: dict[int, pd.DataFrame] = dict(tuple(gt_df.groupby("frame")))

    # Лінія підрахунку — інтерпольована по верхньому краю полігону.
    line_x1 = int(entrance_zone[0][0])
    line_y1 = int(get_line_y_for_cx(entrance_zone, entrance_zone[0][0]))
    line_x2 = int(entrance_zone[1][0])
    line_y2 = int(get_line_y_for_cx(entrance_zone, entrance_zone[1][0]))

    polygon = entrance_zone.astype(np.int32)

    frame_idx = 0
    total_in = 0
    total_out = 0
    
    # Згрупуємо події по кадрах для швидкого доступу
    events_by_frame = {}
    for ev in gt_events:
        f = ev["frame"]
        events_by_frame.setdefault(f, []).append(ev)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Оновлюємо лічильники
            for ev in events_by_frame.get(frame_idx, []):
                if ev["direction"] == "IN": total_in += 1
                if ev["direction"] == "OUT": total_out += 1

            cv2.polylines(frame, [polygon], isClosed=True, color=GT_POLYGON_COLOR, thickness=2)
            cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), GT_LINE_COLOR, 2)
            cv2.putText(frame, "IN ^", (10, line_y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_LINE_COLOR, 2)
            cv2.putText(frame, "OUT v", (10, line_y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_LINE_COLOR, 2)

            rows = by_frame.get(frame_idx)
            active = 0
            if rows is not None:
                active = len(rows)
                for _, r in rows.iterrows():
                    x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GT_BOX_COLOR, 1)
                    cv2.putText(frame, f"#{int(r.track_id)}", (x1, max(y1 - 4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, GT_BOX_COLOR, 1)

            cv2.putText(frame, f"GT  frame={frame_idx}/{total}", (width - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

            # STATS OVERLAY
            overlay = frame.copy()
            cv2.rectangle(overlay, (width - 220, 40), (width - 10, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"IN: {total_in} | OUT: {total_out}",
                        (width - 210, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
            cv2.putText(frame, f"Bees on ramp: {active}",
                        (width - 210, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

            writer.write(frame)

            if progress_cb and frame_idx % 30 == 0:
                progress_cb(frame_idx, total)
    finally:
        cap.release()
        writer.release()

    logger.info(f"GT video written: {output_path} ({frame_idx} frames)")
    return {"fps": fps, "width": width, "height": height, "total_frames": frame_idx}

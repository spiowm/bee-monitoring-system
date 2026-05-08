"""Оркестратор evaluation: запускає звичайний пайплайн → рендерить GT-відео → рахує метрики."""
import asyncio
import logging
from pathlib import Path

import cv2

from config import settings
from db.mongodb import get_db
from services.ffmpeg_service import convert_to_h264
from services.video_processor import process_video
from services.evaluation.gt_loader import (
    gt_paths, load_gt_tracks, load_entrance_zone, denormalize,
)
from services.evaluation.counting_eval import compute_gt_events, match_events
from services.evaluation.gt_annotator import render_gt_video

logger = logging.getLogger(__name__)


async def run_evaluation(
    job_id: str,
    video_path: str,
    gt_basename: str,
    config: dict,
    viz_config: dict,
) -> None:
    db = get_db()

    # 1. Перевіримо GT-файли заздалегідь
    paths = gt_paths(gt_basename)
    if not paths["tracks"].exists() or not paths["entrance_zone"].exists():
        msg = f"GT-файли для '{gt_basename}' не знайдено: {paths}"
        logger.error(msg)
        await db["jobs"].update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "error": msg}},
        )
        return

    # 2. Запускаємо звичайний пайплайн (детекція + трекінг + counting + анотоване відео).
    await process_video(job_id, video_path, config, viz_config)

    job = await db["jobs"].find_one({"job_id": job_id}, {"_id": 0})
    if not job or job.get("status") != "complete":
        logger.warning(f"Job {job_id} не завершився успішно — пропускаю evaluation")
        return

    pred_events = (job.get("result") or {}).get("events", []) or []

    # 3. Метаданні відео для денормалізації GT.
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    gt_df = denormalize(load_gt_tracks(paths["tracks"]), width, height)
    # Якщо вхідне відео коротше за повне (наприклад трейлер) — обрізати GT.
    gt_full_frames = int(gt_df["frame"].max())
    if total_frames > 0 and total_frames < gt_full_frames:
        gt_df = gt_df[gt_df["frame"] <= total_frames].copy()
        logger.info(
            f"Job {job_id}: GT обрізано {gt_full_frames} → {total_frames} кадрів "
            f"({len(gt_df):,} детекцій)"
        )
    zone = load_entrance_zone(paths["entrance_zone"])

    # 4. GT-події + matching.
    line_position = float(config.get("line_position", 0.5))
    gt_events = compute_gt_events(gt_df, zone, fps=fps, line_position=line_position)
    metrics = match_events(gt_events, pred_events, frame_window=15)
    logger.info(
        f"Job {job_id} eval: GT {metrics['gt_total_in']}/{metrics['gt_total_out']} IN/OUT, "
        f"pred {metrics['pred_total_in']}/{metrics['pred_total_out']}, "
        f"accuracy={metrics['accuracy']:.3f}"
    )

    # 5. GT-анотоване відео + конвертація в H.264.
    gt_raw_path = f"{settings.OUTPUT_DIR}/{job_id}_gt_raw.mp4"
    gt_final_path = f"{settings.OUTPUT_DIR}/{job_id}_gt.mp4"
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            render_gt_video,
            video_path, gt_df, zone, line_position, gt_raw_path, None, gt_events
        )
        await loop.run_in_executor(None, convert_to_h264, gt_raw_path, gt_final_path)
    except Exception as exc:
        logger.error(f"Job {job_id} GT video render failed: {exc}", exc_info=True)
        await db["jobs"].update_one(
            {"job_id": job_id},
            {"$set": {"evaluation_error": str(exc)}},
        )
        return

    # 6. Зберігаємо evaluation у документ задачі.
    evaluation_doc = {
        **metrics,
        "gt_events": gt_events,
        "gt_video_url": f"/static/output/{job_id}_gt.mp4",
        "gt_basename": gt_basename,
        "video_resolution": [width, height],
        "fps": fps,
        "line_position": line_position,
    }
    await db["jobs"].update_one(
        {"job_id": job_id},
        {"$set": {"evaluation": evaluation_doc}},
    )
    logger.info(f"Job {job_id} evaluation збережено в Mongo")

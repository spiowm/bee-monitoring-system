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
    gt_paths, load_gt_tracks, load_entrance_zone, denormalize, load_gt_behaviors, get_gt_behavior_classes,
)
from services.evaluation.counting_eval import compute_gt_events, match_events
from services.evaluation.gt_annotator import render_gt_video
from services.evaluation.behavior_eval import compute_behavior_metrics

logger = logging.getLogger(__name__)


async def run_evaluation(
    job_id: str,
    video_path: str,
    gt_basename: str,
    config: dict,
    viz_config: dict,
    eval_mode: str = "counting",
    skip_video: bool = False,
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

    # 2. Метаданні відео та GT
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    zone = load_entrance_zone(paths["entrance_zone"])

    # 3. Запускаємо звичайний пайплайн (детекція + трекінг + counting + анотоване відео).
    await process_video(job_id, video_path, config, viz_config, gt_entrance_zone=zone, skip_video=skip_video, eval_mode=eval_mode)

    job = await db["jobs"].find_one({"job_id": job_id}, {"_id": 0})
    if not job or job.get("status") != "complete":
        logger.warning(f"Job {job_id} не завершився успішно — пропускаю evaluation")
        return

    result = job.get("result") or {}
    
    if eval_mode == "behavior":
        gt_df = denormalize(load_gt_behaviors(paths["tracks"]), width, height)
        # Якщо вхідне відео коротше за повне — обрізати GT.
        gt_full_frames = int(gt_df["frame"].max())
        if total_frames > 0 and total_frames < gt_full_frames:
            gt_df = gt_df[gt_df["frame"] <= total_frames].copy()
            
        eval_classes = get_gt_behavior_classes(paths["tracks"])
        pred_per_frame = result.get("per_frame_behaviors", {})
        
        metrics = compute_behavior_metrics(gt_df, pred_per_frame, eval_classes)
        
        evaluation_doc = {
            "eval_mode": "behavior",
            **metrics,
            "gt_basename": gt_basename,
            "video_resolution": [width, height],
            "fps": fps,
        }
        
    else:
        # Default Counting Mode
        pred_events = result.get("events", []) or []

        # 4. Денормалізація GT та matching.
        gt_df = denormalize(load_gt_tracks(paths["tracks"]), width, height)
        gt_full_frames = int(gt_df["frame"].max())
        if total_frames > 0 and total_frames < gt_full_frames:
            gt_df = gt_df[gt_df["frame"] <= total_frames].copy()

        line_position = float(config.get("line_position", 0.0))
        gt_events = compute_gt_events(gt_df, zone, fps=fps, line_position=line_position)
        metrics = match_events(gt_events, pred_events, frame_window=15)

        # 5. GT-анотоване відео + конвертація в H.264.
        gt_video_url = None
        if not skip_video:
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
                gt_video_url = f"/static/output/{job_id}_gt.mp4"
            except Exception as exc:
                logger.error(f"Job {job_id} GT video render failed: {exc}", exc_info=True)
                await db["jobs"].update_one(
                    {"job_id": job_id},
                    {"$set": {"evaluation_error": str(exc)}},
                )
                return

        # 6. Зберігаємо evaluation у документ задачі.
        evaluation_doc = {
            "eval_mode": "counting",
            **metrics,
            "gt_events": gt_events,
            "gt_behaviors": {
                "fanning_tracks": int(gt_df[gt_df['fanning'] == 1]['track_id'].nunique()) if 'fanning' in gt_df.columns else 0,
                "defense_tracks": int(gt_df[gt_df['defensive'] == 1]['track_id'].nunique()) if 'defensive' in gt_df.columns else 0,
                "arrival_tracks": int(gt_df[gt_df['arrival'] == 1]['track_id'].nunique()) if 'arrival' in gt_df.columns else 0,
            },
            "gt_video_url": gt_video_url,
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

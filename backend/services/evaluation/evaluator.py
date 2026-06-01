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
    gt_paths, load_gt_behaviors, load_entrance_zone, denormalize,
)
from services.evaluation.gt_annotator import render_gt_video
from services.evaluation.behavior_eval import build_behavior_evaluation

logger = logging.getLogger(__name__)


async def run_evaluation(
    job_id: str,
    video_path: str,
    gt_basename: str,
    config: dict,
    viz_config: dict,
    eval_mode: str = "behavior",  # параметр лишено для сумісності сигнатури
    skip_video: bool = False,
) -> None:
    db = get_db()

    paths = gt_paths(gt_basename)
    if not paths["tracks"].exists() or not paths["entrance_zone"].exists():
        msg = f"GT-файли для '{gt_basename}' не знайдено: {paths}"
        logger.error(msg)
        await db["jobs"].update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "error": msg}},
        )
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    zone = load_entrance_zone(paths["entrance_zone"])

    # process_video повертає повний result у пам'яті (з per_frame_behaviors —
    # він має int-ключі й НЕ зберігається в Mongo, тож читаємо з повернутого значення)
    result = await process_video(
        job_id, video_path, config, viz_config,
        gt_entrance_zone=zone, skip_video=skip_video, eval_mode="behavior",
    )
    if not result:
        logger.warning(f"Job {job_id} не завершився успішно — пропускаю evaluation")
        return

    gt_df = denormalize(load_gt_behaviors(paths["tracks"]), width, height)
    gt_full_frames = int(gt_df["frame"].max())
    if total_frames > 0 and total_frames < gt_full_frames:
        gt_df = gt_df[gt_df["frame"] <= total_frames].copy()

    pred_per_frame = result.get("per_frame_behaviors", {})
    pred_events = result.get("events", []) or []
    warmup_frames = int(config.get("behavior_eval_warmup_frames", 80))

    metrics = build_behavior_evaluation(
        gt_df, pred_per_frame, pred_events, zone, fps,
        warmup_frames=warmup_frames,
        total_frames=total_frames,
    )

    # GT-відео з поведінковими мітками
    gt_video_url = None
    if not skip_video:
        gt_raw_path = f"{settings.OUTPUT_DIR}/{job_id}_gt_raw.mp4"
        gt_final_path = f"{settings.OUTPUT_DIR}/{job_id}_gt.mp4"
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, render_gt_video,
                video_path, gt_df, zone, 0.0, gt_raw_path, None, None,
            )
            await loop.run_in_executor(None, convert_to_h264, gt_raw_path, gt_final_path)
            gt_video_url = f"/static/output/{job_id}_gt.mp4"
        except Exception as exc:
            logger.error(f"Job {job_id} GT video render failed: {exc}", exc_info=True)

    evaluation_doc = {
        "eval_mode": "behavior",
        **metrics,
        "gt_video_url": gt_video_url,
        "gt_basename": gt_basename,
        "video_resolution": [width, height],
        "fps": fps,
    }

    await db["jobs"].update_one(
        {"job_id": job_id},
        {"$set": {"evaluation": evaluation_doc}},
    )
    logger.info(f"Job {job_id} evaluation збережено в Mongo")

import cv2
import time
import asyncio
import logging
from pathlib import Path
from ultralytics import YOLO

from config import settings
from services.pipeline import VideoPipeline
from services.ffmpeg_service import convert_to_h264
from db.mongodb import get_db

logger = logging.getLogger(__name__)

# Singleton bee model
_bee_model = None
_model_lock = asyncio.Lock()


async def get_bee_model():
    global _bee_model
    async with _model_lock:
        if _bee_model is None:
            logger.info(f"Loading Bee model from {settings.MODEL_PATH}...")
            _bee_model = YOLO(settings.MODEL_PATH)
        return _bee_model


async def process_video(job_id: str, video_path: str, config: dict, viz_config: dict):
    db = get_db()

    try:
        bee_model = await get_bee_model()
        pipeline = VideoPipeline(bee_model, config, viz_config)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_output_path = f"{settings.OUTPUT_DIR}/{job_id}_raw.mp4"
        final_output_path = f"{settings.OUTPUT_DIR}/{job_id}.mp4"
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

        frame_num = 0
        start_time = time.time()
        last_db_update = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            annotated = pipeline.process_frame(frame, frame_num, fps)
            out.write(annotated)

            # Periodic DB update
            if frame_num % 30 == 0 or time.time() - last_db_update > 2.0:
                last_db_update = time.time()
                progress = frame_num / total_frames if total_frames > 0 else 0
                await db["jobs"].update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "progress": progress,
                        "live_stats": pipeline.get_live_stats(frame_num, total_frames),
                    }},
                )

        cap.release()
        out.release()

        convert_to_h264(raw_output_path, final_output_path)

        duration = time.time() - start_time
        result = pipeline.get_result(frame_num, duration)
        result["annotated_video_url"] = f"/static/output/{job_id}.mp4"

        await db["jobs"].update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "complete",
                "progress": 1.0,
                "result": result,
            }},
        )

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await db["jobs"].update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
            }},
        )

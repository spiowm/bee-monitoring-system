import cv2
import time
import asyncio
import logging
import queue
import threading
import torch
from pathlib import Path
from ultralytics import YOLO

from config import settings
from services.pipeline import VideoPipeline
from services.ffmpeg_service import convert_to_h264
from db.mongodb import get_db

logger = logging.getLogger(__name__)

_bee_models: dict = {}
_model_lock = asyncio.Lock()
_cancel_flags: dict[str, asyncio.Event] = {}

_DEVICE = 0 if torch.cuda.is_available() else "cpu"
_HALF = isinstance(_DEVICE, int)
_BATCH_SIZE = 16       # frames per GPU call — larger batch → better GPU utilisation
_READ_AHEAD = 3        # read-queue depth in batches
_WRITE_AHEAD = 32      # write-queue depth in frames


def _resolve_model_path(model_name: str = None) -> str:
    if model_name:
        candidate = Path(settings.MODEL_PATH).parent.parent / model_name / "best.pt"
        if candidate.exists():
            return str(candidate)
        logger.warning(f"Model '{model_name}' not found at {candidate}, using default")
    return settings.MODEL_PATH


async def get_bee_model(model_name: str = None) -> YOLO:
    model_path = _resolve_model_path(model_name)
    async with _model_lock:
        if model_path not in _bee_models:
            logger.info(f"Loading model from {model_path} (device={_DEVICE}, half={_HALF})...")
            _bee_models[model_path] = YOLO(model_path)
        return _bee_models[model_path]


def request_cancel(job_id: str) -> None:
    if job_id in _cancel_flags:
        _cancel_flags[job_id].set()


def _reader_worker(
    cap: cv2.VideoCapture,
    out_q: "queue.Queue[list | None | Exception]",
    batch_size: int,
    stop: threading.Event,
) -> None:
    """Reads frames in a background thread and puts batches into out_q.
    Puts None sentinel when video is exhausted or stop is set.
    Puts the Exception object if an error occurs.
    """
    frame_num = 0
    try:
        while not stop.is_set():
            batch: list[tuple[int, object]] = []
            for _ in range(batch_size):
                if stop.is_set():
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
                batch.append((frame_num, frame))

            if batch:
                out_q.put(batch)

            if not batch or len(batch) < batch_size:
                out_q.put(None)
                return
    except Exception as exc:
        out_q.put(exc)


def _writer_worker(
    writer: cv2.VideoWriter,
    in_q: "queue.Queue[object | None]",
) -> None:
    """Writes annotated frames from in_q to disk until None sentinel arrives."""
    while True:
        try:
            item = in_q.get(timeout=30.0)
        except queue.Empty:
            logger.warning("Writer thread timed out waiting for frames")
            break
        if item is None:
            break
        writer.write(item)


async def process_video(job_id: str, video_path: str, config: dict, viz_config: dict):
    db = get_db()
    cancel = asyncio.Event()
    _cancel_flags[job_id] = cancel

    cap = None
    out = None
    raw_output_path = f"{settings.OUTPUT_DIR}/{job_id}_raw.mp4"
    final_output_path = f"{settings.OUTPUT_DIR}/{job_id}.mp4"

    try:
        bee_model = await get_bee_model(config.get("model_name"))
        pipeline = VideoPipeline(bee_model, config, viz_config)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

        conf = config.get("conf_threshold", 0.35)
        start_time = time.time()
        last_db_update = time.time()
        frame_num = 0

        # Producer-consumer: reader thread fills read_q while main loop does GPU inference
        stop_event = threading.Event()
        read_q: queue.Queue = queue.Queue(maxsize=_READ_AHEAD)
        write_q: queue.Queue = queue.Queue(maxsize=_WRITE_AHEAD)

        reader = threading.Thread(
            target=_reader_worker, args=(cap, read_q, _BATCH_SIZE, stop_event), daemon=True
        )
        writer = threading.Thread(
            target=_writer_worker, args=(out, write_q), daemon=True
        )
        reader.start()
        writer.start()

        try:
            while not cancel.is_set():
                try:
                    batch = read_q.get(timeout=30.0)
                except queue.Empty:
                    logger.warning(f"Job {job_id}: read queue timeout")
                    break

                if batch is None:
                    break
                if isinstance(batch, Exception):
                    raise batch

                # Single GPU call for the whole batch
                batch_start = time.time()
                raw_frames = [f for _, f in batch]
                batch_results = bee_model(
                    raw_frames, verbose=False, conf=conf, device=_DEVICE, half=_HALF
                )

                for (fn, frame), detection in zip(batch, batch_results):
                    annotated = pipeline.process_frame(frame, fn, fps, detection_result=detection)
                    write_q.put(annotated)

                frame_num = batch[-1][0]
                pipeline.pipeline_state["current_fps"] = len(batch) / (time.time() - batch_start + 0.001)

                if time.time() - last_db_update > 2.0:
                    last_db_update = time.time()
                    progress = frame_num / total_frames if total_frames > 0 else 0
                    await db["jobs"].update_one(
                        {"job_id": job_id},
                        {"$set": {
                            "progress": progress,
                            "live_stats": pipeline.get_live_stats(frame_num, total_frames),
                        }},
                    )
        finally:
            stop_event.set()
            write_q.put(None)           # signal writer to drain and exit
            reader.join(timeout=5.0)
            writer.join(timeout=20.0)   # wait for all writes to flush

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await db["jobs"].update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "error": str(e)}},
        )
        return

    finally:
        if cap:
            cap.release()
        if out:
            out.release()
        _cancel_flags.pop(job_id, None)

    if cancel.is_set():
        Path(raw_output_path).unlink(missing_ok=True)
        Path(video_path).unlink(missing_ok=True)
        return  # DB record already deleted by DELETE endpoint

    # Run FFmpeg in thread pool so the event loop stays responsive to other requests
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, convert_to_h264, raw_output_path, final_output_path)

    duration = time.time() - start_time
    result = pipeline.get_result(frame_num, duration)
    result["annotated_video_url"] = f"/static/output/{job_id}.mp4"

    logger.info(
        f"Job {job_id} done: {frame_num} frames / {duration:.1f}s = "
        f"{result['fps_processed']:.1f} fps processed"
    )

    await db["jobs"].update_one(
        {"job_id": job_id},
        {"$set": {
            "status": "complete",
            "progress": 1.0,
            "result": result,
        }},
    )

import cv2
import numpy as np
import time
import asyncio
import logging
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

from config import settings
from services.ramp_detector import RampDetector
from services.tracker_factory import TrackerFactory
from services.behavior import BehaviorAnalyzer
from services.counter import TrafficCounter
from services.annotator import FrameAnnotator
from services.ffmpeg_service import convert_to_h264
from db.mongodb import get_db

logger = logging.getLogger(__name__)

# Singletons cache
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
        ramp_detector = RampDetector()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize components
        tracker = TrackerFactory.create(config.get("tracker_name", "bytetrack"))
        counter = TrafficCounter(
            line_position=config.get("line_position", 0.5),
            approach=config.get("approach", "A"),
            angle_threshold_deg=config.get("angle_threshold_deg", 60.0)
        )
        behavior_analyzer = BehaviorAnalyzer()
        annotator = FrameAnnotator(viz_config)
        
        raw_output_path = f"{settings.OUTPUT_DIR}/{job_id}_raw.mp4"
        final_output_path = f"{settings.OUTPUT_DIR}/{job_id}.mp4"
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        all_events = []
        behavior_counts = {"foraging": 0, "fanning": 0, "guarding": 0, "washboarding": 0}
        total_in, total_out = 0, 0
        pose_confirmed, fallback_events = 0, 0
        
        start_time = time.time()
        last_db_update = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_start = time.time()
            frame_num += 1
            
            ramp_bbox = ramp_detector.detect(frame)
            
            # Predict
            results = bee_model(frame, verbose=False, conf=config.get("conf_threshold", 0.35))
            
            filtered_boxes = []
            filtered_confs = []
            filtered_class_ids = []
            filtered_kpts = []
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy()
                kpts = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None
                
                for i in range(len(boxes)):
                    if RampDetector.is_in_ramp(boxes[i], ramp_bbox):
                        filtered_boxes.append(boxes[i])
                        filtered_confs.append(confs[i])
                        filtered_class_ids.append(cls_ids[i])
                        if kpts is not None and len(kpts) > i:
                            filtered_kpts.append(kpts[i])
                            
            if len(filtered_boxes) > 0:
                detections = sv.Detections(
                    xyxy=np.array(filtered_boxes),
                    confidence=np.array(filtered_confs),
                    class_id=np.array(filtered_class_ids)
                )
            else:
                detections = sv.Detections.empty()
                
            tracked = tracker.update_with_detections(detections)
            
            behavior_analyzer.update_history(frame_num, tracked, fps)
            current_behaviors = {}
            if frame_num % 15 == 0:
                current_behaviors = behavior_analyzer.analyze(fps)
                behavior_counts = {k: 0 for k in behavior_counts}
                for b in current_behaviors.values():
                    if b in behavior_counts:
                        behavior_counts[b] += 1
                        
            # Match keypoints to tracked detections by spatial overlap (IoU)
            # Simplification: we might re-match or keep it simple if sv keeps index
            # supervision update_with_detections sorts/filters differently
            # For simplicity let's assume track bounding box centers can map to kpts
            # We will just pass the filtered_kpts and let the logic handle it or not map perfectly right now
            # To be precise, we need trackers that maintain original indices, but standard ByteTrack might not.
            # Workaround: For approach B, we find closest keypoints by box center
            mapped_kpts = []
            if len(filtered_kpts) > 0 and tracked.tracker_id is not None:
                for xyxy in tracked.xyxy:
                    tcx, tcy = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
                    best_dist = float('inf')
                    best_kpt = None
                    for j, fb in enumerate(filtered_boxes):
                        fcx, fcy = (fb[0]+fb[2])/2, (fb[1]+fb[3])/2
                        dist = (tcx-fcx)**2 + (tcy-fcy)**2
                        if dist < best_dist and dist < 1000:
                            best_dist = dist
                            best_kpt = filtered_kpts[j]
                    mapped_kpts.append(best_kpt)
            
            events = counter.update(
                frame_num, tracked, ramp_bbox, mapped_kpts, 
                current_behaviors, fps
            )
            
            for e in events:
                all_events.append(e)
                if e["direction"] == "IN": total_in += 1
                if e["direction"] == "OUT": total_out += 1
                if e["method"] == "pose_confirmed": pose_confirmed += 1
                if e["method"] == "trajectory_fallback": fallback_events += 1
                
            stats_state = {
                "total_in": total_in,
                "total_out": total_out,
                "fps": 1.0 / (time.time() - frame_start + 0.001)
            }
            
            annotated_frame = annotator.annotate(
                frame, tracked, ramp_bbox, mapped_kpts, current_behaviors, counter, events, stats_state
            )
            
            out.write(annotated_frame)
            
            if frame_num % 30 == 0 or time.time() - last_db_update > 2.0:
                last_db_update = time.time()
                await db["jobs"].update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "progress": frame_num / total_frames if total_frames > 0 else 0,
                        "live_stats": {
                            "current_frame": frame_num,
                            "total_frames": total_frames,
                            "bees_on_ramp": len(tracked.tracker_id) if tracked.tracker_id is not None else 0,
                            "total_in": total_in,
                            "total_out": total_out,
                            "current_fps": stats_state["fps"],
                            "tracker_name": config.get("tracker_name", "bytetrack"),
                            "approach": config.get("approach", "A"),
                            "pose_confirmed": pose_confirmed,
                            "fallback_events": fallback_events,
                            "behavior_counts": behavior_counts
                        }
                    }}
                )

        cap.release()
        out.release()
        
        convert_to_h264(raw_output_path, final_output_path)
        
        duration = time.time() - start_time
        
        await db["jobs"].update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "complete",
                "progress": 1.0,
                "result": {
                    "total_in": total_in,
                    "total_out": total_out,
                    "total_frames": frame_num,
                    "duration_sec": duration,
                    "fps_processed": frame_num / duration,
                    "approach_used": config.get("approach", "A"),
                    "tracker_used": config.get("tracker_name", "bytetrack"),
                    "pose_confirmed_events": pose_confirmed,
                    "fallback_events": fallback_events,
                    "ramp_detected": ramp_bbox is not None,
                    "annotated_video_url": f"/static/output/{job_id}.mp4",
                    "behavior_summary": {
                        f"{k}_detections": v for k,v in behavior_counts.items()
                    },
                    "events": all_events
                }
            }}
        )
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await db["jobs"].update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "failed",
                "error": str(e)
            }}
        )

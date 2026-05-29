import numpy as np
import time
import logging
from ultralytics import YOLO

from services.tracker_factory import TrackerFactory
from services.behavior import BehaviorAnalyzer
from services.counter import TrafficCounter
from services.annotator import FrameAnnotator
from services.track_history import TrackHistory
from services.pipeline_stages import (
    FrameContext, DetectionStage, TrackingStage, TrackUpdateStage,
    BehaviorStage, DefenseStage, CountingStage, AnnotationStage,
)

logger = logging.getLogger(__name__)

class VideoPipeline:
    """Detection → tracking → counting → annotation orchestrated via modular PipelineStages."""

    def __init__(self, bee_model: YOLO, config: dict, viz_config: dict, gt_entrance_zone=None):
        self.config = config
        
        # Instantiate services
        tracker = TrackerFactory.create(config.get("tracker_name", "bytetrack"), config=config)
        counter = TrafficCounter(
            line_position=config.get("line_position", 0.0),
            approach=config.get("approach", "A"),
            angle_threshold_deg=config.get("angle_threshold_deg", 60.0),
        )
        behavior_analyzer = BehaviorAnalyzer(config)
        annotator = FrameAnnotator(viz_config)
        history = TrackHistory()

        # Orchestrate stages
        self.stages = [
            DetectionStage(bee_model, config, gt_entrance_zone=gt_entrance_zone),
            TrackingStage(tracker),
            TrackUpdateStage(history),
            BehaviorStage(behavior_analyzer),
            DefenseStage(config),
            CountingStage(counter),
            AnnotationStage(annotator, counter),
        ]

        # Accumulated pipeline state
        self.pipeline_state = {
            "history": history,
            "total_in": 0,
            "total_out": 0,
            "pose_confirmed": 0,
            "fallback_events": 0,
            "pose_rejected": 0,
            "all_events": [],
            "rejected_events": [],
            "defense_events": [],
            "behavior_counts": {"foraging": 0, "fanning": 0, "washboarding": 0, "defense": 0, "unknown": 0},
            "current_behaviors": {},
            "per_frame_behaviors": {}, # frame -> {track_id: behavior}
            "active_bees": 0,
            "current_fps": 0.0,
        }

    def process_frame(self, frame: np.ndarray, frame_num: int, fps: float, detection_result=None) -> np.ndarray:
        frame_start = time.perf_counter()

        ctx = FrameContext(frame=frame, frame_num=frame_num, fps=fps, detection_result=detection_result)

        for stage in self.stages:
            stage_start = time.perf_counter()
            stage.process(ctx, self.pipeline_state)
            stage_ms = (time.perf_counter() - stage_start) * 1000.0
            
            if isinstance(stage, DetectionStage):
                ctx.timings.detection_ms += stage_ms
            elif isinstance(stage, TrackingStage) or isinstance(stage, TrackUpdateStage):
                ctx.timings.tracking_ms += stage_ms
            elif isinstance(stage, BehaviorStage):
                ctx.timings.behavior_ms += stage_ms
            elif isinstance(stage, DefenseStage):
                ctx.timings.defense_ms += stage_ms
            elif isinstance(stage, CountingStage):
                ctx.timings.counting_ms += stage_ms
            elif isinstance(stage, AnnotationStage):
                ctx.timings.annotation_ms += stage_ms

        ctx.timings.total_ms = (time.perf_counter() - frame_start) * 1000.0
        
        if "recent_timings" not in self.pipeline_state:
            self.pipeline_state["recent_timings"] = []
        self.pipeline_state["recent_timings"].append(ctx.timings)
        if len(self.pipeline_state["recent_timings"]) > 30:
            self.pipeline_state["recent_timings"].pop(0)

        self.pipeline_state["current_fps"] = 1000.0 / (ctx.timings.total_ms + 0.001)

        # Collect behavior evaluation data.
        # Записуємо ВСІХ затреканих бджіл (включно з "unknown"), щоб evaluation
        # міг відрізнити провал детекції (бджоли немає взагалі) від провалу
        # класифікації (бджола є, але стан "unknown"/не той).
        if "per_frame_behaviors" in self.pipeline_state:
            frame_behaviors = {}
            if ctx.tracked_detections is not None and ctx.tracked_detections.tracker_id is not None:
                for i, tid in enumerate(ctx.tracked_detections.tracker_id):
                    tid_int = int(tid)
                    b = self.pipeline_state["current_behaviors"].get(tid_int)
                    frame_behaviors[tid_int] = {
                        "bbox": ctx.tracked_detections.xyxy[i].tolist(),
                        "behavior": b or "unknown",
                    }
            if frame_behaviors:
                self.pipeline_state["per_frame_behaviors"][ctx.frame_num] = frame_behaviors

        return ctx.annotated_frame

    def get_live_stats(self, frame_num: int, total_frames: int) -> dict:
        all_events = self.pipeline_state.get("all_events", [])
        rejected_events = self.pipeline_state.get("rejected_events", [])
        
        # Count reasons
        reject_reasons = {"no_keypoints": 0, "angle_mismatch": 0}
        angle_histogram = {"0-15": 0, "15-30": 0, "30-45": 0, "45-60": 0, ">60": 0}

        for rev in rejected_events:
            reason = rev.get("reject_reason") or "no_keypoints"
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            if rev.get("angle_deg") is not None:
                ang = rev["angle_deg"]
                if ang <= 15: angle_histogram["0-15"] += 1
                elif ang <= 30: angle_histogram["15-30"] += 1
                elif ang <= 45: angle_histogram["30-45"] += 1
                elif ang <= 60: angle_histogram["45-60"] += 1
                else: angle_histogram[">60"] += 1

        for ev in all_events:
            if ev.get("angle_deg") is not None:
                ang = ev["angle_deg"]
                if ang <= 15: angle_histogram["0-15"] += 1
                elif ang <= 30: angle_histogram["15-30"] += 1
                elif ang <= 45: angle_histogram["30-45"] += 1
                elif ang <= 60: angle_histogram["45-60"] += 1
                else: angle_histogram[">60"] += 1

        recent_timings = self.pipeline_state.get("recent_timings", [])
        avg_timings = None
        if recent_timings:
            n = len(recent_timings)
            avg_timings = {
                "detection_ms": sum(t.detection_ms for t in recent_timings) / n,
                "tracking_ms": sum(t.tracking_ms for t in recent_timings) / n,
                "behavior_ms": sum(t.behavior_ms for t in recent_timings) / n,
                "defense_ms": sum(t.defense_ms for t in recent_timings) / n,
                "counting_ms": sum(t.counting_ms for t in recent_timings) / n,
                "annotation_ms": sum(t.annotation_ms for t in recent_timings) / n,
                "total_ms": sum(t.total_ms for t in recent_timings) / n,
            }
            avg_timings["pipeline_fps"] = 1000.0 / avg_timings["total_ms"] if avg_timings["total_ms"] > 0 else 0
            avg_timings["model_fps"] = 1000.0 / avg_timings["detection_ms"] if avg_timings["detection_ms"] > 0 else 0

        return {
            "current_frame": frame_num,
            "total_frames": total_frames,
            "bees_on_ramp": self.pipeline_state["active_bees"],
            "total_in": self.pipeline_state["total_in"],
            "total_out": self.pipeline_state["total_out"],
            "current_fps": self.pipeline_state["current_fps"],
            "tracker_name": self.config.get("tracker_name", "bytetrack"),
            "approach": self.config.get("approach", "A"),
            "pose_confirmed": self.pipeline_state["pose_confirmed"],
            "fallback_events": self.pipeline_state["fallback_events"],
            "pose_rejected": len(rejected_events),
            "reject_reasons": reject_reasons,
            "angle_histogram": angle_histogram,
            "behavior_counts": self.pipeline_state["behavior_counts"],
            "recent_events": all_events[-10:] if all_events else [],
            "timings": avg_timings,
        }

    def get_result(self, total_frames: int, duration: float) -> dict:
        from services.ramp_detector import RampDetector
        from services.pipeline_stages import CountingStage
        ramp_bbox = RampDetector().cached_bbox
        
        debounce_blocks = 0
        for stage in self.stages:
            if isinstance(stage, CountingStage):
                debounce_blocks = stage.counter.debounce_blocks

        all_events = self.pipeline_state.get("all_events", [])
        rejected_events = self.pipeline_state.get("rejected_events", [])
        # Count reasons
        reject_reasons = {"no_keypoints": 0, "angle_mismatch": 0}
        angle_histogram = {"0-15": 0, "15-30": 0, "30-45": 0, "45-60": 0, ">60": 0}
        
        for rev in rejected_events:
            reason = rev.get("reject_reason") or "no_keypoints"
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            if rev.get("angle_deg") is not None:
                ang = rev["angle_deg"]
                if ang <= 15: angle_histogram["0-15"] += 1
                elif ang <= 30: angle_histogram["15-30"] += 1
                elif ang <= 45: angle_histogram["30-45"] += 1
                elif ang <= 60: angle_histogram["45-60"] += 1
                else: angle_histogram[">60"] += 1

        for ev in all_events:
            if ev.get("angle_deg") is not None:
                ang = ev["angle_deg"]
                if ang <= 15: angle_histogram["0-15"] += 1
                elif ang <= 30: angle_histogram["15-30"] += 1
                elif ang <= 45: angle_histogram["30-45"] += 1
                elif ang <= 60: angle_histogram["45-60"] += 1
                else: angle_histogram[">60"] += 1

        return {
            "total_in": self.pipeline_state["total_in"],
            "total_out": self.pipeline_state["total_out"],
            "total_frames": total_frames,
            "duration_sec": duration,
            "fps_processed": total_frames / duration if duration > 0 else 0,
            "approach_used": self.config.get("approach", "A"),
            "tracker_used": self.config.get("tracker_name", "bytetrack"),
            "pose_confirmed_events": self.pipeline_state["pose_confirmed"],
            "fallback_events": self.pipeline_state["fallback_events"],
            "pose_rejected_events": len(rejected_events),
            "debounce_blocks": debounce_blocks,
            "reject_reasons": reject_reasons,
            "angle_histogram": angle_histogram,
            "ramp_detected": ramp_bbox is not None,
            "behavior_summary": {
                f"{k}_detections": v for k, v in self.pipeline_state["behavior_counts"].items()
            },
            "defense_events": self.pipeline_state.get("defense_events", []),
            "events": self.pipeline_state["all_events"],
            "rejected_events": rejected_events,
            "per_frame_behaviors": self.pipeline_state.get("per_frame_behaviors", {}),
        }

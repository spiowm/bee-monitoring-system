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
    BehaviorStage, CountingStage, AnnotationStage
)

logger = logging.getLogger(__name__)

class VideoPipeline:
    """Detection → tracking → counting → annotation orchestrated via modular PipelineStages."""

    def __init__(self, bee_model: YOLO, config: dict, viz_config: dict):
        self.config = config
        
        # Instantiate services
        tracker = TrackerFactory.create(config.get("tracker_name", "bytetrack"))
        counter = TrafficCounter(
            line_position=config.get("line_position", 0.5),
            approach=config.get("approach", "A"),
            angle_threshold_deg=config.get("angle_threshold_deg", 60.0),
        )
        behavior_analyzer = BehaviorAnalyzer(config)
        annotator = FrameAnnotator(viz_config)
        history = TrackHistory()

        # Orchestrate stages
        self.stages = [
            DetectionStage(bee_model, config),
            TrackingStage(tracker),
            TrackUpdateStage(history),
            BehaviorStage(behavior_analyzer),
            CountingStage(counter),
            AnnotationStage(annotator, counter)
        ]

        # Accumulated pipeline state
        self.pipeline_state = {
            "history": history,
            "total_in": 0,
            "total_out": 0,
            "pose_confirmed": 0,
            "fallback_events": 0,
            "all_events": [],
            "behavior_counts": {"foraging": 0, "fanning": 0, "guarding": 0, "washboarding": 0},
            "current_behaviors": {},
            "active_bees": 0,
            "current_fps": 0.0,
        }

    def process_frame(self, frame: np.ndarray, frame_num: int, fps: float, detection_result=None) -> np.ndarray:
        frame_start = time.time()

        ctx = FrameContext(frame=frame, frame_num=frame_num, fps=fps, detection_result=detection_result)

        for stage in self.stages:
            stage.process(ctx, self.pipeline_state)

        self.pipeline_state["current_fps"] = 1.0 / (time.time() - frame_start + 0.001)

        return ctx.annotated_frame

    def get_live_stats(self, frame_num: int, total_frames: int) -> dict:
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
            "behavior_counts": self.pipeline_state["behavior_counts"],
        }

    def get_result(self, total_frames: int, duration: float) -> dict:
        from services.ramp_detector import RampDetector
        ramp_bbox = RampDetector().cached_bbox
        
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
            "ramp_detected": ramp_bbox is not None,
            "behavior_summary": {
                f"{k}_detections": v for k, v in self.pipeline_state["behavior_counts"].items()
            },
            "events": self.pipeline_state["all_events"],
        }

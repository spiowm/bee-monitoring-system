from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import supervision as sv
import torch

_DEVICE = 0 if torch.cuda.is_available() else "cpu"
_HALF = isinstance(_DEVICE, int)

from services.ramp_detector import RampDetector
from services.track_history import TrackHistory
from services.counter import TrafficCounter
from services.behavior import BehaviorAnalyzer
from services.annotator import FrameAnnotator

@dataclass
class FrameContext:
    frame: np.ndarray
    frame_num: int
    fps: float

    detection_result: object = None  # pre-computed YOLO result (batch mode)
    ramp_bbox: tuple = None
    detections: sv.Detections = None
    filtered_boxes: list = field(default_factory=list)
    filtered_kpts: list = field(default_factory=list)

    tracked_detections: sv.Detections = None
    mapped_kpts: list = field(default_factory=list)

    events: list = field(default_factory=list)
    annotated_frame: np.ndarray = None

class PipelineStage(ABC):
    @abstractmethod
    def process(self, ctx: FrameContext, state: dict):
        pass

class DetectionStage(PipelineStage):
    def __init__(self, bee_model, config):
        self.bee_model = bee_model
        self.config = config
        self.ramp_detector = RampDetector()

    def process(self, ctx: FrameContext, state: dict):
        ctx.ramp_bbox = self.ramp_detector.detect(ctx.frame)

        if ctx.detection_result is not None:
            results = [ctx.detection_result]
        else:
            results = self.bee_model(
                ctx.frame, verbose=False,
                conf=self.config.get("conf_threshold", 0.35),
                device=_DEVICE, half=_HALF,
            )

        if results and len(results[0].boxes) > 0:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            confs   = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()
            kpts    = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

            # Vectorized centroid-in-ramp filter
            if ctx.ramp_bbox is not None:
                rx1, ry1, rx2, ry2 = ctx.ramp_bbox
                cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
                cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
                mask = (cx >= rx1 - 10) & (cx <= rx2 + 10) & (cy >= ry1 - 10) & (cy <= ry2 + 10)
            else:
                mask = np.ones(len(boxes), dtype=bool)

            boxes_f   = boxes[mask]
            confs_f   = confs[mask]
            cls_ids_f = cls_ids[mask]
            kpts_f    = kpts[mask].tolist() if kpts is not None else []

            if len(boxes_f):
                ctx.detections = sv.Detections(
                    xyxy=boxes_f,
                    confidence=confs_f,
                    class_id=cls_ids_f,
                )
                ctx.filtered_boxes = boxes_f
                ctx.filtered_kpts  = kpts_f
                return

        ctx.detections    = sv.Detections.empty()
        ctx.filtered_boxes = []
        ctx.filtered_kpts  = []


class TrackingStage(PipelineStage):
    def __init__(self, tracker):
        self.tracker = tracker
        
    def _iou_matrix(self, tracked: np.ndarray, filtered: np.ndarray) -> np.ndarray:
        """Vectorized IoU: (N,4) × (M,4) → (N,M)."""
        x_left   = np.maximum(tracked[:, None, 0], filtered[None, :, 0])
        y_top    = np.maximum(tracked[:, None, 1], filtered[None, :, 1])
        x_right  = np.minimum(tracked[:, None, 2], filtered[None, :, 2])
        y_bottom = np.minimum(tracked[:, None, 3], filtered[None, :, 3])
        inter = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
        area_t = (tracked[:, 2] - tracked[:, 0]) * (tracked[:, 3] - tracked[:, 1])
        area_f = (filtered[:, 2] - filtered[:, 0]) * (filtered[:, 3] - filtered[:, 1])
        union  = area_t[:, None] + area_f[None, :] - inter
        return np.where(union > 0, inter / union, 0.0)

    def process(self, ctx: FrameContext, state: dict):
        ctx.tracked_detections = self.tracker.update_with_detections(ctx.detections)
        state["active_bees"] = (
            len(ctx.tracked_detections.tracker_id)
            if ctx.tracked_detections.tracker_id is not None else 0
        )

        mapped: list = []
        if (ctx.filtered_kpts
                and ctx.tracked_detections.tracker_id is not None
                and len(ctx.tracked_detections.xyxy) > 0):
            filtered_arr = np.asarray(ctx.filtered_boxes)
            iou = self._iou_matrix(ctx.tracked_detections.xyxy, filtered_arr)
            best_j   = np.argmax(iou, axis=1)
            best_iou = iou[np.arange(len(ctx.tracked_detections.xyxy)), best_j]
            mapped = [
                ctx.filtered_kpts[j] if v > 0.3 else None
                for j, v in zip(best_j, best_iou)
            ]
        ctx.mapped_kpts = mapped


class TrackUpdateStage(PipelineStage):
    def __init__(self, history: TrackHistory):
        self.history = history
        
    def process(self, ctx: FrameContext, state: dict):
        if ctx.tracked_detections.tracker_id is not None:
            for i, track_id in enumerate(ctx.tracked_detections.tracker_id):
                xyxy = ctx.tracked_detections.xyxy[i]
                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                self.history.update(track_id, cx, cy, ctx.frame_num)
        self.history.prune_stale(ctx.frame_num)


class BehaviorStage(PipelineStage):
    def __init__(self, analyzer: BehaviorAnalyzer):
        self.analyzer = analyzer
        
    def process(self, ctx: FrameContext, state: dict):
        if ctx.frame_num % 15 == 0:
            current_behaviors = self.analyzer.analyze(state["history"], ctx.fps)
            state["current_behaviors"] = current_behaviors
            
            counts = {k: 0 for k in state["behavior_counts"]}
            for b in current_behaviors.values():
                if b in counts:
                    counts[b] += 1
            state["behavior_counts"] = counts


class CountingStage(PipelineStage):
    def __init__(self, counter: TrafficCounter):
        self.counter = counter
        
    def process(self, ctx: FrameContext, state: dict):
        events = self.counter.update(
            ctx.frame_num, ctx.tracked_detections, ctx.ramp_bbox, ctx.mapped_kpts,
            state["history"], state["current_behaviors"], ctx.fps,
        )
        ctx.events = events
        for e in events:
            state["all_events"].append(e)
            if e["direction"] == "IN": state["total_in"] += 1
            if e["direction"] == "OUT": state["total_out"] += 1
            if e["method"] == "pose_confirmed": state["pose_confirmed"] += 1
            if e["method"] == "trajectory_fallback": state["fallback_events"] += 1


class AnnotationStage(PipelineStage):
    def __init__(self, annotator: FrameAnnotator, counter: TrafficCounter):
        self.annotator = annotator
        self.counter = counter
        
    def process(self, ctx: FrameContext, state: dict):
        stats_state = {
            "total_in": state["total_in"],
            "total_out": state["total_out"],
            "fps": state["current_fps"],
        }
        ctx.annotated_frame = self.annotator.annotate(
            ctx.frame, ctx.tracked_detections, ctx.ramp_bbox, ctx.mapped_kpts,
            state["current_behaviors"], self.counter, ctx.events, stats_state, state["history"],
        )

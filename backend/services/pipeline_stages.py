from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import supervision as sv

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
        
        results = self.bee_model(
            ctx.frame, verbose=False,
            conf=self.config.get("conf_threshold", 0.35),
        )
        filtered_boxes, filtered_confs, filtered_class_ids, filtered_kpts = [], [], [], []

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()
            kpts = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

            for i in range(len(boxes)):
                if RampDetector.is_in_ramp(boxes[i], ctx.ramp_bbox):
                    filtered_boxes.append(boxes[i])
                    filtered_confs.append(confs[i])
                    filtered_class_ids.append(cls_ids[i])
                    if kpts is not None and len(kpts) > i:
                        filtered_kpts.append(kpts[i])

        if filtered_boxes:
            ctx.detections = sv.Detections(
                xyxy=np.array(filtered_boxes),
                confidence=np.array(filtered_confs),
                class_id=np.array(filtered_class_ids),
            )
        else:
            ctx.detections = sv.Detections.empty()
            
        ctx.filtered_boxes = filtered_boxes
        ctx.filtered_kpts = filtered_kpts


class TrackingStage(PipelineStage):
    def __init__(self, tracker):
        self.tracker = tracker
        
    def _compute_iou(self, box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top: return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def process(self, ctx: FrameContext, state: dict):
        ctx.tracked_detections = self.tracker.update_with_detections(ctx.detections)
        state["active_bees"] = len(ctx.tracked_detections.tracker_id) if ctx.tracked_detections.tracker_id is not None else 0
        
        # Match Keypoints mapping
        mapped = []
        if ctx.filtered_kpts and ctx.tracked_detections.tracker_id is not None:
            for xyxy in ctx.tracked_detections.xyxy:
                best_iou = 0.0
                best_kpt = None
                for j, fb in enumerate(ctx.filtered_boxes):
                    iou = self._compute_iou(xyxy, fb)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_kpt = ctx.filtered_kpts[j]
                mapped.append(best_kpt)
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

import numpy as np
import time
import logging
from ultralytics import YOLO
import supervision as sv

from services.ramp_detector import RampDetector
from services.tracker_factory import TrackerFactory
from services.behavior import BehaviorAnalyzer
from services.counter import TrafficCounter
from services.annotator import FrameAnnotator
from services.track_history import TrackHistory

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Detection → tracking → counting → annotation for a single video."""

    def __init__(self, bee_model: YOLO, config: dict, viz_config: dict):
        self.bee_model = bee_model
        self.config = config
        self.ramp_detector = RampDetector()
        self.tracker = TrackerFactory.create(config.get("tracker_name", "bytetrack"))
        self.counter = TrafficCounter(
            line_position=config.get("line_position", 0.5),
            approach=config.get("approach", "A"),
            angle_threshold_deg=config.get("angle_threshold_deg", 60.0),
        )
        self.behavior_analyzer = BehaviorAnalyzer()
        self.annotator = FrameAnnotator(viz_config)
        self.history = TrackHistory()

        # Accumulated state
        self.total_in = 0
        self.total_out = 0
        self.pose_confirmed = 0
        self.fallback_events = 0
        self.all_events = []
        self.behavior_counts = {"foraging": 0, "fanning": 0, "guarding": 0, "washboarding": 0}
        self.current_behaviors = {}
        self.active_bees = 0
        self.current_fps = 0.0

    def process_frame(self, frame: np.ndarray, frame_num: int, fps: float) -> np.ndarray:
        frame_start = time.time()

        ramp_bbox = self.ramp_detector.detect(frame)
        detections, filtered_boxes, filtered_kpts = self._detect(frame, ramp_bbox)
        tracked = self.tracker.update_with_detections(detections)
        self.active_bees = len(tracked.tracker_id) if tracked.tracker_id is not None else 0

        # Update shared history
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                xyxy = tracked.xyxy[i]
                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                self.history.update(track_id, cx, cy, frame_num)
        self.history.prune_stale(frame_num)

        # Behavior analysis every 15 frames
        if frame_num % 15 == 0:
            self.current_behaviors = self.behavior_analyzer.analyze(self.history, fps)
            self.behavior_counts = {k: 0 for k in self.behavior_counts}
            for b in self.current_behaviors.values():
                if b in self.behavior_counts:
                    self.behavior_counts[b] += 1

        mapped_kpts = self._match_keypoints(tracked, filtered_boxes, filtered_kpts)

        events = self.counter.update(
            frame_num, tracked, ramp_bbox, mapped_kpts,
            self.history, self.current_behaviors, fps,
        )
        self._accumulate_events(events)

        stats = {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "fps": 1.0 / (time.time() - frame_start + 0.001),
        }
        self.current_fps = stats["fps"]

        return self.annotator.annotate(
            frame, tracked, ramp_bbox, mapped_kpts,
            self.current_behaviors, self.counter, events, stats, self.history,
        )

    # ------------------------------------------------------------------

    def _detect(self, frame, ramp_bbox):
        results = self.bee_model(
            frame, verbose=False,
            conf=self.config.get("conf_threshold", 0.35),
        )
        filtered_boxes, filtered_confs, filtered_class_ids, filtered_kpts = [], [], [], []

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

        if filtered_boxes:
            detections = sv.Detections(
                xyxy=np.array(filtered_boxes),
                confidence=np.array(filtered_confs),
                class_id=np.array(filtered_class_ids),
            )
        else:
            detections = sv.Detections.empty()

        return detections, filtered_boxes, filtered_kpts

    def _match_keypoints(self, tracked, filtered_boxes, filtered_kpts):
        def compute_iou(box1, box2):
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - intersection_area
            return intersection_area / union_area if union_area > 0 else 0.0

        mapped = []
        if filtered_kpts and tracked.tracker_id is not None:
            for xyxy in tracked.xyxy:
                best_iou = 0.0
                best_kpt = None
                for j, fb in enumerate(filtered_boxes):
                    iou = compute_iou(xyxy, fb)
                    if iou > best_iou and iou > 0.3:  # require at least 30% overlap
                        best_iou = iou
                        best_kpt = filtered_kpts[j]
                mapped.append(best_kpt)
        return mapped

    def _accumulate_events(self, events):
        for e in events:
            self.all_events.append(e)
            if e["direction"] == "IN":
                self.total_in += 1
            if e["direction"] == "OUT":
                self.total_out += 1
            if e["method"] == "pose_confirmed":
                self.pose_confirmed += 1
            if e["method"] == "trajectory_fallback":
                self.fallback_events += 1

    def get_live_stats(self, frame_num: int, total_frames: int) -> dict:
        return {
            "current_frame": frame_num,
            "total_frames": total_frames,
            "bees_on_ramp": self.active_bees,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "current_fps": self.current_fps,
            "tracker_name": self.config.get("tracker_name", "bytetrack"),
            "approach": self.config.get("approach", "A"),
            "pose_confirmed": self.pose_confirmed,
            "fallback_events": self.fallback_events,
            "behavior_counts": self.behavior_counts,
        }

    def get_result(self, total_frames: int, duration: float) -> dict:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_frames": total_frames,
            "duration_sec": duration,
            "fps_processed": total_frames / duration if duration > 0 else 0,
            "approach_used": self.config.get("approach", "A"),
            "tracker_used": self.config.get("tracker_name", "bytetrack"),
            "pose_confirmed_events": self.pose_confirmed,
            "fallback_events": self.fallback_events,
            "ramp_detected": self.ramp_detector.cached_bbox is not None,
            "behavior_summary": {
                f"{k}_detections": v for k, v in self.behavior_counts.items()
            },
            "events": self.all_events,
        }

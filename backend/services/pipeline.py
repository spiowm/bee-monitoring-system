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
        """Process one frame. Returns annotated frame."""
        frame_start = time.time()

        ramp_bbox = self.ramp_detector.detect(frame)

        detections, filtered_boxes, filtered_kpts = self._detect(frame, ramp_bbox)

        tracked = self.tracker.update_with_detections(detections)
        self.active_bees = len(tracked.tracker_id) if tracked.tracker_id is not None else 0

        # Behavior analysis (every 15 frames)
        self.behavior_analyzer.update_history(frame_num, tracked, fps)
        if frame_num % 15 == 0:
            self.current_behaviors = self.behavior_analyzer.analyze(fps)
            self.behavior_counts = {k: 0 for k in self.behavior_counts}
            for b in self.current_behaviors.values():
                if b in self.behavior_counts:
                    self.behavior_counts[b] += 1

        mapped_kpts = self._match_keypoints(tracked, filtered_boxes, filtered_kpts)

        events = self.counter.update(
            frame_num, tracked, ramp_bbox, mapped_kpts,
            self.current_behaviors, fps,
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
            self.current_behaviors, self.counter, events, stats,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect(self, frame, ramp_bbox):
        """Run YOLO and filter detections inside ramp area."""
        results = self.bee_model(
            frame, verbose=False,
            conf=self.config.get("conf_threshold", 0.35),
        )

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
        """Match keypoints to tracked detections by nearest box center."""
        mapped = []
        if filtered_kpts and tracked.tracker_id is not None:
            for xyxy in tracked.xyxy:
                tcx, tcy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                best_dist = float("inf")
                best_kpt = None
                for j, fb in enumerate(filtered_boxes):
                    fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                    dist = (tcx - fcx) ** 2 + (tcy - fcy) ** 2
                    if dist < best_dist and dist < 1000:
                        best_dist = dist
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

    # ------------------------------------------------------------------
    # Stats / results (queried by the caller)
    # ------------------------------------------------------------------

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

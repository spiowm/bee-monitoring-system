from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import cv2
import numpy as np
import supervision as sv
import torch

_DEVICE = 0 if torch.cuda.is_available() else "cpu"
_HALF = isinstance(_DEVICE, int)

logger = logging.getLogger(__name__)

from services.ramp_detector import RampDetector
from services.track_history import TrackHistory
from services.counter import TrafficCounter
from services.behavior import BehaviorAnalyzer
from services.annotator import FrameAnnotator
from services.orientation import aligned, aligned_strict, get_orientation_vector
from services.temporal_motion import compute_crop_motion_gray

@dataclass
class FrameTimings:
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    behavior_ms: float = 0.0
    defense_ms: float = 0.0
    counting_ms: float = 0.0
    annotation_ms: float = 0.0
    total_ms: float = 0.0

@dataclass
class FrameContext:
    frame: np.ndarray
    frame_num: int
    fps: float

    detection_result: object = None  # pre-computed YOLO result (batch mode)
    ramp_bbox: tuple = None
    ramp_kpts: list = None
    detections: sv.Detections = None
    filtered_boxes: list = field(default_factory=list)
    filtered_kpts: list = field(default_factory=list)

    tracked_detections: sv.Detections = None
    mapped_kpts: list = field(default_factory=list)

    events: list = field(default_factory=list)
    rejected_events: list = field(default_factory=list)   # відсіяні pose-фільтром у цьому кадрі
    defense_clusters: list = field(default_factory=list)  # активні захисні кластери цього кадру
    annotated_frame: np.ndarray = None
    timings: FrameTimings = field(default_factory=FrameTimings)

class PipelineStage(ABC):
    @abstractmethod
    def process(self, ctx: FrameContext, state: dict):
        pass

class DetectionStage(PipelineStage):
    def __init__(self, bee_model, config, gt_entrance_zone=None):
        self.bee_model = bee_model
        self.config = config
        self.ramp_detector = RampDetector()
        self.gt_entrance_zone = gt_entrance_zone

    def process(self, ctx: FrameContext, state: dict):
        if self.gt_entrance_zone is not None:
            # Якщо ми в режимі Evaluation, використовуємо GT полігон як рампу
            zone = self.gt_entrance_zone
            x_min, y_min = np.min(zone, axis=0)
            x_max, y_max = np.max(zone, axis=0)
            ctx.ramp_bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
            # Для entrance_vec нам потрібні хоча б 2 верхні точки. Візьмемо top-left та top-right
            ctx.ramp_kpts = [zone[0].tolist(), zone[1].tolist(), zone[2].tolist(), zone[3].tolist()]
        else:
            ctx.ramp_bbox, ctx.ramp_kpts = self.ramp_detector.detect(ctx.frame)

        if ctx.detection_result is not None:
            results = [ctx.detection_result]
        else:
            meta = getattr(self.bee_model, "bee_meta", None) or {}
            imgsz = self.config.get("imgsz") or meta.get("imgsz") or 640
            results = self.bee_model(
                ctx.frame, verbose=False,
                conf=self.config.get("conf_threshold", 0.35),
                iou=self.config.get("iou_threshold", 0.8),
                max_det=self.config.get("max_detections", 1000),
                imgsz=imgsz,
                device=_DEVICE,
                half=bool(self.config.get("half_precision", False)) and isinstance(_DEVICE, int),
            )

        n_raw = 0
        if results and len(results[0].boxes) > 0:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            confs   = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()
            kpts    = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None
            n_raw = len(boxes)

            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            valid = np.isfinite(cx) & np.isfinite(cy)
            if ctx.ramp_bbox is not None:
                rx1, ry1, rx2, ry2 = ctx.ramp_bbox
                # Зменшуємо зону захоплення, щоб строго відповідати GT (тільки бджоли на льотку).
                # Даємо невеликий запас зверху (ry1 - 20), щоб зафіксувати сам момент перетину.
                mask = valid & (cx >= rx1 - 10) & (cx <= rx2 + 10) & (cy >= ry1 - 20) & (cy <= ry2 + 10)
            else:
                mask = valid

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
                if ctx.frame_num <= 3:
                    n_nan = int((~valid).sum())
                    cap = self.config.get("max_detections", 1000)
                    cap_hit = " [CAP HIT]" if n_raw >= cap else ""
                    cx_v, cy_v = cx[valid], cy[valid]
                    logger.info(
                        f"[Detection frame={ctx.frame_num}] yolo_raw={n_raw}{cap_hit} nan_boxes={n_nan} "
                        f"after_polygon={len(boxes_f)} "
                        f"conf_min={confs_f.min():.3f} conf_max={confs_f.max():.3f} conf_p50={np.median(confs_f):.3f} "
                        f"cx_range=[{cx_v.min():.0f},{cx_v.max():.0f}] cy_range=[{cy_v.min():.0f},{cy_v.max():.0f}] "
                        f"ramp_bbox={ctx.ramp_bbox}"
                    )
                return

        ctx.detections    = sv.Detections.empty()
        ctx.filtered_boxes = []
        ctx.filtered_kpts  = []
        if ctx.frame_num == 1:
            logger.info(
                f"[Detection frame=1] yolo_raw={n_raw} after_polygon=0 "
                f"(all filtered or no detections)"
            )


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
        n_tracked = (
            len(ctx.tracked_detections.tracker_id)
            if ctx.tracked_detections.tracker_id is not None else 0
        )
        state["active_bees"] = n_tracked
        if ctx.frame_num <= 3:
            n_det = len(ctx.detections.xyxy) if ctx.detections is not None else 0
            logger.info(f"[Tracking frame={ctx.frame_num}] detections_in={n_det} tracked_out={n_tracked}")

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
    def __init__(self, history: TrackHistory, config: dict | None = None):
        self.history = history
        self.id_mapping = {}
        config = config or {}
        self.stitch_max_dist = float(config.get("stitch_max_dist", 30.0))
        self.stitch_max_frames = int(config.get("stitch_max_frames", 30))
        
    def process(self, ctx: FrameContext, state: dict):
        prev_gray = state.get("prev_gray")
        # Single full-frame BGR→GRAY conversion instead of per-crop
        curr_gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)

        if ctx.tracked_detections.tracker_id is not None:
            active_mapped_ids = set()
            new_indices = []
            
            # 1. Remap IDs and stitch (Temporal Re-ID)
            for i in range(len(ctx.tracked_detections.tracker_id)):
                raw_id = int(ctx.tracked_detections.tracker_id[i])
                if raw_id in self.id_mapping:
                    mapped_id = self.id_mapping[raw_id]
                    ctx.tracked_detections.tracker_id[i] = mapped_id
                    active_mapped_ids.add(mapped_id)
                else:
                    new_indices.append((i, raw_id))
                    
            for i, raw_id in new_indices:
                xyxy = ctx.tracked_detections.xyxy[i]
                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                
                # Check for recently disappeared tracks
                candidate_id = self.history.find_stitching_candidate(
                    cx, cy, ctx.frame_num, active_mapped_ids, 
                    max_dist=self.stitch_max_dist, 
                    max_frames=self.stitch_max_frames
                )
                if candidate_id is not None:
                    self.id_mapping[raw_id] = candidate_id
                    ctx.tracked_detections.tracker_id[i] = candidate_id
                    active_mapped_ids.add(candidate_id)
                else:
                    active_mapped_ids.add(raw_id)

            # 2. Update history
            for i, track_id in enumerate(ctx.tracked_detections.tracker_id):
                xyxy = ctx.tracked_detections.xyxy[i]
                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2

                # TEM-like motion intensity per bee crop (optimized grayscale)
                motion = 0.0
                if prev_gray is not None:
                    try:
                        motion = compute_crop_motion_gray(prev_gray, curr_gray, tuple(xyxy))
                    except Exception:
                        pass

                self.history.update(int(track_id), cx, cy, ctx.frame_num, motion)
                
        self.history.prune_stale(ctx.frame_num)
        state["prev_gray"] = curr_gray


class BehaviorStage(PipelineStage):
    def __init__(self, analyzer: BehaviorAnalyzer):
        self.analyzer = analyzer

    def process(self, ctx: FrameContext, state: dict):
        if ctx.frame_num % 15 == 0:
            # Збираємо keypoints поточного кадру за track_id
            kpts_by_track: dict[int, object] = {}
            if ctx.tracked_detections is not None and ctx.tracked_detections.tracker_id is not None:
                for i, tid in enumerate(ctx.tracked_detections.tracker_id):
                    if i < len(ctx.mapped_kpts):
                        kpts_by_track[int(tid)] = ctx.mapped_kpts[i]

            current_behaviors = self.analyzer.analyze(
                state["history"],
                keypoints_by_track=kpts_by_track,
                ramp_kpts=ctx.ramp_kpts,
                fps=ctx.fps,
            )
            state["current_behaviors"] = current_behaviors

            counts = {k: 0 for k in state["behavior_counts"]}
            for b in current_behaviors.values():
                if b in counts:
                    counts[b] += 1
            state["behavior_counts"] = counts


class DefenseStage(PipelineStage):
    """
    Виявлення кластерів охоронців довкола кандидата-«крадія» (PLOS ONE 2025, Defense).
    Параметри: Rdef = factor * bee_length, Adef ±45°, Ndef ≥ 2, Tdef ≥ 1.0 s.
    Перезаписує state["current_behaviors"] для thief + defenders на "defense".
    """
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.radius_factor = float(config.get("defense_radius_factor", 2.0))
        self.angle_deg = float(config.get("defense_angle_deg", 45.0))
        self.min_defenders = int(config.get("defense_min_defenders", 2))
        self.duration_sec = float(config.get("defense_duration_sec", 0.3))
        # pos_key -> список frame_num де цей кластер був виявлений (sliding window)
        self.cluster_history: dict[tuple, list] = {}
        self._pos_grid = 30   # px — комірка позиційного грид
        self._window = int(config.get("defense_window_frames", 60))   # скільки кадрів назад аналізуємо
        self._min_appearances = int(config.get("defense_min_appearances", 3))  # мінімум появ у вікні
        # Defender має АКТИВНО рухатись (атакувати), а не стояти на місці.
        # Це відсіює хибні «кластери» зі стаціонарних fanning-бджіл у fan-відео,
        # зберігаючи реальний defense (де охоронці кидаються на крадія).
        self.defender_min_motion = float(config.get("defense_defender_min_motion", 0.0))
        self.motion_window = int(config.get("defense_motion_window", 15))

    def process(self, ctx: FrameContext, state: dict):
        ctx.defense_clusters = []
        td = ctx.tracked_detections
        if td is None or td.tracker_id is None or len(td.tracker_id) < self.min_defenders + 1:
            # Зберігаємо history незмінним — через нестабільність трекера кластери можуть
            # тимчасово зникати на 1-2 кадри навіть при наявності реального defense поведінки.
            return

        boxes = td.xyxy
        ids = [int(t) for t in td.tracker_id]
        n = len(ids)
        centers = np.array([
            [(boxes[i][0] + boxes[i][2]) / 2.0, (boxes[i][1] + boxes[i][3]) / 2.0]
            for i in range(n)
        ])
        bee_lengths = np.array([
            max(boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1])
            for i in range(n)
        ])

        body_vecs: list = []
        for i in range(n):
            kp = ctx.mapped_kpts[i] if i < len(ctx.mapped_kpts) else None
            body_vecs.append(get_orientation_vector(np.asarray(kp)) if kp is not None else None)

        # Виявлення кластерів у поточному кадрі (keyed by track ID для доступу до бджіл)
        # Бджоли з підтвердженим fanning не можуть бути defenders (різна поведінка)
        current_behaviors = state.get("current_behaviors") or {}
        history = state.get("history")
        active_clusters: dict[int, dict] = {}
        for i in range(n):
            radius = float(bee_lengths[i] * self.radius_factor)
            defenders: list[int] = []
            for j in range(n):
                if i == j or body_vecs[j] is None:
                    continue
                # Пропускаємо бджіл з підтвердженим fanning або washboarding
                if current_behaviors.get(ids[j]) in ("fanning", "washboarding"):
                    continue
                # Defender має активно рухатись (атакувати), інакше це стаціонарна
                # fanning/idle бджола, яка лише геометрично «дивиться» на сусіда.
                if self.defender_min_motion > 0 and history is not None:
                    ent = history.get(ids[j])
                    if ent is None or ent.recent_displacement(self.motion_window) < self.defender_min_motion:
                        continue
                vec_ij = centers[i] - centers[j]
                dist = float(np.linalg.norm(vec_ij))
                if dist > radius or dist < 1e-6:
                    continue
                vec_unit = vec_ij / dist
                if aligned_strict(body_vecs[j], vec_unit, self.angle_deg):
                    defenders.append(ids[j])
            if len(defenders) >= self.min_defenders:
                active_clusters[ids[i]] = {
                    "thief_track_id": ids[i],
                    "defender_track_ids": defenders,
                    "center": (float(centers[i][0]), float(centers[i][1])),
                    "radius": radius,
                }

        # Positional sliding-window history — стійкий до зміни track IDs ByteTrack
        g = self._pos_grid
        active_pos: dict[tuple, dict] = {}
        for cluster in active_clusters.values():
            cx, cy = cluster["center"]
            pkey = (int(cx / g), int(cy / g))
            active_pos[pkey] = cluster

        frame_num = ctx.frame_num
        cutoff = frame_num - self._window

        # Додаємо поточний кадр до history для активних позицій (з пошуком сусідніх клітинок)
        for pkey in active_pos:
            best_key = None
            for dk in [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                nkey = (pkey[0]+dk[0], pkey[1]+dk[1])
                if nkey in self.cluster_history:
                    best_key = nkey
                    break
            if best_key is not None and best_key != pkey:
                # Перемістити запис до нового ключа
                self.cluster_history[pkey] = self.cluster_history.pop(best_key)
            if pkey not in self.cluster_history:
                self.cluster_history[pkey] = []
            self.cluster_history[pkey].append(frame_num)

        # Прибираємо старі записи з вікна і пусті ключі
        for pkey in list(self.cluster_history.keys()):
            frames_list = [f for f in self.cluster_history[pkey] if f > cutoff]
            if frames_list:
                self.cluster_history[pkey] = frames_list
            else:
                del self.cluster_history[pkey]

        # Підтверджені: позиція, де в sliding window було >= min_appearances появ
        confirmed = []
        behaviors = state.get("current_behaviors") or {}
        for pkey, frames_list in self.cluster_history.items():
            if len(frames_list) < self._min_appearances:
                continue
            cluster = active_pos.get(pkey)
            if cluster is None:
                continue
            tid = cluster["thief_track_id"]
            confirmed.append({**cluster, "frame": frame_num})
            # Не перезаписуємо підтверджену fanning/washboarding поведінку
            if current_behaviors.get(tid) not in ("fanning", "washboarding"):
                behaviors[tid] = "defense"
            for did in cluster["defender_track_ids"]:
                if current_behaviors.get(did) not in ("fanning", "washboarding"):
                    behaviors[did] = "defense"

        if confirmed:
            state["current_behaviors"] = behaviors
            state.setdefault("defense_events", []).extend(confirmed)
            counts = state.get("behavior_counts", {})
            counts["defense"] = counts.get("defense", 0) + len(confirmed)
            state["behavior_counts"] = counts

        ctx.defense_clusters = confirmed


class CountingStage(PipelineStage):
    def __init__(self, counter: TrafficCounter):
        self.counter = counter
        
    def process(self, ctx: FrameContext, state: dict):
        events, rejected_events = self.counter.update(
            ctx.frame_num, ctx.tracked_detections, ctx.ramp_bbox, ctx.ramp_kpts, ctx.mapped_kpts,
            state["history"], state["current_behaviors"], ctx.fps,
        )
        ctx.events = events
        ctx.rejected_events = rejected_events
        for e in events:
            state["all_events"].append(e)
            if e["direction"] == "IN": state["total_in"] += 1
            if e["direction"] == "OUT": state["total_out"] += 1
            if e["method"] == "pose_confirmed": state["pose_confirmed"] += 1
            if e["method"] == "trajectory_fallback": state["fallback_events"] += 1
        for e in rejected_events:
            state.setdefault("rejected_events", []).append(e)
            state["pose_rejected"] = state.get("pose_rejected", 0) + 1


class AnnotationStage(PipelineStage):
    HIGHLIGHT_DURATION_SEC = 1.5

    def __init__(self, annotator: FrameAnnotator, counter: TrafficCounter):
        self.annotator = annotator
        self.counter = counter
        # track_id -> {"frames_remaining": int, "direction": "IN"|"OUT"}
        self.highlight_buffer: dict[int, dict] = {}

    def process(self, ctx: FrameContext, state: dict):
        # Декрементуємо попередні підсвічування і прибираємо вичерпані.
        for tid in list(self.highlight_buffer.keys()):
            self.highlight_buffer[tid]["frames_remaining"] -= 1
            if self.highlight_buffer[tid]["frames_remaining"] <= 0:
                del self.highlight_buffer[tid]

        # Записуємо нові підсвічування за події цього кадру.
        fps = ctx.fps or 30.0
        duration_frames = max(1, int(fps * self.HIGHLIGHT_DURATION_SEC))
        for e in ctx.events:
            self.highlight_buffer[int(e["track_id"])] = {
                "frames_remaining": duration_frames,
                "kind": e["direction"],  # "IN" або "OUT" — зарахована подія
            }
        for e in ctx.rejected_events:
            # Не перезаписуємо існуючий counted-highlight (counted має пріоритет)
            tid = int(e["track_id"])
            if tid not in self.highlight_buffer:
                self.highlight_buffer[tid] = {
                    "frames_remaining": duration_frames,
                    "kind": "REJECTED",  # відсіяна pose-фільтром
                }

        highlight_meta = {tid: meta["kind"] for tid, meta in self.highlight_buffer.items()}

        stats_state = {
            "total_in": state["total_in"],
            "total_out": state["total_out"],
            "fps": state["current_fps"],
        }
        ctx.annotated_frame = self.annotator.annotate(
            ctx.frame, ctx.tracked_detections, ctx.ramp_bbox, ctx.ramp_kpts, ctx.mapped_kpts,
            state["current_behaviors"], self.counter, ctx.events, stats_state, state["history"],
            highlight_meta=highlight_meta,
            defense_clusters=ctx.defense_clusters,
        )

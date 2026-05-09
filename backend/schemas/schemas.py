from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class ProcessConfig(BaseModel):
    tracker_name: str = "bytetrack"
    approach: str = "A"
    line_position: float = 0.0
    conf_threshold: float = 0.20
    iou_threshold: float = 0.8
    max_detections: int = 1000
    imgsz: Optional[int] = None      # None → fall back to model's training imgsz
    half_precision: bool = False
    batch_size: Optional[int] = None  # None → 2 (FP32) or 4 (FP16)
    kp_conf_threshold: float = 0.5
    track_tail_length: int = 30
    angle_threshold_deg: float = 60.0
    ramp_detect_interval: int = 30
    model_name: Optional[str] = None  # None = default bee_pose model
    # Behavior thresholds (PLOS ONE 2025 spec, Table 1)
    # Foraging: Vfor>100 px/s, Tfor>0.1s, Afor=±60° from entrance
    behavior_foraging_speed_min: float = 100.0
    behavior_foraging_angle_deg: float = 60.0
    # Fanning: Dfan<10 px, Tfan>1s, Afan=±90° from entrance
    behavior_fanning_max_displacement_px: float = 10.0
    behavior_fanning_duration_min: float = 1.0
    behavior_fanning_angle_deg: float = 90.0
    # Washboarding: Vwash<60 px/s, Twash>2s, ZCR>2Hz
    behavior_washboarding_speed_max: float = 60.0
    behavior_washboarding_duration_min: float = 2.0
    behavior_washboarding_zcr_min: float = 2.0
    # Defense: Rdef = factor * bee_length, Adef=±45°, Ndef≥2, Tdef≥1s
    defense_radius_factor: float = 2.0
    defense_angle_deg: float = 45.0
    defense_min_defenders: int = 2
    defense_duration_sec: float = 1.0
    # Legacy (kept for backward-compat with existing job docs)
    behavior_fanning_speed_max: float = 15.0
    behavior_guarding_speed_min: float = 15.0
    behavior_guarding_speed_max: float = 80.0
    behavior_guarding_spread_ratio: float = 1.5

class VizConfig(BaseModel):
    show_boxes: bool = True
    show_ids: bool = True
    show_confidence: bool = True
    show_keypoints: bool = True
    show_ramp: bool = True
    show_behaviors: bool = True
    show_counting_line: bool = True
    show_stats_overlay: bool = True
    show_tracks: bool = True
    show_orientation: bool = True
    show_recent_events: bool = True

class JobCreateResponse(BaseModel):
    job_id: str
    status: str

class ModelInfo(BaseModel):
    name: str
    arch: Optional[str] = None             # e.g. 'yolo11s-pose.pt'
    variant: Optional[str] = None          # 'n', 's', 'm', 'l', 'x'
    task: Optional[str] = None             # 'pose', 'detect', 'segment'
    imgsz: Optional[int] = None            # native training imgsz
    trained_with_half: bool = False

class LiveStats(BaseModel):
    current_frame: int = 0
    total_frames: int = 0
    bees_on_ramp: int = 0
    total_in: int = 0
    total_out: int = 0
    current_fps: float = 0.0
    tracker_name: str = ""
    approach: str = ""
    pose_confirmed: int = 0
    fallback_events: int = 0
    behavior_counts: Dict[str, int] = {}
    recent_events: List[Dict[str, Any]] = []
    timings: Optional[Dict[str, float]] = None
    reject_reasons: Optional[Dict[str, int]] = None
    angle_histogram: Optional[Dict[str, int]] = None


class TestPair(BaseModel):
    basename: str
    video_filename: str
    size_mb: float


class EvaluateJobRequest(BaseModel):
    filename: str           # e.g. "20230711a-fan/video.mp4" (legacy, now resolved from gt_basename)
    gt_basename: str        # e.g. "20230711a-fan" — subfolder name in GT_DATASET_PATH
    config: ProcessConfig
    viz_config: VizConfig


class DirectionMetrics(BaseModel):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mae: int
    gt_count: int
    pred_count: int


class EvaluationResult(BaseModel):
    in_metrics: DirectionMetrics
    out_metrics: DirectionMetrics
    accuracy: float
    gt_total_in: int
    gt_total_out: int
    pred_total_in: int
    pred_total_out: int
    frame_window: int
    gt_video_url: str
    gt_basename: str
    line_position: float
    fps: float
    video_resolution: List[int]
    gt_events: List[Dict[str, Any]] = []

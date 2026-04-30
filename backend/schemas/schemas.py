from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class ProcessConfig(BaseModel):
    tracker_name: str = "bytetrack"
    approach: str = "A"
    line_position: float = 0.5
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
    # Behavior thresholds (configurable, passed to HeuristicBehaviorStrategy)
    behavior_foraging_speed_min: float = 100.0
    behavior_fanning_speed_max: float = 15.0
    behavior_fanning_duration_min: float = 2.0
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

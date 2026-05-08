"""
Класифікація поведінки за академічною специфікацією
(Visual recognition of honeybee behavior patterns at the hive entrance, PLOS ONE 2025).

Стани класифікуються per-track. Defense обчислюється окремо у DefenseStage —
там потрібен multi-bee аналіз (кластер охоронців довкола кандидата-«крадія»).
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from services.track_history import TrackHistory
from services.orientation import (
    aligned,
    get_orientation_vector,
    vector_to_entrance,
)


class BehaviorStrategy(ABC):
    @abstractmethod
    def analyze(
        self,
        history: TrackHistory,
        keypoints_by_track: Dict[int, object],
        ramp_kpts,
        fps: float,
    ) -> Dict[int, Optional[str]]:
        ...


class HeuristicBehaviorStrategy(BehaviorStrategy):
    """
    Класифікує per-track поведінку згідно з PLOS ONE 2025 (Table 1):

      Foraging:    Vfor > 100 px/s, Tfor > 0.1 s, motion vec до льотка ±60°
      Fanning:     Dfan < 10 px, Tfan > 1 s, body vec до льотка ±90°
      Washboarding: Vwash < 60 px/s, Twash > 2 s, ZCR прискорення > 2 Hz

    Defense НЕ класифікується тут — це робота `DefenseStage`.
    """

    MIN_HISTORY_FRAMES = 15  # потрібно мінімум для надійних метрик

    def __init__(self, config: dict | None = None):
        config = config or {}
        # Foraging
        self.foraging_speed_min = float(config.get("behavior_foraging_speed_min", 100.0))
        self.foraging_angle_deg = float(config.get("behavior_foraging_angle_deg", 60.0))
        # Fanning
        self.fanning_max_displacement_px = float(
            config.get("behavior_fanning_max_displacement_px", 10.0)
        )
        self.fanning_duration_min = float(config.get("behavior_fanning_duration_min", 1.0))
        self.fanning_angle_deg = float(config.get("behavior_fanning_angle_deg", 90.0))
        # Washboarding
        self.washboarding_speed_max = float(config.get("behavior_washboarding_speed_max", 60.0))
        self.washboarding_duration_min = float(
            config.get("behavior_washboarding_duration_min", 2.0)
        )
        self.washboarding_zcr_min = float(config.get("behavior_washboarding_zcr_min", 2.0))

    def analyze(
        self,
        history: TrackHistory,
        keypoints_by_track: Dict[int, object],
        ramp_kpts,
        fps: float = 30.0,
    ) -> Dict[int, Optional[str]]:
        behaviors: Dict[int, Optional[str]] = {}

        for track_id, entry in history.all_entries().items():
            if len(entry.positions) < self.MIN_HISTORY_FRAMES:
                behaviors[track_id] = None
                continue

            metrics = entry.compute_metrics(fps)
            avg_speed = metrics["avg_speed"]
            duration_sec = len(entry.positions) / fps
            max_disp = metrics["max_displacement"]
            zcr = metrics["zero_cross_rate"]

            # Поточна позиція і вектори
            cx, cy = entry.positions[-1]
            entrance_vec = vector_to_entrance((cx, cy), ramp_kpts)
            motion_vec_raw = metrics["track_dir_vec"]
            motion_norm = float(np.linalg.norm(motion_vec_raw))
            motion_unit = (
                np.array(motion_vec_raw) / motion_norm if motion_norm > 1e-6 else None
            )

            kp = keypoints_by_track.get(track_id) if keypoints_by_track else None
            body_vec = get_orientation_vector(np.asarray(kp)) if kp is not None else None

            behavior: Optional[str] = None

            # 1. Foraging: швидкість > Vfor + рух у напрямі льотка ±Afor
            if avg_speed > self.foraging_speed_min and aligned(
                motion_unit, entrance_vec, self.foraging_angle_deg
            ):
                behavior = "foraging"

            # 2. Fanning: майже нерухома, тривала, орієнтована до льотка ±Afan
            elif (
                max_disp < self.fanning_max_displacement_px
                and duration_sec > self.fanning_duration_min
                and aligned(body_vec, entrance_vec, self.fanning_angle_deg)
            ):
                behavior = "fanning"

            # 3. Washboarding: середня швидкість + тривалість + періодичність прискорення
            elif (
                avg_speed < self.washboarding_speed_max
                and duration_sec > self.washboarding_duration_min
                and zcr > self.washboarding_zcr_min
            ):
                behavior = "washboarding"

            entry.behavior = behavior
            behaviors[track_id] = behavior

        return behaviors


class BehaviorAnalyzer:
    """Контекст для обраної BehaviorStrategy."""

    def __init__(self, config: dict | None = None, strategy: BehaviorStrategy | None = None):
        self.strategy = strategy or HeuristicBehaviorStrategy(config)

    def analyze(
        self,
        history: TrackHistory,
        keypoints_by_track: Dict[int, object] | None = None,
        ramp_kpts=None,
        fps: float = 30.0,
    ) -> Dict[int, Optional[str]]:
        return self.strategy.analyze(history, keypoints_by_track or {}, ramp_kpts, fps)

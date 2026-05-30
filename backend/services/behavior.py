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
    aligned_strict,
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

      Foraging:     Vfor > 100 px/s, motion vec до льотка ±60°
      Fanning:      ZCR > 15 Hz (аліасинг крилобиття при 50 fps), Tfan > 1 s,
                    body vec до льотка ±90° (strict), avg_motion > 0.05
      Washboarding: ZCR 3–15 Hz (ритмічні рухи), 5 < V < 60, Twash > 2 s,
                    displacement < 40 px, body ≈ до льотка ±120°
      Unknown:      достатньо історії, але жоден патерн не підходить

    Defense НЕ класифікується тут — це робота `DefenseStage`.
    """

    MIN_HISTORY_FRAMES = 15  # потрібно мінімум для надійних метрик

    def __init__(self, config: dict | None = None):
        config = config or {}
        # Foraging
        self.foraging_speed_min = float(config.get("behavior_foraging_speed_min", 100.0))
        self.foraging_angle_deg = float(config.get("behavior_foraging_angle_deg", 60.0))
        # Foraging має РЕАЛЬНО переміщатись (net displacement), а не лише мати
        # роздуту jitter-ом avg_speed. Інакше краде стаціонарних fanning-бджіл.
        self.foraging_min_disp = float(config.get("behavior_foraging_min_disp", 0.0))
        # Fanning — головний дискримінатор: СТАЦІОНАРНІСТЬ (low max_disp).
        # ZCR насичена (~38 Гц у всіх) і нічого не розділяє; залишена як дешевий гейт.
        self.fanning_zcr_min = float(config.get("behavior_fanning_zcr_min", 15.0))
        self.fanning_duration_min = float(config.get("behavior_fanning_duration_min", 1.0))
        self.fanning_angle_deg = float(config.get("behavior_fanning_angle_deg", 90.0))
        self.fanning_motion_min = float(config.get("behavior_fanning_motion_min", 0.05))
        # Якщо >0 — fanning вимагає max_disp < цього (бджола стоїть на місці).
        self.fanning_max_disp = float(config.get("behavior_fanning_max_disp", 0.0))
        # Чи вимагати keypoints для body-alignment. False → м'яко (None дозволено),
        # бо ~50% fanning-бджіл не мають keypoints і хибно йшли в unknown.
        self.fanning_require_body = bool(config.get("behavior_fanning_require_body", True))
        # Якщо True — стаціонарна fanning має пріоритет над foraging. Інакше
        # jitter-роздутий avg_speed краде стаціонарних fanning-бджіл у foraging.
        # Безпечно лише разом з fanning_max_disp (кросери мають велике зміщення).
        self.fanning_priority = bool(config.get("behavior_fanning_priority", False))
        # Washboarding — ZCR нижчий ніж у fanning
        self.washboarding_speed_min = float(config.get("behavior_washboarding_speed_min", 5.0))
        self.washboarding_speed_max = float(config.get("behavior_washboarding_speed_max", 60.0))
        self.washboarding_max_disp = float(config.get("behavior_washboarding_max_disp", 40.0))
        self.washboarding_duration_min = float(
            config.get("behavior_washboarding_duration_min", 2.0)
        )
        self.washboarding_zcr_min = float(config.get("behavior_washboarding_zcr_min", 3.0))
        self.washboarding_body_angle_deg = float(
            config.get("behavior_washboarding_body_angle_deg", 120.0)
        )
        # Debug: коли увімкнено, у self.debug_records збираються per-track фічі
        # для аналізу розподілів (TP vs FP). Не впливає на роботу коли вимкнено.
        self.debug_enabled = False
        self.debug_frame = 0
        self.debug_records: list = []

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
            avg_motion = metrics.get("avg_motion_intensity", 0.0)

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

            # Передобчислені гейти
            forage_disp_ok = (
                max_disp > self.foraging_min_disp if self.foraging_min_disp > 0 else True
            )
            fan_disp_ok = (
                max_disp < self.fanning_max_disp if self.fanning_max_disp > 0 else True
            )
            _body_check = aligned_strict if self.fanning_require_body else aligned
            fan_body_ok = _body_check(body_vec, entrance_vec, self.fanning_angle_deg)

            # Передобчислені кандидати поведінки
            is_foraging = (
                avg_speed > self.foraging_speed_min
                and forage_disp_ok
                and aligned(motion_unit, entrance_vec, self.foraging_angle_deg)
            )
            # Fanning: СТАЦІОНАРНА (low max_disp) + тривале + є рух крил;
            # body-alignment м'який бонус (keypoints часто відсутні)
            is_fanning = (
                zcr > self.fanning_zcr_min
                and duration_sec > self.fanning_duration_min
                and avg_motion >= self.fanning_motion_min
                and fan_disp_ok
                and fan_body_ok
            )
            # Washboarding: ритмічні рухи, ZCR нижча ніж у fanning, обмежений рух
            is_washboarding = (
                avg_speed > self.washboarding_speed_min
                and avg_speed < self.washboarding_speed_max
                and max_disp < self.washboarding_max_disp
                and duration_sec > self.washboarding_duration_min
                and zcr > self.washboarding_zcr_min
                and zcr <= self.fanning_zcr_min
                and aligned_strict(body_vec, entrance_vec, self.washboarding_body_angle_deg)
            )

            # Пріоритет: стаціонарна fanning може випереджати foraging (jitter-крадіжка)
            if is_fanning and self.fanning_priority:
                behavior = "fanning"
            elif is_foraging:
                behavior = "foraging"
            elif is_fanning:
                behavior = "fanning"
            elif is_washboarding:
                behavior = "washboarding"
            else:
                behavior = "unknown"

            entry.behavior = behavior
            behaviors[track_id] = behavior

            if self.debug_enabled:
                self.debug_records.append({
                    "frame": self.debug_frame,
                    "track_id": int(track_id),
                    "cx": float(cx), "cy": float(cy),
                    "zcr": float(zcr),
                    "avg_motion": float(avg_motion),
                    "max_disp": float(max_disp),
                    "avg_speed": float(avg_speed),
                    "duration": float(duration_sec),
                    "n_pos": len(entry.positions),
                    "has_body": body_vec is not None,
                    "body_aligned": bool(aligned_strict(body_vec, entrance_vec, self.fanning_angle_deg)),
                    "behavior": behavior,
                })

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

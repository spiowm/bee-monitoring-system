import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackEntry:
    positions: list = field(default_factory=list)   # [(cx, cy), ...]
    frames: list = field(default_factory=list)       # [frame_num, ...]
    motion_intensity: list = field(default_factory=list)  # TEM-like motion score per frame
    behavior: Optional[str] = None

    # Max positions to keep (longest consumer is BehaviorAnalyzer: 150)
    _MAX_LEN: int = 150

    def append(self, cx: float, cy: float, frame_num: int, motion: float = 0.0):
        self.positions.append((cx, cy))
        self.frames.append(frame_num)
        self.motion_intensity.append(motion)
        if len(self.positions) > self._MAX_LEN:
            self.positions.pop(0)
            self.frames.pop(0)
            self.motion_intensity.pop(0)

    def last_n_positions(self, n: int) -> list:
        return self.positions[-n:]

    def recent_displacement(self, k: int = 15) -> float:
        """Чисте зміщення центру за останні k кадрів (px). Дешевий маркер
        «бджола активно переміщується» (атакує) vs «стоїть на місці» (fanning/idle)."""
        if len(self.positions) < 2:
            return 0.0
        a = self.positions[-1]
        b = self.positions[-min(k, len(self.positions))]
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    def last_frame(self) -> Optional[int]:
        return self.frames[-1] if self.frames else None

    def compute_metrics(self, fps: float = 30.0) -> dict:
        """
        Computes various velocity and spatial metrics for the track memory window.
        Returns useful heuristics for TrafficCounter and BehaviorAnalyzer.
        """
        if len(self.positions) < 2:
            return {
                "avg_speed": 0.0,
                "current_speed": 0.0,
                "spread_x": 0.0,
                "spread_y": 0.0,
                "track_dir_vec": (0.0, 0.0),
                "instant_dir_vec": (0.0, 0.0),
                "max_displacement": 0.0,
                "zero_cross_rate": 0.0,
            }
        
        pos_np = np.array(self.positions)
        diffs = np.diff(pos_np, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total_dist = float(np.sum(distances))
        
        duration_sec = len(self.positions) / fps
        avg_speed = total_dist / duration_sec if duration_sec > 0 else 0.0
        
        last_dist = float(distances[-1])
        frames_diff = self.frames[-1] - self.frames[-2]
        current_speed = last_dist / (frames_diff / fps) if frames_diff > 0 else 0.0
        
        spread_x = float(np.max(pos_np[:, 0]) - np.min(pos_np[:, 0]))
        spread_y = float(np.max(pos_np[:, 1]) - np.min(pos_np[:, 1]))

        # Vector from first seen in window to last
        track_dir_vec = float(pos_np[-1, 0] - pos_np[0, 0]), float(pos_np[-1, 1] - pos_np[0, 1])

        # Instant vector (last two frames)
        instant_dir = float(pos_np[-1, 0] - pos_np[-2, 0]), float(pos_np[-1, 1] - pos_np[-2, 1])

        # EMA Direction Vector
        ema_dir_vec = (0.0, 0.0)
        if len(diffs) > 0:
            alpha = 0.3
            ema_x, ema_y = float(diffs[0, 0]), float(diffs[0, 1])
            for i in range(1, len(diffs)):
                ema_x = alpha * float(diffs[i, 0]) + (1 - alpha) * ema_x
                ema_y = alpha * float(diffs[i, 1]) + (1 - alpha) * ema_y
            ema_dir_vec = (ema_x, ema_y)

        # Maximum displacement from the first window position (для Fanning Dfan)
        first = pos_np[0]
        max_displacement = float(np.max(np.linalg.norm(pos_np - first, axis=1)))

        # Zero-cross rate of acceleration (для Washboarding ZCR > 2 Hz)
        zero_cross_rate = 0.0
        if len(pos_np) >= 4 and duration_sec > 0:
            velocities = diffs * fps
            speeds = np.linalg.norm(velocities, axis=1)
            accelerations = np.diff(speeds) * fps
            if accelerations.size >= 2:
                signs = np.sign(accelerations)
                signs[signs == 0] = 1.0
                crossings = int(np.sum(signs[:-1] != signs[1:]))
                zero_cross_rate = crossings / duration_sec

        # TEM-like motion intensity stats
        mi = np.array(self.motion_intensity) if self.motion_intensity else np.array([0.0])
        avg_motion_intensity = float(np.mean(mi))
        motion_intensity_std = float(np.std(mi))

        return {
            "avg_speed": avg_speed,
            "current_speed": current_speed,
            "spread_x": spread_x,
            "spread_y": spread_y,
            "track_dir_vec": track_dir_vec,
            "instant_dir_vec": instant_dir,
            "ema_dir_vec": ema_dir_vec,
            "max_displacement": max_displacement,
            "zero_cross_rate": zero_cross_rate,
            "avg_motion_intensity": avg_motion_intensity,
            "motion_intensity_std": motion_intensity_std,
        }


class TrackHistory:
    """
    Single source of truth for per-track position history.
    Replaces parallel dicts in TrafficCounter, BehaviorAnalyzer, FrameAnnotator.
    """

    def __init__(self):
        self._tracks: dict[int, TrackEntry] = {}

    def update(self, track_id: int, cx: float, cy: float, frame_num: int, motion: float = 0.0):
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackEntry()
        self._tracks[track_id].append(cx, cy, frame_num, motion)

    def prune_stale(self, current_frame: int, max_age: int = 60):
        """Remove tracks not seen for max_age frames."""
        stale = [
            tid for tid, entry in self._tracks.items()
            if entry.last_frame() is not None and (current_frame - entry.last_frame()) > max_age
        ]
        for tid in stale:
            del self._tracks[tid]

    def get(self, track_id: int) -> Optional[TrackEntry]:
        return self._tracks.get(track_id)

    def all_entries(self) -> dict[int, TrackEntry]:
        return self._tracks

    def active_ids(self, current_ids: set) -> set:
        return set(self._tracks.keys()) & current_ids

    def find_stitching_candidate(self, cx: float, cy: float, current_frame: int, active_mapped_ids: set, max_dist: float = 30.0, max_frames: int = 30) -> Optional[int]:
        """
        Знаходить кандидата для зшивання треків (Temporal Re-ID).
        Повертає track_id, якщо знайдено недавно зниклий трек близько до (cx, cy).
        """
        best_id = None
        best_dist = max_dist
        
        for tid, entry in self._tracks.items():
            if tid in active_mapped_ids:
                continue
                
            last_frame = entry.last_frame()
            if last_frame is None:
                continue
                
            frames_missed = current_frame - last_frame
            if 0 < frames_missed <= max_frames:
                last_cx, last_cy = entry.positions[-1]
                dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
                    
        return best_id

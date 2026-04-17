import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackEntry:
    positions: list = field(default_factory=list)   # [(cx, cy), ...]
    frames: list = field(default_factory=list)       # [frame_num, ...]
    behavior: Optional[str] = None

    # Max positions to keep (longest consumer is BehaviorAnalyzer: 150)
    _MAX_LEN: int = 150

    def append(self, cx: float, cy: float, frame_num: int):
        self.positions.append((cx, cy))
        self.frames.append(frame_num)
        if len(self.positions) > self._MAX_LEN:
            self.positions.pop(0)
            self.frames.pop(0)

    def last_n_positions(self, n: int) -> list:
        return self.positions[-n:]

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
                "instant_dir_vec": (0.0, 0.0)
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

        return {
            "avg_speed": avg_speed,
            "current_speed": current_speed,
            "spread_x": spread_x,
            "spread_y": spread_y,
            "track_dir_vec": track_dir_vec,
            "instant_dir_vec": instant_dir
        }


class TrackHistory:
    """
    Single source of truth for per-track position history.
    Replaces parallel dicts in TrafficCounter, BehaviorAnalyzer, FrameAnnotator.
    """

    def __init__(self):
        self._tracks: dict[int, TrackEntry] = {}

    def update(self, track_id: int, cx: float, cy: float, frame_num: int):
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackEntry()
        self._tracks[track_id].append(cx, cy, frame_num)

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

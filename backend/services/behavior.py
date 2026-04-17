from services.track_history import TrackHistory
import numpy as np


class BehaviorAnalyzer:
    """
    Classifies per-track behavior using shared TrackHistory and configurable heuristics.
    Results cached in TrackEntry.behavior.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        # Configure thresholds with defaults
        self.foraging_speed_min = self.config.get("behavior_foraging_speed_min", 100.0)
        self.fanning_speed_max = self.config.get("behavior_fanning_speed_max", 15.0)
        self.fanning_duration_min = self.config.get("behavior_fanning_duration_min", 2.0)
        self.guarding_speed_min = self.config.get("behavior_guarding_speed_min", 15.0)
        self.guarding_speed_max = self.config.get("behavior_guarding_speed_max", 80.0)
        self.guarding_spread_ratio = self.config.get("behavior_guarding_spread_ratio", 1.5)

    def analyze(self, history: TrackHistory, fps: float = 30.0) -> dict:
        """
        Returns dict: track_id → 'foraging'|'fanning'|'guarding'|'washboarding'|None.
        Call every 15 frames.
        """
        behaviors = {}

        for track_id, entry in history.all_entries().items():
            positions = entry.positions
            if len(positions) < 15:
                behaviors[track_id] = None
                continue

            pos_np = np.array(positions)
            diffs = np.diff(pos_np, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_dist = np.sum(distances)
            duration_sec = len(positions) / fps
            avg_speed = total_dist / duration_sec

            if avg_speed > self.foraging_speed_min:
                behavior = "foraging"
            elif avg_speed < self.fanning_speed_max and duration_sec > self.fanning_duration_min:
                behavior = "fanning"
            elif self.guarding_speed_min <= avg_speed <= self.guarding_speed_max:
                spread_x = np.max(pos_np[:, 0]) - np.min(pos_np[:, 0])
                spread_y = np.max(pos_np[:, 1]) - np.min(pos_np[:, 1])
                behavior = "guarding" if spread_x > spread_y * self.guarding_spread_ratio else "washboarding"
            else:
                behavior = "washboarding"

            entry.behavior = behavior
            behaviors[track_id] = behavior

        return behaviors

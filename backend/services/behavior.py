from services.track_history import TrackHistory
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BehaviorStrategy(ABC):
    @abstractmethod
    def analyze(self, history: TrackHistory, fps: float) -> Dict[int, Optional[str]]:
        pass

class HeuristicBehaviorStrategy(BehaviorStrategy):
    """
    Classifies per-track behavior using empirical thresholds based on speed and spread.
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

    def analyze(self, history: TrackHistory, fps: float = 30.0) -> Dict[int, Optional[str]]:
        behaviors = {}

        for track_id, entry in history.all_entries().items():
            if len(entry.positions) < 15:
                behaviors[track_id] = None
                continue

            metrics = entry.compute_metrics(fps)
            avg_speed = metrics["avg_speed"]
            duration_sec = len(entry.positions) / fps

            if avg_speed > self.foraging_speed_min:
                behavior = "foraging"
            elif avg_speed < self.fanning_speed_max and duration_sec > self.fanning_duration_min:
                behavior = "fanning"
            elif self.guarding_speed_min <= avg_speed <= self.guarding_speed_max:
                spread_x = metrics["spread_x"]
                spread_y = metrics["spread_y"]
                behavior = "guarding" if spread_x > spread_y * self.guarding_spread_ratio else "washboarding"
            else:
                behavior = "washboarding"

            entry.behavior = behavior
            behaviors[track_id] = behavior

        return behaviors

class BehaviorAnalyzer:
    """
    Context class that executes a chosen BehaviorStrategy.
    """
    def __init__(self, config: dict = None, strategy: BehaviorStrategy = None):
        self.strategy = strategy or HeuristicBehaviorStrategy(config)

    def analyze(self, history: TrackHistory, fps: float = 30.0) -> Dict[int, Optional[str]]:
        return self.strategy.analyze(history, fps)

import pytest
import numpy as np
from unittest.mock import MagicMock
import supervision as sv

from services.counter import TrafficCounter


def _make_detections(boxes, track_ids):
    """Helper: create sv.Detections with tracker_id."""
    det = sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.ones(len(boxes)),
        class_id=np.zeros(len(boxes), dtype=int),
    )
    det.tracker_id = np.array(track_ids)
    return det


RAMP = [0.0, 0.0, 200.0, 200.0]  # rx1, ry1, rx2, ry2; line_y = 100 at position=0.5


class TestTrafficCounterApproachA:
    def test_counts_upward_crossing_as_in(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        # Move bee from y=120 (below line) to y=80 (above line) → IN
        det1 = _make_detections([[90, 110, 110, 130]], [1])  # cy=120
        det2 = _make_detections([[90, 70, 110, 90]], [1])    # cy=80
        counter.update(1, det1, RAMP, [], {}, 30.0)
        events = counter.update(2, det2, RAMP, [], {}, 30.0)
        assert len(events) == 1
        assert events[0]["direction"] == "IN"

    def test_counts_downward_crossing_as_out(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        det1 = _make_detections([[90, 70, 110, 90]], [1])    # cy=80 (above)
        det2 = _make_detections([[90, 110, 110, 130]], [1])  # cy=120 (below)
        counter.update(1, det1, RAMP, [], {}, 30.0)
        events = counter.update(2, det2, RAMP, [], {}, 30.0)
        assert len(events) == 1
        assert events[0]["direction"] == "OUT"

    def test_no_crossing_no_events(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        det1 = _make_detections([[90, 10, 110, 30]], [1])   # cy=20 (above)
        det2 = _make_detections([[90, 20, 110, 40]], [1])   # cy=30 (still above)
        counter.update(1, det1, RAMP, [], {}, 30.0)
        events = counter.update(2, det2, RAMP, [], {}, 30.0)
        assert events == []

    def test_debounce_prevents_double_count(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        det_above = _make_detections([[90, 70, 110, 90]], [1])
        det_below = _make_detections([[90, 110, 110, 130]], [1])
        # First crossing
        counter.update(1, det_above, RAMP, [], {}, 30.0)
        events1 = counter.update(2, det_below, RAMP, [], {}, 30.0)
        # Immediate re-cross (within debounce window)
        events2 = counter.update(3, det_above, RAMP, [], {}, 30.0)
        assert len(events1) == 1
        assert events2 == []

    def test_no_ramp_no_events(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        det = _make_detections([[90, 70, 110, 90]], [1])
        events = counter.update(1, det, None, [], {}, 30.0)
        assert events == []

    def test_empty_detections_no_events(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        events = counter.update(1, sv.Detections.empty(), RAMP, [], {}, 30.0)
        assert events == []

    def test_multiple_tracks_independent(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        # Two bees cross in opposite directions simultaneously
        det1 = _make_detections([[0, 70, 20, 90], [100, 110, 120, 130]], [1, 2])
        det2 = _make_detections([[0, 110, 20, 130], [100, 70, 120, 90]], [1, 2])
        counter.update(1, det1, RAMP, [], {}, 30.0)
        events = counter.update(2, det2, RAMP, [], {}, 30.0)
        directions = {e["track_id"]: e["direction"] for e in events}
        assert directions[1] == "OUT"
        assert directions[2] == "IN"

    def test_event_contains_required_fields(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        det1 = _make_detections([[90, 110, 110, 130]], [1])
        det2 = _make_detections([[90, 70, 110, 90]], [1])
        counter.update(1, det1, RAMP, [], {}, 30.0)
        events = counter.update(2, det2, RAMP, [], {}, 30.0)
        e = events[0]
        for key in ("frame", "timestamp_sec", "track_id", "direction", "method", "speed_px_per_sec"):
            assert key in e


class TestTrafficCounterApproachB:
    def test_pose_confirmed_when_aligned(self):
        counter = TrafficCounter(line_position=0.5, approach="B", angle_threshold_deg=60.0)
        det1 = _make_detections([[90, 110, 110, 130]], [1])
        det2 = _make_detections([[90, 70, 110, 90]], [1])
        # Bee moving up (IN), head above tail → aligned
        kp = [np.array([[100.0, 60.0], [100.0, 80.0]])]  # head above tail
        counter.update(1, det1, RAMP, [], {}, 30.0)
        events = counter.update(2, det2, RAMP, kp, {}, 30.0)
        assert len(events) == 1
        assert events[0]["method"] == "pose_confirmed"

    def test_rejected_when_misaligned(self):
        counter = TrafficCounter(line_position=0.5, approach="B", angle_threshold_deg=30.0)
        det1 = _make_detections([[90, 110, 110, 130]], [1])
        det2 = _make_detections([[90, 70, 110, 90]], [1])
        # Moving up but head below tail → misaligned
        kp = [np.array([[100.0, 80.0], [100.0, 60.0]])]
        counter.update(1, det1, RAMP, [], {}, 30.0)
        events = counter.update(2, det2, RAMP, kp, {}, 30.0)
        assert events == []

    def test_fallback_when_no_keypoints(self):
        counter = TrafficCounter(line_position=0.5, approach="B")
        det1 = _make_detections([[90, 110, 110, 130]], [1])
        det2 = _make_detections([[90, 70, 110, 90]], [1])
        counter.update(1, det1, RAMP, [], {}, 30.0)
        events = counter.update(2, det2, RAMP, [], {}, 30.0)
        assert len(events) == 1
        assert events[0]["method"] == "trajectory_fallback"

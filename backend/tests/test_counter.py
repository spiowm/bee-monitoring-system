import pytest
import numpy as np
from unittest.mock import MagicMock
import supervision as sv

from services.counter import TrafficCounter
from services.track_history import TrackHistory


def _make_detections(boxes, track_ids):
    """Helper: create sv.Detections with tracker_id."""
    det = sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.ones(len(boxes)),
        class_id=np.zeros(len(boxes), dtype=int),
    )
    det.tracker_id = np.array(track_ids)
    return det


def _update_history(history: TrackHistory, frame_num: int, dets: sv.Detections):
    if dets.tracker_id is not None:
        for i, track_id in enumerate(dets.tracker_id):
            xyxy = dets.xyxy[i]
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            history.update(track_id, cx, cy, frame_num)


RAMP = [0.0, 0.0, 200.0, 200.0]  # rx1, ry1, rx2, ry2; line_y = 100 at position=0.5


class TestTrafficCounterApproachA:
    def test_counts_upward_crossing_as_in(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        history = TrackHistory()
        # Move bee from y=120 (below line) to y=80 (above line) → IN
        det1 = _make_detections([[90, 110, 110, 130]], [1])  # cy=120
        det2 = _make_detections([[90, 70, 110, 90]], [1])    # cy=80
        
        _update_history(history, 1, det1)
        counter.update(1, det1, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det2)
        events = counter.update(2, det2, RAMP, [], history, {}, 30.0)
        assert len(events) == 1
        assert events[0]["direction"] == "IN"

    def test_counts_downward_crossing_as_out(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        history = TrackHistory()
        det1 = _make_detections([[90, 70, 110, 90]], [1])    # cy=80 (above)
        det2 = _make_detections([[90, 110, 110, 130]], [1])  # cy=120 (below)
        
        _update_history(history, 1, det1)
        counter.update(1, det1, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det2)
        events = counter.update(2, det2, RAMP, [], history, {}, 30.0)
        assert len(events) == 1
        assert events[0]["direction"] == "OUT"

    def test_no_crossing_no_events(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        history = TrackHistory()
        det1 = _make_detections([[90, 10, 110, 30]], [1])   # cy=20 (above)
        det2 = _make_detections([[90, 20, 110, 40]], [1])   # cy=30 (still above)
        
        _update_history(history, 1, det1)
        counter.update(1, det1, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det2)
        events = counter.update(2, det2, RAMP, [], history, {}, 30.0)
        assert events == []

    def test_debounce_prevents_double_count(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        history = TrackHistory()
        det_above = _make_detections([[90, 70, 110, 90]], [1])
        det_below = _make_detections([[90, 110, 110, 130]], [1])
        
        _update_history(history, 1, det_above)
        counter.update(1, det_above, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det_below)
        events1 = counter.update(2, det_below, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 3, det_above)
        events2 = counter.update(3, det_above, RAMP, [], history, {}, 30.0)
        assert len(events1) == 1
        assert events2 == []

    def test_no_ramp_no_events(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        history = TrackHistory()
        det = _make_detections([[90, 70, 110, 90]], [1])
        
        _update_history(history, 1, det)
        events = counter.update(1, det, None, [], history, {}, 30.0)
        assert events == []

    def test_empty_detections_no_events(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        history = TrackHistory()
        det = sv.Detections.empty()
        
        _update_history(history, 1, det)
        events = counter.update(1, det, RAMP, [], history, {}, 30.0)
        assert events == []

    def test_multiple_tracks_independent(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        history = TrackHistory()
        det1 = _make_detections([[0, 70, 20, 90], [100, 110, 120, 130]], [1, 2])
        det2 = _make_detections([[0, 110, 20, 130], [100, 70, 120, 90]], [1, 2])
        
        _update_history(history, 1, det1)
        counter.update(1, det1, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det2)
        events = counter.update(2, det2, RAMP, [], history, {}, 30.0)
        directions = {e["track_id"]: e["direction"] for e in events}
        assert directions[1] == "OUT"
        assert directions[2] == "IN"

    def test_event_contains_required_fields(self):
        counter = TrafficCounter(line_position=0.5, approach="A")
        history = TrackHistory()
        det1 = _make_detections([[90, 110, 110, 130]], [1])
        det2 = _make_detections([[90, 70, 110, 90]], [1])
        
        _update_history(history, 1, det1)
        counter.update(1, det1, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det2)
        events = counter.update(2, det2, RAMP, [], history, {}, 30.0)
        e = events[0]
        for key in ("frame", "timestamp_sec", "track_id", "direction", "method", "speed_px_per_sec"):
            assert key in e


class TestTrafficCounterApproachB:
    def test_pose_confirmed_when_aligned(self):
        counter = TrafficCounter(line_position=0.5, approach="B", angle_threshold_deg=60.0)
        history = TrackHistory()
        det1 = _make_detections([[90, 110, 110, 130]], [1])
        det2 = _make_detections([[90, 70, 110, 90]], [1])
        
        _update_history(history, 1, det1)
        counter.update(1, det1, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det2)
        kp = [np.array([[100.0, 60.0], [100.0, 80.0]])]  # head above tail
        events = counter.update(2, det2, RAMP, kp, history, {}, 30.0)
        assert len(events) == 1
        assert events[0]["method"] == "pose_confirmed"

    def test_rejected_when_misaligned(self):
        counter = TrafficCounter(line_position=0.5, approach="B", angle_threshold_deg=30.0)
        history = TrackHistory()
        det1 = _make_detections([[90, 110, 110, 130]], [1])
        det2 = _make_detections([[90, 70, 110, 90]], [1])
        
        _update_history(history, 1, det1)
        counter.update(1, det1, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det2)
        kp = [np.array([[100.0, 80.0], [100.0, 60.0]])]
        events = counter.update(2, det2, RAMP, kp, history, {}, 30.0)
        assert events == []

    def test_fallback_when_no_keypoints(self):
        counter = TrafficCounter(line_position=0.5, approach="B")
        history = TrackHistory()
        det1 = _make_detections([[90, 110, 110, 130]], [1])
        det2 = _make_detections([[90, 70, 110, 90]], [1])
        
        _update_history(history, 1, det1)
        counter.update(1, det1, RAMP, [], history, {}, 30.0)
        
        _update_history(history, 2, det2)
        events = counter.update(2, det2, RAMP, [], history, {}, 30.0)
        assert len(events) == 1
        assert events[0]["method"] == "trajectory_fallback"

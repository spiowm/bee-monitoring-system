from services.orientation import get_orientation_vector, should_count_crossing
from services.track_history import TrackHistory
import numpy as np


class TrafficCounter:
    """
    Detects line crossings per track using shared TrackHistory.
    """

    def __init__(self, line_position=0.5, approach="A", angle_threshold_deg=60.0):
        self.line_position = line_position
        self.approach = approach
        self.angle_threshold_deg = angle_threshold_deg
        # track_id → last counted frame (debounce)
        self.track_counted: dict[int, int] = {}

    def get_line_y(self, ramp_bbox):
        if ramp_bbox is None:
            return None
        rx1, ry1, rx2, ry2 = ramp_bbox
        return ry1 + self.line_position * (ry2 - ry1)

    def update(self, frame_num, tracked_detections, ramp_bbox, keypoints_xy,
               history: TrackHistory, behaviors=None, fps=30.0):
        events = []
        line_y = self.get_line_y(ramp_bbox)

        if line_y is None or tracked_detections.tracker_id is None:
            return events

        for i, track_id in enumerate(tracked_detections.tracker_id):
            entry = history.get(track_id)
            if entry is None or len(entry.positions) < 2:
                continue

            # Debounce
            if track_id in self.track_counted:
                if frame_num - self.track_counted[track_id] < 45:
                    continue

            positions = entry.last_n_positions(5)
            prev_y = positions[-2][1]
            curr_y = positions[-1][1]

            if not ((prev_y < line_y and curr_y >= line_y) or
                    (prev_y > line_y and curr_y <= line_y)):
                continue

            direction = "OUT" if curr_y > prev_y else "IN"
            method = "trajectory_only"
            angle_deg = None
            valid = True

            metrics = entry.compute_metrics(fps)
            speed_px_per_sec = metrics["current_speed"]

            if self.approach == "B":
                track_dir_vec = np.array(metrics["track_dir_vec"]) if len(positions) >= 3 else np.array(metrics["instant_dir_vec"])

                kp = keypoints_xy[i] if i < len(keypoints_xy) else None
                orient_vec = get_orientation_vector(kp)

                if orient_vec is not None:
                    track_norm = np.linalg.norm(track_dir_vec)
                    if track_norm > 1e-6:
                        t_dir = track_dir_vec / track_norm
                        dot = np.clip(np.dot(t_dir, orient_vec), -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(dot))

                    valid = should_count_crossing(track_dir_vec, orient_vec, self.angle_threshold_deg)
                    method = "pose_confirmed" if valid else "rejected"
                else:
                    method = "trajectory_fallback"
                    valid = True

            if valid and method != "rejected":
                self.track_counted[track_id] = frame_num
                behav = behaviors.get(track_id) if behaviors else None
                events.append({
                    "frame": frame_num,
                    "timestamp_sec": frame_num / fps,
                    "track_id": int(track_id),
                    "direction": direction,
                    "method": method,
                    "speed_px_per_sec": float(speed_px_per_sec),
                    "behavior_class": behav,
                    "angle_deg": float(angle_deg) if angle_deg is not None else None,
                })

        return events

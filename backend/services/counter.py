from services.orientation import get_orientation_vector, should_count_crossing
import numpy as np

class TrafficCounter:
    """
    Стан per-track для детекції перетину лінії.
    """
    def __init__(self, line_position=0.5, approach="A", angle_threshold_deg=60.0):
        self.line_position = line_position
        self.approach = approach # "A" або "B"
        self.angle_threshold_deg = angle_threshold_deg
        
        # track_id -> list of last 5 centers (y coordinates)
        self.track_centers = {}
        # track_id -> last counted frame (debounce)
        self.track_counted = {}
        
    def get_line_y(self, ramp_bbox):
        if ramp_bbox is None:
            return None
        rx1, ry1, rx2, ry2 = ramp_bbox
        return ry1 + self.line_position * (ry2 - ry1)
        
    def update(self, frame_num, tracked_detections, ramp_bbox, keypoints_xy, behaviors=None, fps=30.0):
        """
        tracked_detections: sv.Detections
        keypoints_xy: list of keypoints for each detection (matched by index)
        """
        events = []
        line_y = self.get_line_y(ramp_bbox)
        
        if line_y is None or tracked_detections.tracker_id is None:
            return events
            
        for i, track_id in enumerate(tracked_detections.tracker_id):
            xyxy = tracked_detections.xyxy[i]
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            
            if track_id not in self.track_centers:
                self.track_centers[track_id] = []
            
            self.track_centers[track_id].append((cx, cy))
            if len(self.track_centers[track_id]) > 5:
                self.track_centers[track_id].pop(0)
                
            # Debounce check
            if track_id in self.track_counted:
                if frame_num - self.track_counted[track_id] < 45:
                    continue
                    
            history = self.track_centers[track_id]
            if len(history) >= 2:
                prev_y = history[-2][1]
                curr_y = history[-1][1]
                
                # Check intersection
                if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
                    direction = "OUT" if curr_y > prev_y else "IN"
                    
                    method = "trajectory_only"
                    angle_deg = None
                    valid = True
                    
                    # Compute speed
                    prev_x = history[-2][0]
                    curr_x = history[-1][0]
                    dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    speed_px_per_sec = dist * fps
                    
                    if self.approach == "B":
                        if len(history) >= 3:
                            start_pt = np.array(history[0])
                            end_pt = np.array(history[-1])
                            track_dir_vec = end_pt - start_pt
                        else:
                            track_dir_vec = np.array([curr_x - prev_x, curr_y - prev_y])
                            
                        kp = keypoints_xy[i] if i < len(keypoints_xy) else None
                        orient_vec = get_orientation_vector(kp)
                        
                        if orient_vec is not None:
                            # calculate actual angle for logging
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
                            "angle_deg": float(angle_deg) if angle_deg is not None else None
                        })
                        
        return events

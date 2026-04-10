import numpy as np

class BehaviorAnalyzer:
    """
    Розпізнавання поведінкових патернів на основі треків.
    """
    def __init__(self):
        # track_id -> dict with history of positions, speeds, etc.
        self.tracks_history = {}

    def update_history(self, frame_num, tracked_detections, fps):
        """Оновлює історію треків для подальшого аналізу. Викликається на кожному кадрі."""
        current_ids = set()
        
        # tracked_detections is supervision Detections object with tracker_id
        if tracked_detections.tracker_id is not None:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                current_ids.add(track_id)
                xyxy = tracked_detections.xyxy[i]
                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                
                if track_id not in self.tracks_history:
                    self.tracks_history[track_id] = {
                        'positions': [],
                        'frames': [],
                        'behavior': None
                    }
                
                self.tracks_history[track_id]['positions'].append((cx, cy))
                self.tracks_history[track_id]['frames'].append(frame_num)
                
                # Keep only last 150 frames (approx 5 sec at 30fps)
                if len(self.tracks_history[track_id]['frames']) > 150:
                    self.tracks_history[track_id]['positions'].pop(0)
                    self.tracks_history[track_id]['frames'].pop(0)

        # Cleanup old tracks
        tracks_to_remove = [tid for tid in self.tracks_history.keys() if tid not in current_ids]
        for tid in tracks_to_remove:
            # We delay removal slightly or immediately remove
            pass
            
        # Optional: cleanup tracks not seen for 60 frames
        for tid in list(self.tracks_history.keys()):
            if self.tracks_history[tid]['frames'] and (frame_num - self.tracks_history[tid]['frames'][-1]) > 60:
                del self.tracks_history[tid]

    def analyze(self, fps: float = 30.0) -> dict:
        """
        Аналізує поточні треки. Викликається кожні 15 кадрів.
        Повертає dict: track_id -> 'foraging'|'fanning'|'guarding'|'washboarding' або None
        """
        behaviors = {}
        for track_id, data in self.tracks_history.items():
            positions = data['positions']
            if len(positions) < 15: # Need at least 0.5 sec of data
                behaviors[track_id] = None
                continue
                
            pos_np = np.array(positions)
            # Calculate speeds and distances
            diffs = np.diff(pos_np, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_dist = np.sum(distances)
            duration_sec = len(positions) / fps
            avg_speed = total_dist / duration_sec
            
            # Simple heuristic classification
            if avg_speed > 100.0:
                behavior = "foraging"
            elif avg_speed < 15.0 and duration_sec > 2.0:
                behavior = "fanning"
            elif 15.0 <= avg_speed <= 80.0:
                # Check bounding box horizontal spread vs vertical spread
                spread_x = np.max(pos_np[:, 0]) - np.min(pos_np[:, 0])
                spread_y = np.max(pos_np[:, 1]) - np.min(pos_np[:, 1])
                if spread_x > spread_y * 1.5:
                    behavior = "guarding"
                else:
                    behavior = "washboarding" # simplified fallback
            else:
                behavior = "washboarding"
                
            data['behavior'] = behavior
            behaviors[track_id] = behavior
            
        return behaviors

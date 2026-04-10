import cv2
import numpy as np

def track_color(track_id: int) -> tuple:
    hue = (track_id * 137.508) % 360
    # simple HSV to BGR for drawing using cv2
    rgb = cv2.cvtColor(np.uint8([[[hue / 2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

class FrameAnnotator:
    def __init__(self, viz_config: dict):
        self.viz_config = viz_config
        self.track_tails = {} # track_id -> list of centers

    def annotate(self, frame, tracked_detections, ramp_bbox, keypoints_xy, behaviors, counter, events, stats_state):
        """ Draw everything based on viz_config parameters. """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. RAMP OVERLAY
        if self.viz_config.get("show_ramp", True) and ramp_bbox is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in ramp_bbox]
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (41, 180, 240), -1) # #f0b429 BGR roughly
            cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
            cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (41, 180, 240), 2)
            cv2.putText(annotated_frame, "RAMP", (rx1 + 5, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (41, 180, 240), 2)

        # 2. COUNTING LINE
        if self.viz_config.get("show_counting_line", True) and ramp_bbox is not None:
            line_y = int(counter.get_line_y(ramp_bbox))
            rx1, ry1, rx2, ry2 = [int(v) for v in ramp_bbox]
            cv2.line(annotated_frame, (rx1, line_y), (rx2, line_y), (120, 187, 72), 2) # IN / OUT color base
            cv2.putText(annotated_frame, "IN ^", (rx1 + 5, line_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 187, 72), 2)
            cv2.putText(annotated_frame, "OUT v", (rx2 - 50, line_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (129, 129, 252), 2)

        # Process detections
        active_ids = set()
        if tracked_detections.tracker_id is not None:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                active_ids.add(track_id)
                xyxy = tracked_detections.xyxy[i]
                bx1, by1, bx2, by2 = [int(v) for v in xyxy]
                cx, cy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
                color = track_color(track_id)
                
                if track_id not in self.track_tails:
                    self.track_tails[track_id] = []
                self.track_tails[track_id].append((cx, cy))
                
                tail_len = self.viz_config.get("track_tail_length", 30)
                if len(self.track_tails[track_id]) > tail_len:
                    self.track_tails[track_id] = self.track_tails[track_id][-tail_len:]

                # 3. TRACKS
                if self.viz_config.get("show_tracks", True):
                    pts = self.track_tails[track_id]
                    for pt_idx in range(1, len(pts)):
                        alpha = pt_idx / len(pts)
                        # OpenCV doesn't easily do alpha lines without overlay, simplified fade below:
                        thickness = int(2 * alpha) + 1
                        cv2.line(annotated_frame, pts[pt_idx-1], pts[pt_idx], color, thickness)

                # 4. BOUNDING BOXES
                if self.viz_config.get("show_boxes", True):
                    thickness = 3 if track_id in [e['track_id'] for e in events] else 2
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, thickness)
                    if self.viz_config.get("show_confidence", True) and tracked_detections.confidence is not None:
                        conf = float(tracked_detections.confidence[i])
                        cv2.putText(annotated_frame, f"{conf:.2f}", (bx1, by2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # 5. TRACK IDs
                if self.viz_config.get("show_ids", True):
                    cv2.rectangle(annotated_frame, (bx1, by1 - 20), (bx1 + 30, by1), (0,0,0), -1)
                    cv2.putText(annotated_frame, str(track_id), (bx1 + 2, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 6. BEHAVIOR LABELS
                if self.viz_config.get("show_behaviors", True) and behaviors and track_id in behaviors:
                    b_mapped = {"foraging": "FOR", "fanning": "FAN", "guarding": "GRD", "washboarding": "WSH"}
                    b_lbl = b_mapped.get(behaviors[track_id], "")
                    if b_lbl:
                        cv2.putText(annotated_frame, b_lbl, (bx1, by2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 7 & 8. KEYPOINTS and ORIENTATION
                kp = keypoints_xy[i] if i < len(keypoints_xy) else None
                if kp is not None and len(kp) >= 2:
                    hx, hy = int(kp[0][0]), int(kp[0][1])
                    tx, ty = int(kp[1][0]), int(kp[1][1])
                    
                    if self.viz_config.get("show_keypoints", True):
                        cv2.circle(annotated_frame, (hx, hy), 4, (0, 255, 0), -1) # Head green
                        cv2.circle(annotated_frame, (tx, ty), 4, (0, 0, 255), -1) # Tail red
                        
                    if self.viz_config.get("show_orientation", True):
                        dist = np.sqrt((hx-tx)**2 + (hy-ty)**2)
                        if dist > 10:
                            cv2.arrowedLine(annotated_frame, (tx, ty), (hx, hy), (41, 180, 240), 2, tipLength=0.4)

        # Cleanup old tails
        for tid in list(self.track_tails.keys()):
            if tid not in active_ids:
                del self.track_tails[tid]

        # 9. STATS OVERLAY
        if self.viz_config.get("show_stats_overlay", True):
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (w - 220, 10), (w - 10, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            cv2.putText(annotated_frame, f"IN: {stats_state.get('total_in', 0)} | OUT: {stats_state.get('total_out', 0)}", (w - 210, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Bees on ramp: {len(active_ids)}", (w - 210, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"FPS: {stats_state.get('fps', 0):.1f}", (w - 210, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated_frame

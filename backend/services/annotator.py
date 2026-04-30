import cv2
import numpy as np
from services.track_history import TrackHistory


def track_color(track_id: int) -> tuple:
    hue = (track_id * 137.508) % 360
    rgb = cv2.cvtColor(np.uint8([[[hue / 2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


class FrameAnnotator:
    def __init__(self, viz_config: dict):
        self.viz_config = viz_config

    def annotate(self, frame, tracked_detections, ramp_bbox, ramp_kpts, keypoints_xy,
                 behaviors, counter, events, stats_state, history: TrackHistory):
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        tail_len = self.viz_config.get("track_tail_length", 30)

        # 1. RAMP OVERLAY
        if self.viz_config.get("show_ramp", True) and ramp_bbox is not None:
            overlay = annotated_frame.copy()
            if ramp_kpts is not None and len(ramp_kpts) >= 4:
                pts = np.array(ramp_kpts[:4], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (41, 180, 240))
                cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
                cv2.polylines(annotated_frame, [pts], True, (41, 180, 240), 2)
                
                # Annotate points for clarity
                for idx, pt in enumerate(ramp_kpts[:4]):
                    cv2.putText(annotated_frame, str(idx+1), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                rx1, ry1 = int(ramp_bbox[0]), int(ramp_bbox[1])
                cv2.putText(annotated_frame, "RAMP", (rx1 + 5, ry1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (41, 180, 240), 2)
            else:
                rx1, ry1, rx2, ry2 = [int(v) for v in ramp_bbox]
                cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (41, 180, 240), -1)
                cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (41, 180, 240), 2)
                cv2.putText(annotated_frame, "RAMP", (rx1 + 5, ry1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (41, 180, 240), 2)

        # 2. COUNTING LINE
        if self.viz_config.get("show_counting_line", True) and ramp_bbox is not None:
            if ramp_kpts is not None and len(ramp_kpts) >= 2:
                x1, y1 = int(ramp_kpts[0][0]), int(ramp_kpts[0][1])
                x2, y2 = int(ramp_kpts[1][0]), int(ramp_kpts[1][1])
                shift_down = 8
                y1 += shift_down
                y2 += shift_down
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (120, 187, 72), 2)
                cv2.putText(annotated_frame, "IN ^", (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 187, 72), 2)
                cv2.putText(annotated_frame, "OUT v", (x2 - 50, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (129, 129, 252), 2)
            else:
                rx1, ry1, rx2, ry2 = [int(v) for v in ramp_bbox]
                # Default fallback line if no keypoints
                line_y = int(ry1 + 0.5 * (ry2 - ry1))
                cv2.line(annotated_frame, (rx1, line_y), (rx2, line_y), (120, 187, 72), 2)
                cv2.putText(annotated_frame, "IN ^", (rx1 + 5, line_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 187, 72), 2)
                cv2.putText(annotated_frame, "OUT v", (rx2 - 50, line_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (129, 129, 252), 2)

        # 3–8. Per-detection overlays
        if tracked_detections.tracker_id is not None:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                xyxy = tracked_detections.xyxy[i]
                bx1, by1, bx2, by2 = [int(v) for v in xyxy]
                color = track_color(track_id)

                # TRACKS (tail from shared history)
                if self.viz_config.get("show_tracks", True):
                    entry = history.get(track_id)
                    if entry:
                        pts = [(int(x), int(y)) for x, y in entry.last_n_positions(tail_len)]
                        for pt_idx in range(1, len(pts)):
                            alpha = pt_idx / len(pts)
                            thickness = int(2 * alpha) + 1
                            cv2.line(annotated_frame, pts[pt_idx - 1], pts[pt_idx], color, thickness)

                # BOUNDING BOXES
                if self.viz_config.get("show_boxes", True):
                    thickness = 3 if track_id in [e["track_id"] for e in events] else 2
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, thickness)
                    if self.viz_config.get("show_confidence", True) and tracked_detections.confidence is not None:
                        conf = float(tracked_detections.confidence[i])
                        cv2.putText(annotated_frame, f"{conf:.2f}", (bx1, by2 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # TRACK IDs
                if self.viz_config.get("show_ids", True):
                    cv2.rectangle(annotated_frame, (bx1, by1 - 20), (bx1 + 30, by1), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, str(track_id), (bx1 + 2, by1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # BEHAVIOR LABELS
                if self.viz_config.get("show_behaviors", True) and behaviors and track_id in behaviors:
                    b_mapped = {"foraging": "FOR", "fanning": "FAN", "guarding": "GRD", "washboarding": "WSH"}
                    b_lbl = b_mapped.get(behaviors[track_id], "")
                    if b_lbl:
                        cv2.putText(annotated_frame, b_lbl, (bx1, by2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # KEYPOINTS & ORIENTATION
                kp = keypoints_xy[i] if i < len(keypoints_xy) else None
                if kp is not None and len(kp) >= 2:
                    hx, hy = int(kp[0][0]), int(kp[0][1])
                    tx, ty = int(kp[1][0]), int(kp[1][1])
                    if self.viz_config.get("show_keypoints", True):
                        cv2.circle(annotated_frame, (hx, hy), 4, (0, 255, 0), -1)
                        cv2.circle(annotated_frame, (tx, ty), 4, (0, 0, 255), -1)
                    if self.viz_config.get("show_orientation", True):
                        dist = np.sqrt((hx - tx) ** 2 + (hy - ty) ** 2)
                        if dist > 10:
                            cv2.arrowedLine(annotated_frame, (tx, ty), (hx, hy),
                                            (41, 180, 240), 2, tipLength=0.4)

        # 9. STATS OVERLAY
        if self.viz_config.get("show_stats_overlay", True):
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (w - 220, 10), (w - 10, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            cv2.putText(annotated_frame,
                        f"IN: {stats_state.get('total_in', 0)} | OUT: {stats_state.get('total_out', 0)}",
                        (w - 210, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            active = len(tracked_detections.tracker_id) if tracked_detections.tracker_id is not None else 0
            cv2.putText(annotated_frame, f"Bees on ramp: {active}",
                        (w - 210, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"FPS: {stats_state.get('fps', 0):.1f}",
                        (w - 210, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated_frame

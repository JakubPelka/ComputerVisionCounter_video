# cv_video_overlay.py
from __future__ import annotations
import numpy as np
import cv2

try:
    import supervision as sv
except Exception:
    sv = None

# Reuse shared HUD helpers (removes duplication with runner)
from cv_video_hud import draw_counts_panel, draw_lines_zones
# (we keep draw_detections here – it depends on supervision)

# ---------- detections ----------
def draw_detections(frame_bgr,
                    det_boxes, det_confs, det_cids, det_ids,
                    names, overlay_mode: str,
                    polygons=None,
                    show_anchor=False,
                    anchor_points=None):
    """
    Draw detection overlays.
    """
    if overlay_mode == "polygon" and polygons is not None and len(polygons) > 0 and sv is not None:
        for poly in polygons:
            if poly is None or len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame_bgr, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        if overlay_mode in ("boxes", "boxes_conf") and sv is not None and len(det_boxes) > 0:
            det = sv.Detections(
                xyxy=np.array(det_boxes, dtype=np.float32),
                confidence=np.array(det_confs, dtype=np.float32) if det_confs else None,
                class_id=np.array(det_cids, dtype=int) if det_cids else None,
                tracker_id=np.array(det_ids, dtype=int) if det_ids else None
            )
            labels = []
            for s, c, tid in zip(det_confs or [], det_cids or [], det_ids or []):
                nm = names[c] if isinstance(names, dict) else names[c]
                labels.append(f"{nm} ID{tid}" + (f" {s:.2f}" if overlay_mode == "boxes_conf" and s is not None else ""))

            try:
                frame_bgr[:] = sv.BoxAnnotator(thickness=2).annotate(frame_bgr, det, labels=labels)
            except TypeError:
                frame_bgr[:] = sv.BoxAnnotator(thickness=2).annotate(frame_bgr, det)
                try:
                    for (x1, y1, x2, y2), lab in zip(det.xyxy, labels):
                        cv2.putText(frame_bgr, lab, (int(x1), max(14, int(y1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                except Exception:
                    pass
        else:
            if show_anchor and anchor_points is not None:
                for (tid, pt) in zip(det_ids or [], anchor_points or []):
                    if pt is None:
                        continue
                    cx, cy = int(pt[0]), int(pt[1])
                    cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                    if tid is not None:
                        cv2.putText(frame_bgr, f"ID {tid}", (cx + 6, cy - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# ---------- counters & HUD ----------
def draw_counters(frame_bgr, lines_cfg, line_counts, zones_cfg, zone_counts, trails=None, trail_color=(255, 255, 0)):
    """
    Wrapper kept for compatibility: draws lines/zones and a bottom-right HUD panel.
    (Trails are drawn by the runner; here we focus on the static graphics + panel.)
    """
    draw_lines_zones(frame_bgr, lines_cfg, zones_cfg, frame_color=(0,165,255), frame_thickness=2)
    draw_counts_panel(frame_bgr, lines_cfg, line_counts, zones_cfg, zone_counts, anchor="br")

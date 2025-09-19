# cv_video_overlay.py
from __future__ import annotations
import numpy as np
import cv2

try:
    import supervision as sv
except Exception:
    sv = None


def draw_detections(frame_bgr,
                    det_boxes, det_confs, det_cids, det_ids,
                    names, overlay_mode: str,
                    polygons=None,
                    show_anchor=False,
                    anchor_points=None):
    """
    polygons: list[np.ndarray Nx2] lub None (używane przy overlay_mode='polygon')
    anchor_points: opcjonalnie lista (cx,cy) – dla kropki anchor
    """
    if overlay_mode == "polygon" and polygons is not None and len(polygons) > 0 and sv is not None:
        # jeżeli mamy maski/polygony – użyj PolygonAnnotator
        dets = sv.Detections.empty()
        # supervision PolygonAnnotator bierze polygony osobno; narysujmy „na surowo”
        for poly in polygons:
            if poly is None or len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(frame_bgr, [pts], True, (0,255,255), 2, cv2.LINE_AA)
    else:
        # Boksy / Boksy+conf / Centroid
        if overlay_mode in ("boxes","boxes_conf") and sv is not None and len(det_boxes) > 0:
            det = sv.Detections(xyxy=np.array(det_boxes, dtype=np.float32),
                                confidence=np.array(det_confs, dtype=np.float32) if det_confs else None,
                                class_id=np.array(det_cids, dtype=int) if det_cids else None,
                                tracker_id=np.array(det_ids, dtype=int) if det_ids else None)
            labels = []
            for s,c,tid in zip(det_confs or [], det_cids or [], det_ids or []):
                nm = names[c] if isinstance(names, dict) else names[c]
                labels.append(f"{nm} ID{tid}" + (f" {s:.2f}" if overlay_mode=="boxes_conf" and s is not None else ""))
            try:
                frame_bgr[:] = sv.BoxAnnotator(thickness=2).annotate(frame_bgr, det, labels=labels)
            except TypeError:
                frame_bgr[:] = sv.BoxAnnotator(thickness=2).annotate(frame_bgr, det)
                # dopisz label ręcznie
                try:
                    for (x1,y1,x2,y2), lab in zip(det.xyxy, labels):
                        cv2.putText(frame_bgr, lab, (int(x1), max(14,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                except Exception:
                    pass
        else:
            # centroid (kropka anchor)
            if anchor_points is not None:
                for (tid, pt) in zip(det_ids, anchor_points):
                    if pt is None: continue
                    cx,cy = int(pt[0]), int(pt[1])
                    cv2.circle(frame_bgr, (cx,cy), 4, (0,255,0), -1, lineType=cv2.LINE_AA)
                    cv2.putText(frame_bgr, f"ID {tid}", (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)


def draw_counters(frame_bgr, lines_cfg, line_counts, zones_cfg, zone_counts, trails=None, trail_color=(255,255,0)):
    # linie + liczniki
    for li, ln in enumerate(lines_cfg):
        a = (int(ln["a"][0]), int(ln["a"][1])); b2 = (int(ln["b"][0]), int(ln["b"][1]))
        cv2.line(frame_bgr, a, b2, (0,255,255), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"{ln['name']}  A->B:{line_counts[li]['ab']}  B->A:{line_counts[li]['ba']}",
                    (min(a[0],b2[0])+6, min(a[1],b2[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

    # strefy + liczniki
    for zi, zn in enumerate(zones_cfg):
        poly = np.array(zn["pts"], dtype=np.int32)
        cv2.polylines(frame_bgr, [poly], True, (0,165,255), 2)
        cx = int(np.mean(poly[:,0])); cy = int(np.mean(poly[:,1]))
        cv2.putText(frame_bgr, f"{zn['name']} IN:{zone_counts[zi]['in']} OUT:{zone_counts[zi]['out']}",
                    (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)

    # ślady
    if trails:
        for dq in trails.values():
            if len(dq) < 2: continue
            for i in range(1, len(dq)):
                cv2.line(frame_bgr, (int(dq[i-1][0]), int(dq[i-1][1])),
                                   (int(dq[i][0]),   int(dq[i][1])),
                                   trail_color, 2, cv2.LINE_AA)

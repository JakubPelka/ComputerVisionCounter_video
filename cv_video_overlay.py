# cv_video_overlay.py — rysowanie boxów/etykiet/trajektorii + linii/stref
from __future__ import annotations
import cv2, numpy as np

try:
    import supervision as sv
except Exception:
    sv = None

def _labels_for(det_confs, det_cids, det_ids, names, show_conf: bool):
    def nm(cid): return (names[cid] if isinstance(names, dict) else names[cid])
    labels = []
    for s,c,tid in zip(det_confs, det_cids, det_ids):
        lbl = f"{nm(c)} ID{tid}"
        if show_conf: lbl += f" {s:.2f}"
        labels.append(lbl)
    return labels

def draw_detections(frame, det_boxes, det_confs, det_cids, det_ids, names, mode: str):
    show_conf = (mode == "boxes_conf")
    if mode in ("boxes", "boxes_conf"):
        if sv is not None and len(det_boxes) > 0:
            det = sv.Detections(
                xyxy=np.array(det_boxes, dtype=np.float32).reshape(-1,4),
                confidence=np.array(det_confs, dtype=np.float32).reshape(-1),
                class_id=np.array(det_cids, dtype=int).reshape(-1),
                tracker_id=np.array(det_ids, dtype=int).reshape(-1),
            )
            labels = _labels_for(det_confs, det_cids, det_ids, names, show_conf)
            try:
                frame[:] = sv.BoxAnnotator(thickness=2).annotate(frame, det, labels=labels)
            except TypeError:
                frame[:] = sv.BoxAnnotator(thickness=2).annotate(frame, det)
                try:
                    for (x1,y1,x2,y2), _lab in zip(det.xyxy, labels):
                        x1,y1 = int(x1), int(y1)
                        cv2.putText(frame, str(_lab), (x1, max(14, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                except Exception:
                    pass
        else:
            for (x1,y1,x2,y2), s, c, tid in zip(det_boxes, det_confs, det_cids, det_ids):
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                lbl = f"ID{tid} " + (names[c] if isinstance(names, dict) else names[c])
                if show_conf: lbl += f" {s:.2f}"
                cv2.putText(frame, lbl, (int(x1), max(14, int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    else:
        for (tid, (cx,cy)) in zip(det_ids, [(0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])) for b in det_boxes]):
            cv2.circle(frame, (int(cx), int(cy)), 4, (0,255,0), -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"ID {tid}", (int(cx)+6, int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

def draw_counters(frame, lines_cfg, line_counts, zones_cfg, zone_counts, trails: dict[int, list[tuple[int,int]]]):
    # trajektorie
    for _tid, dq in trails.items():
        if len(dq) > 1:
            for i in range(1, len(dq)):
                cv2.line(frame, dq[i-1], dq[i], (0,200,0), 2)
    # linie (bez strzałki) + liczniki
    for li, ln in enumerate(lines_cfg):
        a = (int(ln["a"][0]), int(ln["a"][1])); b2 = (int(ln["b"][0]), int(ln["b"][1]))
        cv2.line(frame, a, b2, (0,255,255), 3)
        cv2.putText(frame, f"{ln['name']}  A->B:{line_counts[li]['ab']}  B->A:{line_counts[li]['ba']}",
                    (min(a[0],b2[0])+6, min(a[1],b2[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    # strefy
    for zi, zn in enumerate(zones_cfg):
        poly = np.array(zn["pts"], dtype=np.int32)
        cv2.polylines(frame, [poly], True, (0,165,255), 2)
        cx = int(np.mean(poly[:,0])); cy = int(np.mean(poly[:,1]))
        cv2.putText(frame, f"{zn['name']} IN:{zone_counts[zi]['in']} OUT:{zone_counts[zi]['out']}",
                    (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)

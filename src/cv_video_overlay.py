# cv_video_overlay.py
from __future__ import annotations
import numpy as np
import cv2

try:
    import supervision as sv
except Exception:
    sv = None


# ---------- helpers for the HUD ----------
def _ab_dir_hint_for_line(ln: dict) -> str:
    """
    Human hint for what A->B means on screen (same cross-product sign
    as used by the counter).
    """
    ax, ay = ln["a"]; bx, by = ln["b"]
    dx, dy = bx - ax, by - ay
    mx, my = (ax + bx) * 0.5, (ay + by) * 0.5

    def s(px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    if abs(dx) < abs(dy):  # vertical-ish → Left/Right
        sl = s(mx - 10, my)
        sr = s(mx + 10, my)
        return "Left->Right" if sl < sr else "Right->Left"
    else:                   # horizontal-ish → Up/Down
        su = s(mx, my - 10)
        sd = s(mx, my + 10)
        return "Up->Down" if su < sd else "Down->Up"


def _draw_hud_panel(
    img,
    lines: list[str],
    anchor: str = "br",  # tl / tr / bl / br
    margin: int = 12,
    pad: int = 8,
    font = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.6,
    color = (0, 255, 255),
    bgcolor = (0, 0, 0),
    alpha: float = 0.55,
    thickness: int = 2,
):
    """Draws a semi-transparent black panel with text lines."""
    if not lines:
        return
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    maxw = max((w for (w, h) in sizes), default=0)
    lineh = max((h for (w, h) in sizes), default=14)
    panel_w = maxw + 2 * pad
    panel_h = len(lines) * (lineh + 6) - 6 + 2 * pad  # 6px line spacing

    H, W = img.shape[:2]
    if anchor == "tl":
        x0, y0 = margin, margin
    elif anchor == "tr":
        x0, y0 = W - margin - panel_w, margin
    elif anchor == "bl":
        x0, y0 = margin, H - margin - panel_h
    else:  # "br"
        x0, y0 = W - margin - panel_w, H - margin - panel_h

    # semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), bgcolor, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # text
    y = y0 + pad + lineh
    for t in lines:
        cv2.putText(img, t, (x0 + pad, y), font, scale, color, thickness, cv2.LINE_AA)
        y += lineh + 6


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
            # Centroid markers
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
    Draws:
      • counting lines with arrow A->B and A/B endpoints,
      • zones outlines,
      • a bottom-right HUD panel with all results on black background.
    """
    # 1) lines (graphics only – no mid-line labels)
    for ln in lines_cfg or []:
        a = (int(ln["a"][0]), int(ln["a"][1])); b = (int(ln["b"][0]), int(ln["b"][1]))
        col = (0, 255, 255)
        try:
            cv2.arrowedLine(frame_bgr, a, b, col, 2, tipLength=0.08)
        except Exception:
            cv2.line(frame_bgr, a, b, col, 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, a, 4, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame_bgr, b, 4, col, -1, lineType=cv2.LINE_AA)
        cv2.putText(frame_bgr, "A", (a[0] + 6, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "B", (b[0] + 6, b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # 2) zones (graphics only)
    for zn in zones_cfg or []:
        pts = np.array(zn["pts"], dtype=np.int32)
        if len(pts) >= 3:
            cv2.polylines(frame_bgr, [pts], True, (0, 165, 255), 2)

    # 3) trails (optional)
    if trails:
        for dq in trails.values():
            if len(dq) < 2:
                continue
            for i in range(1, len(dq)):
                cv2.line(
                    frame_bgr,
                    (int(dq[i - 1][0]), int(dq[i - 1][1])),
                    (int(dq[i][0]),     int(dq[i][1])),
                    trail_color, 2, cv2.LINE_AA
                )

    # 4) HUD with results in bottom-right
    hud_lines: list[str] = []
    for i, ln in enumerate(lines_cfg or []):
        ab = line_counts[i]["ab"]; ba = line_counts[i]["ba"]
        hint = _ab_dir_hint_for_line(ln)
        hud_lines.append(f"{ln['name']}:  A->B {ab}  |  B->A {ba}  ({hint})")
    for i, zn in enumerate(zones_cfg or []):
        zin = zone_counts[i]["in"]; zout = zone_counts[i]["out"]
        hud_lines.append(f"{zn['name']}:  IN {zin}  |  OUT {zout}")

    _draw_hud_panel(frame_bgr, hud_lines, anchor="br", color=(255, 255, 255), bgcolor=(0, 0, 0), alpha=0.55)

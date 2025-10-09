# cv_video_hud.py
from __future__ import annotations
import cv2
import numpy as np
from cv_video_geom import get_line_pts

# ---- Direction hint ----------------------------------------------------------
def ab_dir_hint_for_line(ln: dict) -> str:
    """
    Human hint for what A->B means. Works for straight lines and polylines.
    Vertical-ish => Left->Right / Right->Left, Horizontal-ish => Up->Down / Down->Up.
    """
    pts = get_line_pts(ln)
    ax, ay = pts[0]; bx, by = pts[-1]
    dx, dy = bx - ax, by - ay
    mx, my = (ax + bx) * 0.5, (ay + by) * 0.5

    def s(px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    if abs(dx) < abs(dy):
        sl = s(mx - 10, my); sr = s(mx + 10, my)
        return "Left->Right" if sl < sr else "Right->Left"
    else:
        su = s(mx, my - 10); sd = s(mx, my + 10)
        return "Up->Down" if su < sd else "Down->Up"

# ---- Draw primitives ---------------------------------------------------------
def draw_lines_zones(frame, lines_cfg, zones_cfg, frame_color=None, frame_thickness=2):
    th = int(frame_thickness if frame_thickness is not None else 2)
    if th <= 0:
        return
    col_line = frame_color if frame_color is not None else (0, 165, 255)

    # Lines (straight & polyline)
    for ln in (lines_cfg or []):
        pts = get_line_pts(ln)
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                a = (int(pts[i-1][0]), int(pts[i-1][1]))
                b = (int(pts[i][0]),   int(pts[i][1]))
                if i == len(pts) - 1:
                    try:
                        cv2.arrowedLine(frame, a, b, col_line, max(1, th), tipLength=0.08)
                    except Exception:
                        cv2.line(frame, a, b, col_line, th, cv2.LINE_AA)
                else:
                    cv2.line(frame, a, b, col_line, th, cv2.LINE_AA)
            a0 = (int(pts[0][0]), int(pts[0][1])); b0 = (int(pts[-1][0]), int(pts[-1][1]))
            cv2.circle(frame, a0, 4, col_line, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, b0, 4, col_line, -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "A", (a0[0]+6, a0[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_line, 1, cv2.LINE_AA)
            cv2.putText(frame, "B", (b0[0]+6, b0[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_line, 1, cv2.LINE_AA)

    # Zones
    for zn in (zones_cfg or []):
        pts = np.array(zn["pts"], dtype=np.int32)
        if len(pts) >= 3:
            cv2.polylines(frame, [pts], True, col_line, th, cv2.LINE_AA)

def draw_trails(frame, trails, trace_color=None, trace_thickness=2):
    th = int(trace_thickness if trace_thickness is not None else 2)
    if not trails or th <= 0:
        return
    for tid, dq in trails.items():
        if len(dq) < 2:
            continue
        if trace_color is None:
            r = (37 * tid) % 256; g = (91 * tid) % 256; b = (157 * tid) % 256
            col = (int(b), int(g), int(r))
        else:
            col = trace_color
        pts = np.array(dq, dtype=np.int32)
        cv2.polylines(frame, [pts], False, col, th, cv2.LINE_AA)

# ---- HUD / results panel -----------------------------------------------------
def draw_counts_panel(frame, lines_cfg, line_counts, zones_cfg, zone_counts,
                      anchor: str = "br", margin: int = 12):
    """
    Compact results panel in the chosen corner (default: bottom-right), on solid black.
    """
    # Build lines of text
    rows: list[str] = []
    for i, ln in enumerate(lines_cfg or []):
        ab = int(line_counts[i].get("ab", 0))
        ba = int(line_counts[i].get("ba", 0))
        hint = ""
        try:
            hint = ab_dir_hint_for_line(ln)
        except Exception:
            pass
        rows.append(f"{ln['name']}:  A->B {ab}  |  B->A {ba}" + (f"  ({hint})" if hint else ""))
    for i, zn in enumerate(zones_cfg or []):
        zin = int(zone_counts[i].get("in", 0)); zout = int(zone_counts[i].get("out", 0))
        rows.append(f"{zn['name']}:  IN {zin}  |  OUT {zout}")

    if not rows:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, th = 1.0, 2
    pad, gap = 8, 6
    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t in rows]
    maxw  = max(w for w, h in sizes)
    lineh = max(h for w, h in sizes)
    panel_w = maxw + 2*pad
    panel_h = len(rows) * (lineh + gap) - gap + 2*pad

    H, W = frame.shape[:2]
    if anchor == "tl":
        x0, y0 = margin, margin
    elif anchor == "tr":
        x0, y0 = W - margin - panel_w, margin
    elif anchor == "bl":
        x0, y0 = margin, H - margin - panel_h
    else:  # "br"
        x0, y0 = W - margin - panel_w, H - margin - panel_h

    # solid black background (best readability)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    y = y0 + pad + lineh
    for t in rows:
        cv2.putText(frame, t, (x0 + pad, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)
        y += lineh + gap

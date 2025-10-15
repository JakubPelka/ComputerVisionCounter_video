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
def draw_counts_panel(frame, lines_cfg, line_counts, zones_cfg, zone_counts, anchor="br", app=None):
    """
    Right-side HUD: per-line AB/BA and per-zone IN/OUT with per-class breakdowns,
    plus a global SUM across zones (per-class).
    Example:
        Lines
        Main: AB 12, BA 9

        Zones
        Zone1: IN (5): person 3, car 2; OUT (2): person 1, car 1
        Zone2: IN (2): car 2; OUT (2): person 2

        SUM (zones)
        IN:  person 5, car 4
        OUT: person 3, car 1
    """
    if frame is None:
        return

    H, W = frame.shape[:2]

    # === unified scaling (same math everywhere) ===
    auto = max(0.6, min(2.2, H / 720.0))
    user = 1.0
    if app is not None:
        try:
            user = float(getattr(app, "adv_params", {}).get("hud_scale", 1.0))
        except Exception:
            user = 1.0
    s = auto * user

    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_scale = 0.6 * s
    th = max(1, int(2 * s))
    pad = int(8 * s)
    gap = int(6 * s)
    mar = int(12 * s)
    sec_gap = int(10 * s)  # extra gap after section headers

    def _fmt_counts(dct):
        if not isinstance(dct, dict) or not dct:
            return ""
        keys = sorted(dct.keys())
        return ", ".join(f"{k} {int(dct[k])}" for k in keys)

    rows = []

    # ─── Lines
    if lines_cfg and line_counts:
        rows.append("Lines")
        for i, ln in enumerate(lines_cfg):
            name = ln.get("name", f"line_{i}")
            lc = line_counts[i] if i < len(line_counts) else {"ab": 0, "ba": 0}
            rows.append(f"{name}: AB {int(lc.get('ab', 0))}, BA {int(lc.get('ba', 0))}")
        rows.append("")  # spacer

    # ─── Zones with per-class breakdown
    have_zones = bool(zones_cfg and zone_counts)
    rows.append("Zones") if have_zones else None

    # per-zone class buckets prepared by run.py
    bz = getattr(app, "_zone_class_totals_by_zone", None)
    for i, zn in enumerate(zones_cfg or []):
        name = zn.get("name", f"zone_{i}")
        zc = zone_counts[i] if i < len(zone_counts) else {"in": 0, "out": 0}
        in_total = int(zc.get("in", 0))
        out_total = int(zc.get("out", 0))

        # pull per-class breakdowns if present
        in_map = {}
        out_map = {}
        try:
            if isinstance(bz, list) and i < len(bz):
                in_map = dict(bz[i].get("in", {}))
                out_map = dict(bz[i].get("out", {}))
        except Exception:
            pass

        in_s = _fmt_counts(in_map)
        out_s = _fmt_counts(out_map)

        # Compose one line per zone, compact
        line = f"{name}: "
        line += f"IN ({in_total})"
        line += f": {in_s}" if in_s else ""
        line += "; "
        line += f"OUT ({out_total})"
        line += f": {out_s}" if out_s else ""
        rows.append(line)

    # ─── Global SUM across zones (per-class)
    gsum = getattr(app, "_zone_class_totals_sum", None)
    if have_zones and isinstance(gsum, dict) and (gsum.get("in") or gsum.get("out")):
        if rows and rows[-1] != "":
            rows.append("")
        rows.append("SUM (zones)")
        zin = _fmt_counts(gsum.get("in", {}))
        zout = _fmt_counts(gsum.get("out", {}))
        if zin:
            rows.append(f"IN:  {zin}")
        if zout:
            rows.append(f"OUT: {zout}")

    # Trim trailing spacer
    while rows and rows[-1] == "":
        rows.pop()

    if not rows:
        return

    # measure text
    sizes = [cv2.getTextSize(t, font, txt_scale, th)[0] for t in rows]
    maxw = max(w for w, h in sizes)
    lineh = max(h for w, h in sizes)

    def _is_header(idx: int) -> bool:
        return rows[idx] in ("Lines", "Zones", "SUM (zones)")

    panel_w = maxw + 2 * pad
    panel_h = 2 * pad
    for i, _t in enumerate(rows):
        panel_h += lineh
        if i < len(rows) - 1:
            panel_h += (sec_gap if _is_header(i) else gap)

    # anchor
    if anchor == "tl":
        x0, y0 = mar, mar
    elif anchor == "tr":
        x0, y0 = W - mar - panel_w, mar
    elif anchor == "bl":
        x0, y0 = mar, H - mar - panel_h
    else:  # "br"
        x0, y0 = W - mar - panel_w, H - mar - panel_h

    # background
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)

    # render
    y = y0 + pad + lineh
    for i, t in enumerate(rows):
        is_header = t in ("Lines", "Zones", "SUM (zones)")
        use_th = th + 1 if is_header else th
        cv2.putText(frame, t, (x0 + pad, y), font, txt_scale, (255, 255, 255), use_th, cv2.LINE_AA)
        if i < len(rows) - 1:
            y += lineh + (sec_gap if is_header else gap)

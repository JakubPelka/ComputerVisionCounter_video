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
    Right-side HUD:
      • Lines: AB/BA totals per line + per-class breakdowns on the same row.
      • Zones: IN/OUT totals per zone + per-class breakdowns (as before).
      • SUM (zones): global per-class IN/OUT.

    Falls back to rebuilding per-class maps from events (app._ev_ref) if accumulators
    are missing.
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
    sec_gap = int(10 * s)

    def _fmt_counts(dct):
        if not isinstance(dct, dict) or not dct:
            return ""
        keys = sorted(dct.keys())
        return ", ".join(f"{k} {int(dct[k])}" for k in keys)

    # ---------- helper: per-line class maps from events ----------
    def _compute_line_class_maps(app, lines_cfg):
        maps = [{"ab": {}, "ba": {}} for _ in (lines_cfg or [])]
        evs = getattr(app, "_ev_ref", None)
        if not (isinstance(evs, list) and evs and lines_cfg):
            return maps
        name2idx = {str(ln.get("name", f"line_{i}")): i for i, ln in enumerate(lines_cfg)}
        for e in evs:
            et = str(e.get("event_type", ""))
            if not et.startswith("line_"):
                continue
            nm = str(e.get("counter_name", ""))
            idx = name2idx.get(nm, None)
            if idx is None:
                continue
            cls = str(e.get("class_name", e.get("class_id", "")))
            if et == "line_ab" or int(e.get("AB", 0)) == 1:
                maps[idx]["ab"][cls] = int(maps[idx]["ab"].get(cls, 0)) + 1
            elif et == "line_ba" or int(e.get("BA", 0)) == 1:
                maps[idx]["ba"][cls] = int(maps[idx]["ba"].get(cls, 0)) + 1
        return maps

    # ---------- get per-zone class maps (fast path + fallback) ----------
    bz = getattr(app, "_zone_class_totals_by_zone", None)
    gsum = getattr(app, "_zone_class_totals_sum", None)
    need_rebuild_zones = (
        not isinstance(bz, list) or len(bz) != len(zones_cfg or []) or
        all((not z or (not z.get("in") and not z.get("out"))) for z in (bz or []))
    )
    if need_rebuild_zones:
        bz = [{"in": {}, "out": {}} for _ in (zones_cfg or [])]
        gsum = {"in": {}, "out": {}}
        evs = getattr(app, "_ev_ref", None)
        if isinstance(evs, list) and evs:
            name2idx = {str(zn.get("name", f"zone_{i}")): i for i, zn in enumerate(zones_cfg or [])}
            for e in evs:
                et = str(e.get("event_type", ""))
                if not et.startswith("zone_"):
                    continue
                nm = str(e.get("counter_name", ""))
                idx = name2idx.get(nm, None)
                if idx is None:
                    continue
                cls = str(e.get("class_name", e.get("class_id", "")))
                direction = "in" if et == "zone_in" else "out"
                zmap = bz[idx][direction]; zmap[cls] = int(zmap.get(cls, 0)) + 1
                gmap = gsum[direction];    gmap[cls] = int(gmap.get(cls, 0)) + 1

    # ---------- build rows ----------
    rows = []

    # ─── Lines with per-class breakdown on the same row
    if lines_cfg and line_counts:
        rows.append("Lines")
        line_cls_maps = _compute_line_class_maps(app, lines_cfg)
        for i, ln in enumerate(lines_cfg):
            name = ln.get("name", f"line_{i}")
            lc = line_counts[i] if i < len(line_counts) else {"ab": 0, "ba": 0}
            ab_total = int(lc.get("ab", 0)); ba_total = int(lc.get("ba", 0))
            ab_map = line_cls_maps[i]["ab"] if i < len(line_cls_maps) else {}
            ba_map = line_cls_maps[i]["ba"] if i < len(line_cls_maps) else {}
            ab_s = _fmt_counts(ab_map); ba_s = _fmt_counts(ba_map)
            dir_label = ln.get("dir_label") or ln.get("direction_label") or ""
            if not dir_label:
                try:
                    dir_label = ab_dir_hint_for_line(ln)
                except Exception:
                    dir_label = ""
            line = f"{name}: AB {ab_total}"
            if ab_s: line += f" ({ab_s})"
            line += "  |  "
            line += f"BA {ba_total}"
            if ba_s: line += f" ({ba_s})"
            if dir_label:
                line += f"  ({dir_label})"
            rows.append(line)
        rows.append("")

    # ─── Zones with per-class breakdown (unchanged from your new spec)
    have_zones = bool(zones_cfg and zone_counts)
    if have_zones:
        rows.append("Zones")
    for i, zn in enumerate(zones_cfg or []):
        name = zn.get("name", f"zone_{i}")
        zc = zone_counts[i] if i < len(zone_counts) else {"in": 0, "out": 0}
        in_total = int(zc.get("in", 0)); out_total = int(zc.get("out", 0))
        in_map = {}; out_map = {}
        try:
            if isinstance(bz, list) and i < len(bz):
                in_map = dict(bz[i].get("in", {})); out_map = dict(bz[i].get("out", {}))
        except Exception:
            pass
        in_s = _fmt_counts(in_map); out_s = _fmt_counts(out_map)
        line = f"{name}: IN ({in_total})"
        line += f": {in_s}" if in_s else ""
        line += "; "
        line += f"OUT ({out_total})"
        line += f": {out_s}" if out_s else ""
        rows.append(line)

    # ─── Global SUM across zones (per-class)
    if have_zones and isinstance(gsum, dict) and (gsum.get("in") or gsum.get("out")):
        if rows and rows[-1] != "":
            rows.append("")
        rows.append("SUM (zones)")
        zin = _fmt_counts(gsum.get("in", {})); zout = _fmt_counts(gsum.get("out", {}))
        if zin:  rows.append(f"IN:  {zin}")
        if zout: rows.append(f"OUT: {zout}")

    while rows and rows[-1] == "":
        rows.pop()
    if not rows:
        return

    # measure
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

    # background + render
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    y = y0 + pad + lineh
    for i, t in enumerate(rows):
        is_header = t in ("Lines", "Zones", "SUM (zones)")
        use_th = th + 1 if is_header else th
        cv2.putText(frame, t, (x0 + pad, y), font, txt_scale, (255, 255, 255), use_th, cv2.LINE_AA)
        if i < len(rows) - 1:
            y += lineh + (sec_gap if is_header else gap)


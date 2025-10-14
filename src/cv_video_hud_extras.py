# cv_video_hud_extras.py
from __future__ import annotations
import cv2
import numpy as np

def draw_run_counters(frame, now_counts, max_counts=None, anchor: str = "bl", app=None):
    """Black HUD:
       Max this run — class: n, ...
       Now — class: n, ...
       Corner: tl/tr/bl/br (default bl). Uses the SAME scaling as draw_counts_panel.
    """
    if frame is None:
        return

    # Build lines
    rows: list[str] = []
    def _fmt(dd):
        if not dd:
            return ""
        return ", ".join([f"{k}: {dd[k]}" for k in sorted(dd.keys())])
    if max_counts:
        s = _fmt(max_counts)
        if s: rows.append("Max this run: " + s)
    s = _fmt(now_counts or {})
    if s: rows.append("Now: " + s)
    if not rows:
        return

    H, W = frame.shape[:2]

    # === unified scaling with draw_counts_panel ===
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

    # measure
    sizes = [cv2.getTextSize(t, font, txt_scale, th)[0] for t in rows]
    if not sizes:
        return
    maxw  = max(w for w, h in sizes)
    lineh = max(h for w, h in sizes)
    panel_w = maxw + 2 * pad
    panel_h = len(rows) * (lineh + gap) - gap + 2 * pad

    # anchor
    if anchor == "tl":
        x0, y0 = mar, mar
    elif anchor == "tr":
        x0, y0 = W - mar - panel_w, mar
    elif anchor == "bl":
        x0, y0 = mar, H - mar - panel_h
    else:  # "br"
        x0, y0 = W - mar - panel_w, H - mar - panel_h

    # draw
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    y = y0 + pad + lineh
    for t in rows:
        cv2.putText(frame, t, (x0 + pad, y), font, txt_scale, (255, 255, 255), th, cv2.LINE_AA)
        y += lineh + gap

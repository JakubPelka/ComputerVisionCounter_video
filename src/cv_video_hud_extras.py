# cv_video_hud_extras.py
from __future__ import annotations
import cv2
import numpy as np

def draw_run_counters(frame, now_counts, max_counts=None, anchor: str = "bl", app=None):
    """Black HUD:
       Max this run — class: n, ...
       Now — class: n, ...
       Corner: tl/tr/bl/br (default bl). Uses hud_scale if present.
    """
    if frame is None:
        return
    rows = []
    def _fmt(dd):
        if not dd: return ""
        return ", ".join([f"{k}: {dd[k]}" for k in sorted(dd.keys())])
    if max_counts:
        s = _fmt(max_counts)
        if s: rows.append("Max this run: " + s)
    s = _fmt(now_counts or {})
    if s: rows.append("Now: " + s)
    if not rows:
        return

    H, W = frame.shape[:2]
    hud_scale = 1.0
    try:
        if app and hasattr(app, 'adv_params'):
            val = float(app.adv_params.get('hud_scale', 1.0))
            if val > 0: hud_scale = float(val)
    except Exception:
        pass

    base_h = int(max(16, round(H * 0.018 * hud_scale)))
    pad = max(6, int(base_h * 0.5))
    lineh = max(18, int(base_h * 1.3))
    gap = max(2, int(base_h * 0.15))
    th = max(1, int(base_h * 0.08))
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_scale = max(0.4, base_h / 30.0)

    max_w = 0
    for t in rows:
        (tw, _), _ = cv2.getTextSize(t, font, txt_scale, th)
        max_w = max(max_w, tw)
    panel_w = int(max_w + 2*pad)
    panel_h = int(len(rows) * lineh + (len(rows)-1) * gap + 2*pad)

    mar = max(8, int(12 * hud_scale))
    if anchor == "tl":   x0, y0 = mar, mar
    elif anchor == "tr": x0, y0 = W - mar - panel_w, mar
    elif anchor == "bl": x0, y0 = mar, H - mar - panel_h
    else:                x0, y0 = W - mar - panel_w, H - mar - panel_h

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0,0,0), -1)
    y = y0 + pad + lineh
    for t in rows:
        cv2.putText(frame, t, (x0 + pad, y), font, txt_scale, (255,255,255), th, cv2.LINE_AA)
        y += lineh + gap

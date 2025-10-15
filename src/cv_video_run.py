# cv_video_run.py — runner with live transparent heatmap overlay & periodic RGBA saves
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import time
import re as _re
import inspect
import math
from collections import deque

import cv2
import numpy as np
import pandas as pd

from cv_video_gui import CounterEditor
from cv_video_core import (
    ensure_dir,
    device_auto_str,
    open_video_writer_collision,
    save_json_collision,
    save_csv_collision,
    VIDEO_PRESETS,
    DEFAULT_QUALITY,
    LINE_MIN_GAP_FRAMES_DEFAULT,
    ZONE_MIN_GAP_FRAMES_DEFAULT,
)
from cv_video_overlay import draw_detections  # centroid overlay helper (project)
from cv_video_geom import (
    get_line_pts,
    line_side,
    segments_intersect,
    point_in_polygon,
    polyline_side,
    polyline_cross_direction,
)
from cv_video_hud import draw_lines_zones, draw_trails, draw_counts_panel
from cv_video_sound import SoundPlayer

# Global counters (Now/Max HUD)
from cv_video_stats import StatsAggregator
from cv_video_hud_extras import draw_run_counters

# Heatmap (transparent zeros + masked overlay)
from cv_video_heatmap import DetectionHeatmap, build_mask_from_zones

try:
    import supervision as sv  # ByteTrack, Detections
except Exception:
    sv = None

# ──────────────────────────────────────────────────────────────────────────────
# small helpers
# ──────────────────────────────────────────────────────────────────────────────
_URL_RE = _re.compile(r"^\s*(rtsp|rtsps|rtmp|http|https)://", flags=_re.I)


def _is_stream_source(src):
    if isinstance(src, (int,)):
        return True
    if isinstance(src, str) and _URL_RE.match(src):
        return True
    return False


def _ensure_bgr(img):
    if img is None:
        return img
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _preview(app, frame_bgr, frame_idx, fps, total_frames):
    try:
        if hasattr(app, "_update_preview"):
            app._update_preview(frame_bgr, frame_idx, fps, total_frames)
            return
        if hasattr(app, "_show_preview_bgr"):
            app._show_preview_bgr(frame_bgr)
            return
        if hasattr(app, "update_preview"):
            app.update_preview(frame_bgr, frame_idx, fps, total_frames)
            return
        if hasattr(app, "show_preview"):
            app.show_preview(frame_bgr)
            return
    except Exception:
        pass


def _parse_color(val):
    if val is None:
        return None
    if isinstance(val, (tuple, list)) and len(val) == 3:
        b, g, r = val
        return (int(b), int(g), int(r))
    s = str(val).strip()
    if not s or s.lower() == "auto":
        return None
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return (b, g, r)
    try:
        parts = [int(x.strip()) for x in s.replace(";", ",").split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    except Exception:
        pass
    return None


def _anchor_from_box(b, anchor_mode: str, ghost_margin: int):
    x1, y1, x2, y2 = map(int, b)
    if anchor_mode == "bottom":
        cx = (x1 + x2) // 2
        cy = max(y1, y2) - int(ghost_margin)
        return (cx, cy)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _fmt_timecode(sec: float) -> str:
    if sec < 0:
        sec = 0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ──────────────────────────────────────────────────────────────────────────────
# YOLO results normalizer (Ultralytics v8/v11 compatible)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_ultra_results(res):
    np = __import__("numpy")
    # choose the first result-like object
    try:
        r0 = res[0] if hasattr(res, "__len__") and len(res) > 0 else res
    except Exception:
        r0 = res
    # v8/v11 path
    try:
        b = getattr(r0, "boxes", None)
        if b is not None:
            xyxy = b.xyxy
            conf = b.conf
            cls = b.cls
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            if hasattr(conf, "cpu"):
                conf = conf.cpu().numpy()
            if hasattr(cls, "cpu"):
                cls = cls.cpu().numpy()
            return (
                np.asarray(xyxy).astype(int),
                np.asarray(conf).astype(float),
                np.asarray(cls).astype(int),
            )
    except Exception:
        pass
    # fallback
    try:
        arr = getattr(r0, "xyxy", None)
        if arr is not None:
            if isinstance(arr, list):
                arr = arr[0]
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            arr = np.asarray(arr)
            return arr[:, :4].astype(int), arr[:, 4].astype(float), arr[:, 5].astype(int)
    except Exception:
        pass
    try:
        b = getattr(r0, "boxes", None)
        data = getattr(b, "data", None) if b is not None else None
        if data is not None:
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            arr = np.asarray(data)
            return arr[:, :4].astype(int), arr[:, 4].astype(float), arr[:, 5].astype(int)
    except Exception:
        pass
    return (
        np.empty((0, 4), dtype=int),
        np.empty((0,), dtype=float),
        np.empty((0,), dtype=int),
    )


# ──────────────────────────────────────────────────────────────────────────────
# simple local overlays (centroid/boxes) to avoid tight coupling
# ──────────────────────────────────────────────────────────────────────────────
_PALETTE = [
    (40, 200, 255),
    (255, 160, 40),
    (120, 220, 60),
    (90, 180, 255),
    (255, 90, 180),
    (180, 130, 255),
    (70, 230, 210),
    (240, 210, 70),
    (120, 120, 255),
    (255, 120, 120),
]


def _class_color(cid: int):
    try:
        return _PALETTE[int(cid) % len(_PALETTE)]
    except Exception:
        return (40, 200, 255)


def _draw_boxes_local(img, boxes, scores, cids, names, show_conf: bool):
    if img is None:
        return img
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cid = int(cids[i]) if (cids is not None and len(cids) > i) else -1
        clr = _class_color(cid)
        label = ""
        if cids is not None and len(cids) > i and cid >= 0:
            try:
                label = names[cid] if isinstance(names, dict) else names[cid]
            except Exception:
                label = str(cid)
        if show_conf and scores is not None and len(scores) > i:
            label = f"{label} {scores[i]:.2f}" if label else f"{scores[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), clr, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y0 = max(0, y1 - th - 6)
            cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y0 + th + 4), clr, -1)
            cv2.putText(
                img,
                label,
                (x1 + 3, y0 + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return img


def _draw_centroids_local(img, boxes):
    if img is None:
        return img
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 4, (40, 200, 255), -1)
    return img


def _draw_detections_safe(img, boxes, scores, cids, det_ids, names, overlay_mode):
    mode = (overlay_mode or "centroid").lower().strip()
    if mode.startswith("box"):  # 'box' or 'box+conf'
        return _draw_boxes_local(img, boxes, scores, cids, names, show_conf=("conf" in mode))
    # centroid → try project overlay first
    try:
        res = draw_detections(img, boxes, scores, cids, det_ids, names, mode="centroid")
        return res if res is not None else _draw_centroids_local(img, boxes)
    except TypeError:
        try:
            res = draw_detections(img, boxes, scores, cids, det_ids, names)
            return res if res is not None else _draw_centroids_local(img, boxes)
        except Exception:
            return _draw_centroids_local(img, boxes)


def _writer_write_safe(writer, ov, fallback_frame, expected_WH, app):
    W, H = expected_WH
    frame = ov
    try:
        if frame is None:
            frame = fallback_frame.copy()
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.dtype != np.uint8:
            try:
                frame = frame.astype(np.uint8, copy=False)
            except Exception:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        h, w = frame.shape[:2]
        if (w, h) != (W, H):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)
        return True
    except Exception as e:
        try:
            app._log(f"[WARN] writer.write failed: {e}")
        except Exception:
            pass
        try:
            writer.write(cv2.resize(_ensure_bgr(fallback_frame.copy()), (W, H)))
            return True
        except Exception as e2:
            try:
                app._log(f"[ERROR] writer.write retry failed: {e2}")
            except Exception:
                pass
            return False


# ──────────────────────────────────────────────────────────────────────────────
# tracker helpers (ByteTrack via Supervision)
# ──────────────────────────────────────────────────────────────────────────────
def _make_bytetrack(conf, track_buffer, match_thresh, min_hits, fps):
    if sv is None or not hasattr(sv, "ByteTrack"):
        return None
    try:
        params = inspect.signature(sv.ByteTrack.__init__).parameters
    except Exception:
        try:
            return sv.ByteTrack()
        except Exception:
            return None

    kwargs = {}
    if "track_thresh" in params:
        kwargs["track_thresh"] = max(0.05, min(float(conf), 0.99))
    if "track_buffer" in params:
        kwargs["track_buffer"] = int(track_buffer)
    if "match_thresh" in params:
        kwargs["match_thresh"] = float(match_thresh)
    if "min_hits" in params:
        kwargs["min_hits"] = int(min_hits)
    if "mot20" in params:
        kwargs["mot20"] = False
    if "frame_rate" in params:
        kwargs["frame_rate"] = float(max(1.0, fps))

    try:
        return sv.ByteTrack(**kwargs)
    except TypeError:
        basic = {k: kwargs[k] for k in ("track_thresh", "track_buffer", "match_thresh") if k in kwargs}
        try:
            return sv.ByteTrack(**basic)
        except Exception:
            try:
                return sv.ByteTrack()
            except Exception:
                return None


def _track_update(tracker, boxes, scores, cids):
    if tracker is None or boxes is None or boxes.shape[0] == 0 or sv is None:
        return boxes, scores, cids, None
    try:
        dets = sv.Detections(xyxy=boxes.astype(float), class_id=cids, confidence=scores)
    except Exception:
        return boxes, scores, cids, None

    for mname in ("update_with_detections", "update"):
        m = getattr(tracker, mname, None)
        if m is None:
            continue
        try:
            tracked = m(dets)
        except Exception:
            continue

        if hasattr(tracked, "xyxy") and getattr(tracked, "tracker_id", None) is not None:
            t_boxes = tracked.xyxy.astype(int)
            t_ids = tracked.tracker_id.astype(int)
            t_cids = (
                tracked.class_id.astype(int)
                if getattr(tracked, "class_id", None) is not None
                else (cids[: len(t_ids)] if cids is not None else None)
            )
            if getattr(tracked, "confidence", None) is not None:
                t_scores = tracked.confidence
            else:
                t_scores = scores[: len(t_ids)] if scores is not None and len(scores) >= len(t_ids) else None
            return t_boxes, t_scores, t_cids, t_ids

        # older trackers as list of objects
        if isinstance(tracked, (list, tuple)) and len(tracked) > 0 and hasattr(tracked[0], "track_id"):
            t_boxes, t_ids = [], []
            for t in tracked:
                tid = int(getattr(t, "track_id"))
                if hasattr(t, "tlbr"):
                    x1, y1, x2, y2 = map(int, t.tlbr)
                elif hasattr(t, "to_tlbr"):
                    x1, y1, x2, y2 = map(int, t.to_tlbr())
                else:
                    continue
                t_ids.append(tid)
                t_boxes.append([x1, y1, x2, y2])
            if t_boxes:
                t_boxes = np.asarray(t_boxes, dtype=int)
                t_ids = np.asarray(t_ids, dtype=int)
                t_scores = scores[: len(t_ids)] if scores is not None else None
                t_cids = cids[: len(t_ids)] if cids is not None else None
                return t_boxes, t_scores, t_cids, t_ids

    return boxes, scores, cids, None


# ──────────────────────────────────────────────────────────────────────────────
# counting + alerts (NOW WITH ALERT MODE)
# ──────────────────────────────────────────────────────────────────────────────
def _update_counts_and_alerts(
    app,
    frame_idx,
    event_time_sec,
    timecode_str,
    clock_str,
    names,
    lines_cfg,
    zones_cfg,
    det_boxes,
    det_confs,
    det_cids,
    det_ids,
    last_anchor,
    line_states,
    line_counts,
    zone_states,
    zone_counts,
    events,
    line_min_gap,
    zone_min_gap,
    anchor_mode,
    ghost_margin,
    alert_enabled,
    selected_class_ids_set,
    alert_freeze_ms,
    alert_when_inside,
    alert_mode,                           # NEW: 1=inside, 0=outside, 2=line
    sound_player: SoundPlayer | None,
    alert_loop: bool,
):
    """
    Updates line/zone counters and triggers sound according to alert_mode:
      - 1 = inside zone (uses loop/single according to alert_loop)
      - 0 = outside zone (uses loop/single according to alert_loop)
      - 2 = line crossing (always single ping with cooldown)
    """
    anchors = [_anchor_from_box(b, anchor_mode, ghost_margin) for b in det_boxes]
    now_ms = int(time.time() * 1000)
    frame_active_ids = set()   # any ID that should fire sound THIS frame

    for (tid, b, s, cid, (cx, cy)) in zip(det_ids, det_boxes, det_confs, det_cids, anchors):
        # class filter
        if selected_class_ids_set and int(cid) not in selected_class_ids_set:
            last_anchor[tid] = (cx, cy)
            continue

        # ───────── Lines
        for li, ln in enumerate(lines_cfg or []):
            st = line_states[li].get(tid, {"last_side": None, "last_frame": -9999})
            prev_c = last_anchor.get(tid, (cx, cy))
            crossed = False
            direction = None

            if "pts" in ln and len(ln["pts"]) >= 2:  # polyline
                pts_line = get_line_pts(ln)
                if st["last_side"] is None:
                    st["last_side"] = polyline_side(pts_line, (cx, cy))
                else:
                    direction = polyline_cross_direction(prev_c, (cx, cy), pts_line)
                    if direction is not None and (frame_idx - st["last_frame"] >= line_min_gap):
                        crossed = True
                st["last_side"] = polyline_side(pts_line, (cx, cy))
            else:  # classic 2-point line
                a = (ln["a"][0], ln["a"][1])
                b2 = (ln["b"][0], ln["b"][1])
                cur_side = line_side(a, b2, (cx, cy))
                prev_side = st["last_side"]
                if prev_side is not None:
                    if segments_intersect(prev_c, (cx, cy), a, b2):
                        if prev_side < 0 and cur_side > 0:
                            direction = "ab"
                        elif prev_side > 0 and cur_side < 0:
                            direction = "ba"
                        if direction is not None and (frame_idx - st["last_frame"] >= line_min_gap):
                            crossed = True
                st["last_side"] = cur_side

            if crossed:
                st["last_frame"] = frame_idx
                line_states[li][tid] = st
                line_counts[li][direction] += 1
                events.append(
                    {
                        "frame": int(frame_idx),
                        "time_sec": float(event_time_sec),
                        "timecode": timecode_str,
                        "clock": clock_str,
                        "track_id": int(tid),
                        "class_id": int(cid),
                        "class_name": (names[cid] if isinstance(names, dict) else names[cid]) if names else str(cid),
                        "event_type": f"line_{direction}",
                        "counter_name": ln.get("name", f"line_{li}"),
                        "conf": float(s),
                        "AB": 1 if direction == "ab" else 0,
                        "BA": 1 if direction == "ba" else 0,
                    }
                )
                # Trigger sound only when in line mode
                if int(alert_mode) == 2:
                    frame_active_ids.add(tid)
            else:
                line_states[li][tid] = st

        # ───────── Zones
        for zi, zn in enumerate(zones_cfg or []):
            sstate = zone_states[zi].get(tid, {"inside": False, "last_change": -9999})
            inside_now = point_in_polygon((cx, cy), zn["pts"])
            if inside_now != sstate["inside"]:
                if frame_idx - sstate["last_change"] >= zone_min_gap:
                    sstate["inside"] = inside_now
                    sstate["last_change"] = frame_idx
                    zone_states[zi][tid] = sstate
                    ev = "zone_in" if inside_now else "zone_out"
                    if inside_now:
                        zone_counts[zi]["in"] += 1
                    else:
                        zone_counts[zi]["out"] += 1
                    events.append(
                        {
                            "frame": int(frame_idx),
                            "time_sec": float(event_time_sec),
                            "timecode": timecode_str,
                            "clock": clock_str,
                            "track_id": int(tid),
                            "class_id": int(cid),
                            "class_name": (names[cid] if isinstance(names, dict) else names[cid]) if names else str(cid),
                            "event_type": ev,
                            "counter_name": zn.get("name", f"zone_{zi}"),
                            "conf": float(s),
                            "AB": 0,
                            "BA": 0,
                        }
                    )
                    # Per-zone/per-class accumulation for HUD (right panel)
                    try:
                        label_name = (names[cid] if isinstance(names, dict) else names[cid]) if names else str(cid)
                    except Exception:
                        label_name = str(cid)

                    # zone-local bucket
                    try:
                        z_buckets = app._zone_class_totals_by_zone
                        bucket = z_buckets[zi]["in" if inside_now else "out"]
                        bucket[label_name] = int(bucket.get(label_name, 0)) + 1
                    except Exception:
                        pass

                    # global sum bucket
                    try:
                        g_bucket = app._zone_class_totals_sum["in" if inside_now else "out"]
                        g_bucket[label_name] = int(g_bucket.get(label_name, 0)) + 1
                    except Exception:
                        pass
                    
                    
                    
                    
                    
                    
            else:
                zone_states[zi][tid] = sstate

            # Sound trigger from zones only when alert_mode != line
            if alert_enabled and int(alert_mode) != 2:
                want_inside = (int(alert_when_inside) == 1)
                cond = (sstate.get("inside", False) is True) if want_inside else (sstate.get("inside", False) is False)
                if cond:
                    frame_active_ids.add(tid)

        last_anchor[tid] = (cx, cy)

    # ───────── Sound dispatch
    if alert_enabled and sound_player:
        loop_enabled = bool(alert_loop) and (int(alert_mode) != 2)
        freeze_ms = int(alert_freeze_ms)

        if loop_enabled:
            # Loop while "active"
            if frame_active_ids:
                if not app._alert_state.get("looping", False):
                    if now_ms - app._alert_state.get("last_ms", 0) >= freeze_ms:
                        sound_player.start_loop()
                        app._alert_state["last_ms"] = now_ms
                        app._alert_state["looping"] = True
                        try: app._log("[ALERT] loop start")
                        except Exception: pass
            else:
                if app._alert_state.get("looping", False):
                    sound_player.stop()
                    app._alert_state["looping"] = False
                    try: app._log("[ALERT] loop stop")
                    except Exception: pass
        else:
            # Single ping with cooldown (used also for line mode)
            if frame_active_ids and now_ms - app._alert_state.get("last_ms", 0) >= freeze_ms:
                sound_player.play_once()
                app._alert_state["last_ms"] = now_ms
                try: app._log("[ALERT] ping")
                except Exception: pass

    return anchors


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def run(app, sources, outp: Path, selected_idx):
    t0 = time.time()
    try:
        vids_dir = ensure_dir(outp / "videos")
        snap_root = ensure_dir(outp / "snapshot")
        ev_dir = ensure_dir(outp / "events")
        summ_dir = ensure_dir(outp / "summary")
        cnt_dir = ensure_dir(outp / "counters")
        heat_dir = ensure_dir(outp / "heatmap")
        ensure_dir(outp / "temp")

        p = getattr(app, "adv_params", {}) or {}
        imgsz = int(p.get("imgsz", VIDEO_PRESETS[DEFAULT_QUALITY]["imgsz"]))
        conf = float(p.get("conf", VIDEO_PRESETS[DEFAULT_QUALITY]["conf"]))
        iou = float(p.get("iou", VIDEO_PRESETS[DEFAULT_QUALITY]["iou"]))
        frame_skip = int(p.get("frame_skip", VIDEO_PRESETS[DEFAULT_QUALITY]["frame_skip"]))
        track_buffer = int(p.get("track_buffer", VIDEO_PRESETS[DEFAULT_QUALITY]["track_buffer"]))
        match_thresh = float(p.get("match_thresh", VIDEO_PRESETS[DEFAULT_QUALITY]["match_thresh"]))
        min_hits = int(p.get("min_hits", VIDEO_PRESETS[DEFAULT_QUALITY]["min_hits"]))
        line_min_gap = int(p.get("line_min_gap", LINE_MIN_GAP_FRAMES_DEFAULT))
        zone_min_gap = int(p.get("zone_min_gap", ZONE_MIN_GAP_FRAMES_DEFAULT))

        device = device_auto_str()

        overlay_mode = getattr(app, "overlay_mode", None).get() if hasattr(app, "overlay_mode") else str(
            p.get("overlay_mode", "centroid")
        )
        anchor_mode = getattr(app, "anchor_mode", None).get() if hasattr(app, "anchor_mode") else str(
            p.get("anchor_mode", "center")
        )
        ghost_margin = (
            int(getattr(app, "ghost_margin", None).get())
            if hasattr(app, "ghost_margin")
            else int(p.get("ghost_margin", 24))
        )

        # HUD scale passthrough
        try:
            hud_scale = float(p.get("hud_scale", 1.0))
            from cv_video_hud import HUD_SCALE_FACTOR
            HUD_SCALE_FACTOR[0] = hud_scale
        except Exception:
            pass

        frame_color = _parse_color(p.get("overlay_frame_color", None))
        frame_thickness = (
            int(p.get("overlay_frame_thickness", 2)) if str(p.get("overlay_frame_thickness", "")).strip() != "" else 2
        )

        # alerts
        alert_enabled = (
            bool(getattr(app, "alert_enabled", None).get()) if hasattr(app, "alert_enabled") else bool(p.get("alert_enabled", False))
        )
        alert_sound_path = ""
        if hasattr(app, "alert_sound"):
            try:
                alert_sound_path = str(app.alert_sound.get()).strip()
            except Exception:
                alert_sound_path = ""
        if not alert_sound_path:
            alert_sound_path = str(p.get("alert_sound", "")).strip()
        alert_loop = bool(getattr(app, "alert_loop", None).get()) if hasattr(app, "alert_loop") else bool(p.get("alert_loop", True))
        if hasattr(app, "alert_freeze_s"):
            try:
                alert_freeze_ms = 1000 * int(app.alert_freeze_s.get())
            except Exception:
                alert_freeze_ms = 1000 * int(p.get("alert_freeze_s", 2))
        else:
            alert_freeze_ms = 1000 * int(p.get("alert_freeze_s", 2))
        alert_when_inside = int(p.get("alert_zone_inside", 1))
        # NEW: derive mode (1=inside, 0=outside, 2=line). Back-compat: fall back to zone_inside.
        alert_mode = int(p.get("alert_mode", 1 if p.get("alert_zone_inside", 1) else 0))

        sound_player = SoundPlayer(alert_sound_path if alert_sound_path else None)

        app._log(
            f"Param: imgsz={imgsz}, conf={conf}, iou={iou}, frame_skip={frame_skip}, "
            f"track_buffer={track_buffer}, match={match_thresh}, hits={min_hits}, device={device}"
        )

        selected_class_ids_set = set(selected_idx or [])  # respected everywhere

        for vi, source in enumerate(sources):
            src_name = (str(source) if not isinstance(source, (int,)) else f"cam_{source}")
            is_stream = _is_stream_source(source)
            app._log(f"\n[Source] {src_name}")

            cap = cv2.VideoCapture(source)
            if not cap or not cap.isOpened():
                app._log(f"[WARN] Cannot open: {src_name}")
                continue

            total_frames = (
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if (not is_stream and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0)
                else None
            )
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 1e-3 else 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            start_perf = time.perf_counter()
            start_epoch = time.time()

            # first frame for file sources (to open editor)
            if is_stream:
                base_stem = _re.sub(r"[^A-Za-z0-9_]+", "_", src_name if isinstance(source, str) else f"cam_{source}")
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, frame_bgr=None, default_cfg_path=default_cfg_path, live_cap=cap)
                result, aborted = editor.run_modal()
            else:
                ok, first_frame = cap.read()
                if not ok or first_frame is None:
                    app._log(f"[ERR] No first frame: {src_name}")
                    cap.release()
                    continue
                base_stem = Path(src_name).stem if isinstance(source, (str, Path)) else f"cam_{source}"
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, frame_bgr=first_frame, default_cfg_path=default_cfg_path, live_cap=None)
                result, aborted = editor.run_modal()

            # allow empty config (no AOI)
            if aborted or result is None:
                app._log("[INFO] Counter configuration cancelled — aborting run.")
                cap.release()
                return

            run_tag = datetime.fromtimestamp(start_epoch).strftime("%Y%m%d_%H%M%S")

            snap_enabled = bool(p.get("snapshot_on_events", False))
            snap_dir = ensure_dir(snap_root / f"{base_stem}_{run_tag}") if snap_enabled else None

            lines_cfg = result.get("lines", []) or []
            zones_cfg = result.get("zones", []) or []
            stride = max(1, int(frame_skip))

            writer, out_path = open_video_writer_collision(vids_dir / f"{base_stem}_annotated.mp4", W, H, fps)
            if not writer or not writer.isOpened():
                app._log(f"[ERR] Cannot open VideoWriter: {src_name}")
                cap.release()
                continue

            tracker = _make_bytetrack(conf, track_buffer, match_thresh, min_hits, fps)

            last_anchor = {}
            line_states = [{} for _ in lines_cfg]
            line_counts = [{"ab": 0, "ba": 0} for _ in lines_cfg]
            zone_states = [{} for _ in zones_cfg]
            zone_counts = [{"in": 0, "out": 0} for _ in zones_cfg]
            events = []
            app._ev_ref = events            # ← expose live events to HUD (per-class lines)
            trails = {} if (getattr(app, "trace_enabled", None).get() if hasattr(app, "trace_enabled") else True) else None
            ev_i_saved = 0
            app._alert_state = {"last_ms": 0, "looping": False}
            # Per-zone per-class totals and global sum (for right HUD)
            app._zone_class_totals_by_zone = [{"in": {}, "out": {}} for _ in zones_cfg]
            app._zone_class_totals_sum = {"in": {}, "out": {}}

            # Global counters per-source
            names_obj = getattr(app, "names", {})
            if isinstance(names_obj, dict):
                max_id = max(names_obj.keys()) if names_obj else -1
                id2name = [""] * (max_id + 1)
                for k, v in names_obj.items():
                    id2name[int(k)] = str(v)
            elif isinstance(names_obj, list):
                id2name = [str(x) for x in names_obj]
            else:
                id2name = []
            stats = StatsAggregator(id2name, selected_ids=selected_class_ids_set or None)

            # HEATMAP: config, init, dynamic decay/window
            heat_alpha = float(p.get("heat_alpha", 0.5))
            heat_sigma = int(p.get("heat_sigma", 8))
            heat_decay = float(p.get("heat_decay", 0.02))  # used only if not window-mode
            heat_use_aoi = bool(p.get("heat_use_aoi", False))
            heat_enabled = bool(p.get("heat_enabled", False))  # accumulate?
            heat_overlay_on = bool(p.get("heat_overlay_on_start", heat_enabled))  # show?
            heat_window_enabled = bool(p.get("heat_window_enabled", False))
            heat_window_minutes = float(p.get("heat_window_minutes", 5.0))
            heat_save_interval_s = int(p.get("heat_save_interval_s", 0))  # 0 = don't save periodically
            # extra tunables for visuals
            heat_gamma = float(p.get("heat_gamma", 1.0))  # 1.0 = linear
            heat_zero_thresh = float(p.get("heat_zero_thresh", 1e-6))  # <= treated as transparent

            last_heat_save_sec = -1.0

            heatmap = DetectionHeatmap(W, H, sigma=heat_sigma, decay=heat_decay)
            if heat_window_enabled and heat_window_minutes > 0 and fps > 0:
                # per-frame decay so that ~1/e weight after window_seconds
                window_s = max(1.0, 60.0 * heat_window_minutes)
                per_frame_decay = 1.0 - math.exp(-1.0 / (fps * window_s))
                heatmap.set_decay(per_frame_decay)
            if heat_use_aoi and zones_cfg:
                try:
                    mask = build_mask_from_zones(zones_cfg, W, H)
                    heatmap.set_mask(mask)
                except Exception:
                    pass

            def _frame_timing(is_stream_local: bool):
                if is_stream_local:
                    sec = time.perf_counter() - start_perf
                    tc = _fmt_timecode(sec)
                    clk = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_epoch + sec))
                    return sec, tc, clk
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    sec = float(pos_ms) / 1000.0
                else:
                    cur_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    sec = float(cur_idx) / float(fps) if (cur_idx and fps > 0) else 0.0
                return sec, _fmt_timecode(sec), ""

            cur_time_sec = 0.0  # updated every frame

            def _handle_frame(frame, frame_idx):
                nonlocal cur_time_sec
                res = app.model(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
                boxes, scores, cids = _parse_ultra_results(res)

                if selected_class_ids_set and boxes.shape[0] > 0:
                    mask_keep = np.isin(cids, np.fromiter(selected_class_ids_set, dtype=int))
                    boxes = boxes[mask_keep]
                    scores = scores[mask_keep]
                    cids = cids[mask_keep]

                boxes, scores, cids, det_ids = _track_update(tracker, boxes, scores, cids)
                if det_ids is None:
                    det_ids = np.arange(1, boxes.shape[0] + 1, dtype=int)

                event_time_sec, timecode_str, clock_str = _frame_timing(is_stream)
                cur_time_sec = float(event_time_sec)
                names = app.names if hasattr(app, "names") else {}

                anchors = _update_counts_and_alerts(
                    app,
                    frame_idx,
                    event_time_sec,
                    timecode_str,
                    clock_str,
                    names,
                    lines_cfg,
                    zones_cfg,
                    boxes,
                    scores,
                    cids,
                    det_ids,
                    last_anchor,
                    line_states,
                    line_counts,
                    zone_states,
                    zone_counts,
                    events,
                    line_min_gap,
                    zone_min_gap,
                    anchor_mode,
                    ghost_margin,
                    alert_enabled,
                    selected_class_ids_set,
                    alert_freeze_ms,
                    alert_when_inside,
                    alert_mode,              # NEW
                    sound_player,
                    alert_loop,
                )

                ov = frame.copy()
                ov = _draw_detections_safe(ov, boxes, scores, cids, det_ids, names, overlay_mode)
                draw_lines_zones(ov, lines_cfg, zones_cfg, frame_color=frame_color, frame_thickness=frame_thickness)

                # trails
                if trails is not None:
                    for tid, a in zip(det_ids, anchors):
                        dq = trails.get(int(tid))
                        if dq is None:
                            dq = deque(maxlen=max(2, int((getattr(app, "trace_len", None).get() if hasattr(app, "trace_len") else 24))))
                            trails[int(tid)] = dq
                        dq.append((int(a[0]), int(a[1])))
                    draw_trails(
                        ov,
                        trails,
                        trace_color=_parse_color(p.get("trace_color", None)),
                        trace_thickness=int(p.get("trace_thickness", 2))
                        if str(p.get("trace_thickness", "")).strip() != ""
                        else 2,
                    )

                # HEATMAP accumulate + optional overlay (transparent zeros)
                try:
                    if heat_enabled and boxes is not None and len(boxes) > 0:
                        cx = ((boxes[:, 0] + boxes[:, 2]) // 2).astype(int)
                        cy = ((boxes[:, 1] + boxes[:, 3]) // 2).astype(int)
                        heatmap.add_points(zip(cx.tolist(), cy.tolist()))
                    if heat_overlay_on:
                        ov = heatmap.render_overlay_masked(
                            ov, alpha=heat_alpha, gamma=heat_gamma, thresh=heat_zero_thresh
                        )
                except Exception:
                    pass

                # Global Now/Max HUD
                try:
                    stats.update_from_cids(cids)
                    now_counts = stats.now_named()
                    max_counts = stats.max_named()
                    if now_counts:
                        draw_run_counters(ov, now_counts, max_counts, anchor="bl", app=app)
                except Exception:
                    pass

                # Existing BR panel
                draw_counts_panel(ov, lines_cfg, line_counts, zones_cfg, zone_counts, anchor="br", app=app)

                # snapshots on new events
                if snap_enabled and snap_dir is not None:
                    nonlocal ev_i_saved
                    total_ev = len(events)
                    if total_ev > ev_i_saved:
                        for idx in range(ev_i_saved, total_ev):
                            ev = events[idx]
                            if is_stream:
                                tlabel = (ev.get("clock") or "").replace(":", "-").replace(" ", "_")
                            else:
                                tlabel = (
                                    ev.get("timecode") or f"{int(float(ev.get('time_sec', 0)) * 1000)}ms"
                                ).replace(":", "-")
                            safe = lambda s: _re.sub(r"[^0-9A-Za-z_\-\.]+", "_", str(s) if s is not None else "")
                            evtype = safe(ev.get("event_type", "event"))
                            cname = safe(ev.get("counter_name", ""))
                            fname = f"{idx+1:06d}_{evtype}_{cname}_f{int(ev.get('frame',0)):06d}_{safe(tlabel)}.jpg"
                            try:
                                cv2.imwrite(str((snap_dir / fname)), ov)
                            except Exception:
                                pass
                        ev_i_saved = total_ev
                return ov

            # first-frame pass for files
            processed = 0
            if not is_stream and "first_frame" in locals() and first_frame is not None:
                if (processed % stride) == 0:
                    ov = _handle_frame(first_frame, processed)
                    _writer_write_safe(writer, _ensure_bgr(ov), first_frame, (W, H), app)
                    _preview(app, ov, processed, fps, total_frames)
                processed += 1

            # live loop
            last_heat_save_sec = -1.0
            while True:
                if app.abort_event.is_set():
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if (processed % stride) != 0:
                    processed += 1
                    continue
                ov = _handle_frame(frame, processed)
                _writer_write_safe(writer, _ensure_bgr(ov), frame, (W, H), app)
                _preview(app, ov, processed, fps, total_frames)

                # periodic heatmap save to /heatmap ONLY
                if heat_enabled and heat_save_interval_s > 0:
                    if (last_heat_save_sec < 0.0) or (cur_time_sec - last_heat_save_sec >= heat_save_interval_s):
                        last_heat_save_sec = cur_time_sec
                        try:
                            hm_path      = heat_dir / f"{base_stem}_{run_tag}_{int(cur_time_sec):06d}s_heatmap.png"
                            hm_rgba_path = heat_dir / f"{base_stem}_{run_tag}_{int(cur_time_sec):06d}s_heatmap_rgba.png"
                            hm_ov_path   = heat_dir / f"{base_stem}_{run_tag}_{int(cur_time_sec):06d}s_heat_overlay.png"

                            # opaque heatmap (compat) + transparent zeros version
                            heatmap.save_png(hm_path, normalize=True)
                            heatmap.save_png_rgba(hm_rgba_path, alpha_scale=1.0, gamma=heat_gamma, thresh=heat_zero_thresh)

                            # overlay preview snapshot (optional)
                            if isinstance(ov, np.ndarray):
                                hm_ov = heatmap.render_overlay_masked(
                                    ov, alpha=heat_alpha, gamma=heat_gamma, thresh=heat_zero_thresh
                                )
                                cv2.imwrite(str(hm_ov_path), hm_ov)

                            app._log(f"[HEAT] periodic save @ {cur_time_sec:.1f}s → {hm_path.name}")
                        except Exception as e:
                            app._log(f"[WARN] periodic heatmap save failed: {e}")

                processed += 1

                # live keyboard (overlay toggle)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                elif k == ord("m"):  # toggle only the overlay; accumulation continues
                    heat_overlay_on = not heat_overlay_on

            try:
                sound_player.stop()
            except Exception:
                pass

            cap.release()
            writer.release()

            # === SAVE RESULTS ===
            ev_df = pd.DataFrame(events)
            ev_path = save_csv_collision(ev_df, ev_dir / f"{base_stem}_{run_tag}_events.csv")
            app._log(f"Saved events: {ev_path}")

            summary = {
                "source": src_name,
                "run_tag": run_tag,
                "frames": int(processed),
                "fps": float(fps),
                "duration_s": float(processed / max(fps, 1e-6)),
                "lines": [{"name": ln["name"], **line_counts[i]} for i, ln in enumerate(lines_cfg)],
                "zones": [{"name": zn["name"], **zone_counts[i]} for i, zn in enumerate(zones_cfg)],
                "advanced": {
                    "trace_color": p.get("trace_color", None),
                    "trace_thickness": int(p.get("trace_thickness", 2)),
                    "overlay_frame_color": p.get("overlay_frame_color", None),
                    "overlay_frame_thickness": int(p.get("overlay_frame_thickness", 2)),
                    "alert_zone_inside": int(p.get("alert_zone_inside", 1)),
                    "alert_mode": int(alert_mode),                     # NEW
                    "alert_sound": alert_sound_path,
                    "alert_loop": alert_loop,
                    "alert_freeze_s": int(alert_freeze_ms / 1000),
                    # heat params
                    "heat_enabled": bool(heat_enabled),
                    "heat_overlay_on_start": bool(p.get("heat_overlay_on_start", heat_enabled)),
                    "heat_alpha": float(heat_alpha),
                    "heat_sigma": int(heat_sigma),
                    "heat_decay": float(heat_decay),
                    "heat_use_aoi": bool(heat_use_aoi),
                    "heat_window_enabled": bool(heat_window_enabled),
                    "heat_window_minutes": float(heat_window_minutes),
                    "heat_save_interval_s": int(heat_save_interval_s),
                    "heat_gamma": float(heat_gamma),
                    "heat_zero_thresh": float(heat_zero_thresh),
                },
            }
            sum_json_path = save_json_collision(summary, summ_dir / f"{base_stem}_{run_tag}_summary.json")
            app._log(f"Saved summary JSON: {sum_json_path}")

            sum_rows = []
            sum_rows.append(
                {
                    "source": src_name,
                    "run_tag": run_tag,
                    "type": "run",
                    "name": "__total__",
                    "frames": int(processed),
                    "fps": float(fps),
                    "duration_s": float(processed / max(fps, 1e-6)),
                    "lines_cfg": len(lines_cfg),
                    "zones_cfg": len(zones_cfg),
                }
            )
            for i, ln in enumerate(lines_cfg):
                sum_rows.append(
                    {
                        "source": src_name,
                        "run_tag": run_tag,
                        "type": "line",
                        "name": ln["name"],
                        "ab": int(line_counts[i]["ab"]),
                        "ba": int(line_counts[i]["ba"]),
                        "total": int(line_counts[i]["ab"] + line_counts[i]["ba"]),
                    }
                )
            for i, zn in enumerate(zones_cfg):
                sum_rows.append(
                    {
                        "source": src_name,
                        "run_tag": run_tag,
                        "type": "zone",
                        "name": zn["name"],
                        "in": int(zone_counts[i]["in"]),
                        "out": int(zone_counts[i]["out"]),
                        "delta": int(zone_counts[i]["in"] - zone_counts[i]["out"]),
                    }
                )
            sum_csv_df = pd.DataFrame(sum_rows)
            sum_csv_path = save_csv_collision(sum_csv_df, summ_dir / f"{base_stem}_{run_tag}_summary.csv")
            app._log(f"Saved summary CSV: {sum_csv_path}")

            # final heatmap save (ONLY when enabled and ONLY to /heatmap)
            if heat_enabled:
                try:
                    hm_path      = heat_dir / f"{base_stem}_{run_tag}_heatmap.png"
                    hm_rgba_path = heat_dir / f"{base_stem}_{run_tag}_heatmap_rgba.png"
                    hm_ov_path   = heat_dir / f"{base_stem}_{run_tag}_heatmap_overlay.png"

                    heatmap.save_png(hm_path, normalize=True)
                    heatmap.save_png_rgba(hm_rgba_path, alpha_scale=1.0, gamma=heat_gamma, thresh=heat_zero_thresh)

                    if "ov" in locals() and isinstance(ov, np.ndarray):
                        hm_ov = heatmap.render_overlay_masked(
                            ov, alpha=heat_alpha, gamma=heat_gamma, thresh=heat_zero_thresh
                        )
                        cv2.imwrite(str(hm_ov_path), hm_ov)

                    app._log(f"Saved heatmap: {hm_path.name}")
                except Exception as e:
                    app._log(f"[WARN] Heatmap save failed: {e}")

        app._set_progress(100.0, "Done.")

    except Exception as e:
        try:
            app._log(f"[ERROR] {e}")
        except Exception:
            print(e)
    finally:
        try:
            app.worker_done.set()
            app.btn_start.config(state="normal")
            app.btn_abort.config(state="disabled")
        except Exception:
            pass
        app._log(f"\nDone in {time.time() - t0:.1f}s")

# cv_video_run.py — uses shared geometry/HUD/sound helpers
from __future__ import annotations
from pathlib import Path
import time, re as _re, threading, subprocess, sys, os, shutil, signal
import cv2, numpy as np, pandas as pd
from collections import deque

from cv_video_gui import CounterEditor
from cv_video_overlay import draw_detections
from cv_video_core import (
    ensure_dir, device_auto_str, open_video_writer_collision,
    save_json_collision, save_csv_collision,
    VIDEO_PRESETS, DEFAULT_QUALITY,
    LINE_MIN_GAP_FRAMES_DEFAULT, ZONE_MIN_GAP_FRAMES_DEFAULT,
)

# NEW: shared helpers
from cv_video_geom import (
    get_line_pts, line_side, segments_intersect, point_in_polygon,
    polyline_side, polyline_cross_direction
)
from cv_video_hud import draw_lines_zones, draw_trails, draw_counts_panel
from cv_video_sound import SoundPlayer

try:
    import supervision as sv
except Exception:
    sv = None

# ---------- utils (kept) ----------
_URL_RE = _re.compile(r'^\s*(rtsp|rtsps|rtmp|http|https)://', flags=_re.I)
def _is_stream_source(src):
    if isinstance(src, (int,)): return True
    if isinstance(src, str) and _URL_RE.match(src): return True
    return False

def _ensure_bgr(img):
    if img is None: return img
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _preview(app, frame_bgr, frame_idx, fps, total_frames):
    try:
        if hasattr(app, "_update_preview"):
            app._update_preview(frame_bgr, frame_idx, fps, total_frames); return
        if hasattr(app, "_show_preview_bgr"):
            app._show_preview_bgr(frame_bgr); return
        if hasattr(app, "update_preview"):
            app.update_preview(frame_bgr, frame_idx, fps, total_frames); return
        if hasattr(app, "show_preview"):
            app.show_preview(frame_bgr); return
    except Exception:
        pass

def _parse_color(val):
    if val is None: return None
    if isinstance(val, (tuple, list)) and len(val) == 3:
        b,g,r = val; return (int(b), int(g), int(r))
    s = str(val).strip()
    if not s or s.lower() == "auto":
        return None
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16); g = int(s[3:5], 16); b = int(s[5:7], 16)
        return (b, g, r)
    try:
        parts = [int(x.strip()) for x in s.replace(";",",").split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    except Exception:
        pass
    return None

def _fmt_timecode(sec: float) -> str:
    if sec < 0: sec = 0.0
    s = int(round(sec))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# ---------- per-frame counting (uses helpers) ----------
def _anchor_from_box(b, mode: str, ghost_margin: int = 0):
    if mode == "bottom":
        return 0.5*(b[0]+b[2]), max(0.0, b[3] - float(ghost_margin))
    return 0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])

def _process_frame_counting(app, frame_idx, fps, names,
                            lines_cfg, zones_cfg,
                            det_boxes, det_confs, det_cids, det_ids,
                            last_anchor, line_states, line_counts, zone_states, zone_counts, events,
                            line_min_gap, zone_min_gap, anchor_mode, ghost_margin,
                            alert_enabled, selected_class_ids_set,
                            alert_freeze_ms, alert_when_inside,
                            event_time_sec, timecode_str, clock_str,
                            sound_player: SoundPlayer | None, alert_loop: bool):
    anchors = [_anchor_from_box(b, anchor_mode, ghost_margin) for b in det_boxes]
    now_ms = int(time.time()*1000)
    frame_active_ids = set()  # IDs that should keep loop playing in this frame

    for (tid, b, s, cid, (cx,cy)) in zip(det_ids, det_boxes, det_confs, det_cids, anchors):
        # --- lines (count once per crossing, supports polyline) ---
        for li, ln in enumerate(lines_cfg or []):
            st = line_states[li].get(tid, {"last_side": None, "last_frame": -9999})
            prev_c = last_anchor.get(tid, (cx,cy))
            crossed = False; direction = None

            if "pts" in ln and len(ln["pts"]) >= 2:
                pts_line = get_line_pts(ln)
                if st["last_side"] is None:
                    st["last_side"] = polyline_side(pts_line, (cx,cy))
                else:
                    direction = polyline_cross_direction(prev_c, (cx,cy), pts_line)
                    if direction is not None and (frame_idx - st["last_frame"] >= line_min_gap):
                        crossed = True
                st["last_side"] = polyline_side(pts_line, (cx,cy))
            else:
                a = (ln["a"][0], ln["a"][1]); b2 = (ln["b"][0], ln["b"][1])
                cur_side = line_side(a, b2, (cx,cy))
                prev_side = st["last_side"]
                if prev_side is not None:
                    if segments_intersect(prev_c, (cx,cy), a, b2):
                        if prev_side < 0 and cur_side > 0: direction = "ab"
                        elif prev_side > 0 and cur_side < 0: direction = "ba"
                        if direction is not None and (frame_idx - st["last_frame"] >= line_min_gap):
                            crossed = True
                st["last_side"] = cur_side

            if crossed:
                st["last_frame"] = frame_idx
                line_states[li][tid] = st
                line_counts[li][direction] += 1
                events.append({
                    "frame": int(frame_idx),
                    "time_sec": float(event_time_sec),
                    "timecode": timecode_str,
                    "clock": clock_str,
                    "track_id": int(tid),
                    "class_id": int(cid),
                    "class_name": (names[cid] if isinstance(names, dict) else names[cid]),
                    "event_type": f"line_{direction}",
                    "counter_name": ln["name"],
                    "conf": float(s)
                })
            else:
                line_states[li][tid] = st

        # --- zones (keep state; drive alert condition) ---
        for zi, zn in enumerate(zones_cfg or []):
            sstate = zone_states[zi].get(tid, {"inside": False, "last_change": -9999})
            inside_now = point_in_polygon((cx,cy), zn["pts"])
            if inside_now != sstate["inside"]:
                if frame_idx - sstate["last_change"] >= zone_min_gap:
                    sstate["inside"] = inside_now
                    sstate["last_change"] = frame_idx
                    zone_states[zi][tid] = sstate
                    ev = "zone_in" if inside_now else "zone_out"
                    if inside_now: zone_counts[zi]["in"] += 1
                    else: zone_counts[zi]["out"] += 1
                    events.append({
                        "frame": int(frame_idx),
                        "time_sec": float(event_time_sec),
                        "timecode": timecode_str,
                        "clock": clock_str,
                        "track_id": int(tid),
                        "class_id": int(cid),
                        "class_name": (names[cid] if isinstance(names, dict) else names[cid]),
                        "event_type": ev,
                        "counter_name": zn["name"],
                        "conf": float(s)
                    })
            else:
                zone_states[zi][tid] = sstate

            # Frame-level sound condition
            if alert_enabled and (cid in selected_class_ids_set):
                want_inside = bool(alert_when_inside)
                cond = (sstate.get("inside", False) is True) if want_inside else (sstate.get("inside", False) is False)
                if cond:
                    frame_active_ids.add(tid)

        last_anchor[tid] = (cx,cy)

    # --- sound: immediate start/stop (loop) or ping with freeze ---
    if alert_enabled and sound_player:
        if alert_loop:
            if frame_active_ids:
                if not app._alert_state.get("looping", False):
                    if now_ms - app._alert_state.get("last_ms", 0) >= int(alert_freeze_ms):
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
            if frame_active_ids and now_ms - app._alert_state.get("last_ms", 0) >= int(alert_freeze_ms):
                sound_player.play_once()
                app._alert_state["last_ms"] = now_ms
                try: app._log("[ALERT] ping")
                except Exception: pass

    return anchors

# ---------- Main (unchanged behavior) ----------
def run(app, sources, outp: Path, selected_idx):
    t0 = time.time()
    try:
        vids_dir = ensure_dir(outp / "videos")
        ev_dir   = ensure_dir(outp / "events")
        summ_dir = ensure_dir(outp / "summary")
        cnt_dir  = ensure_dir(outp / "counters")
        ensure_dir(outp / "temp")

        p = VIDEO_PRESETS.get(int(app.quality.get()), VIDEO_PRESETS.get(DEFAULT_QUALITY)).copy()
        if getattr(app, "advanced_override", False):
            p.update(app.adv_params)
        imgsz = int(p["imgsz"]); conf = float(p["conf"]); iou = float(p["iou"])
        frame_skip = int(p["frame_skip"]); stride = max(1, frame_skip + 1)
        track_buffer = int(p["track_buffer"]); match_thresh = float(p["match_thresh"]); min_hits = int(p["min_hits"])
        line_min_gap = int(p.get("line_min_gap", LINE_MIN_GAP_FRAMES_DEFAULT))
        zone_min_gap = int(p.get("zone_min_gap", ZONE_MIN_GAP_FRAMES_DEFAULT))
        device = device_auto_str()

        names = app.model.names
        id2name = names if isinstance(names, dict) else {i:nm for i,nm in enumerate(names)}
        selected_class_ids_set = set(int(i) for i in selected_idx)

        anchor_mode = getattr(app, "anchor_mode", None).get() if hasattr(app, "anchor_mode") else "bottom"
        overlay_mode = getattr(app, "overlay_mode", None).get() if hasattr(app, "overlay_mode") else "centroid"
        ghost_margin = int(getattr(app, "ghost_margin", None).get() if hasattr(app, "ghost_margin") else 0)

        # TRACE & FRAME (ADVANCED)
        trace_on = getattr(app, "trace_enabled", None).get() if hasattr(app, "trace_enabled") else True
        trace_len = int(getattr(app, "trace_len", None).get() if hasattr(app, "trace_len") else 24)
        trace_color = _parse_color(p.get("trace_color", None))
        trace_thickness = int(p.get("trace_thickness", 2)) if str(p.get("trace_thickness","")).strip() != "" else 2

        frame_color = _parse_color(p.get("overlay_frame_color", None))
        frame_thickness = int(p.get("overlay_frame_thickness", 2)) if str(p.get("overlay_frame_thickness","")).strip() != "" else 2

        # ALERTS
        alert_enabled = bool(getattr(app, "alert_enabled", None).get()) if hasattr(app, "alert_enabled") else bool(p.get("alert_enabled", False))
        alert_sound_path = ""
        if hasattr(app, "alert_sound"):
            try: alert_sound_path = str(app.alert_sound.get()).strip()
            except Exception: alert_sound_path = ""
        if not alert_sound_path:
            alert_sound_path = str(p.get("alert_sound", "")).strip()
        alert_loop = bool(getattr(app, "alert_loop", None).get()) if hasattr(app, "alert_loop") else bool(p.get("alert_loop", True))
        if hasattr(app, "alert_freeze_s"):
            try: alert_freeze_ms = 1000 * int(app.alert_freeze_s.get())
            except Exception:
                alert_freeze_ms = 1000 * int(p.get("alert_freeze_s", 2))
        else:
            alert_freeze_ms = 1000 * int(p.get("alert_freeze_s", 2))
        alert_when_inside = int(p.get("alert_zone_inside", 1))  # 1=in zone, 0=outside

        sound_player = SoundPlayer(alert_sound_path if alert_sound_path else None)

        app._log(f"Param: imgsz={imgsz}, conf={conf}, iou={iou}, frame_skip={frame_skip}, "
                 f"track_buffer={track_buffer}, match={match_thresh}, hits={min_hits}, device={device}")
        tracker_kind = (getattr(app, "tracker_kind", None).get() if hasattr(app, "tracker_kind") else "bytetrack")
        if sv is not None:
            import inspect
        tracker_name = "ByteTrack"  # will be shown later per-source

        for vi, source in enumerate(sources):
            src_name = (str(source) if not isinstance(source, (int,)) else f"cam_{source}")
            app._log(f"\n=== {vi+1}/{len(sources)}: {src_name} ===")

            is_stream = _is_stream_source(source)
            cap = cv2.VideoCapture(source if is_stream or isinstance(source, (int,)) else str(source))
            if not cap or not cap.isOpened():
                app._log(f"[WARN] Cannot open: {src_name}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (not is_stream and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0) else None
            fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps>1e-3 else 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            start_perf = time.perf_counter()
            start_epoch = time.time()

            # --- Counter editor ---
            if is_stream:
                base_stem = _re.sub(r'[^A-Za-z0-9_]+','_', src_name if isinstance(source, str) else f"cam_{source}")
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, frame_bgr=None, default_cfg_path=default_cfg_path, live_cap=cap)
                app.wait_window(editor)
                lines_cfg = editor.lines[:]; zones_cfg = editor.zones[:]
                first_frame = None
            else:
                ok, first_frame = cap.read()
                if not ok or first_frame is None:
                    app._log(f"[ERR] No first frame: {src_name}")
                    cap.release(); continue
                base_stem = (Path(src_name).stem if isinstance(source, (str,Path)) else f"cam_{source}")
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, frame_bgr=first_frame, default_cfg_path=default_cfg_path, live_cap=None)
                app.wait_window(editor)
                lines_cfg = editor.lines[:]; zones_cfg = editor.zones[:]

            if not lines_cfg and not zones_cfg:
                app._log("[WARN] No lines or zones — skipping.")
                cap.release(); continue

            # --- Writer ---
            fps_out = max(1.0, fps / float(stride))
            writer, out_path = open_video_writer_collision(vids_dir / f"{base_stem}_annotated.mp4", W, H, fps_out)
            if not writer or not writer.isOpened():
                app._log(f"[ERR] Cannot open VideoWriter: {src_name}")
                cap.release(); continue

            # Tracker object (ByteTrack / BoT-SORT)
            def _make_bytetrack(conf, track_buffer, match_thresh, min_hits):
                if sv is None: return None
                import inspect
                try:
                    params = inspect.signature(sv.ByteTrack.__init__).parameters
                except Exception:
                    try: return sv.ByteTrack()
                    except Exception: return None
                kwargs = {}
                if "track_thresh" in params: kwargs["track_thresh"] = max(0.05, min(conf, 0.99))
                if "track_buffer" in params: kwargs["track_buffer"] = int(track_buffer)
                if "match_thresh" in params: kwargs["match_thresh"] = float(match_thresh)
                if "min_hits" in params: kwargs["min_hits"] = int(min_hits)
                if "mot20" in params: kwargs["mot20"] = False
                try: return sv.ByteTrack(**kwargs)
                except TypeError:
                    try: return sv.ByteTrack()
                    except Exception: return None

            def _make_botsort(conf, track_buffer, match_thresh, min_hits):
                if sv is None: return None
                import inspect
                BotCls = None
                for nm in ["BoTSORT","BOTSORT","BoTSort"]:
                    BotCls = getattr(sv, nm, None)
                    if BotCls is not None: break
                if BotCls is None: return None
                try:
                    params = inspect.signature(BotCls.__init__).parameters
                except Exception:
                    try: return BotCls()
                    except Exception: return None
                kwargs = {}
                if "track_thresh" in params: kwargs["track_thresh"] = max(0.05, min(conf, 0.99))
                if "track_buffer" in params: kwargs["track_buffer"] = int(track_buffer)
                if "match_thresh" in params: kwargs["match_thresh"] = float(match_thresh)
                if "min_hits" in params: kwargs["min_hits"] = int(min_hits)
                if "mot20" in params: kwargs["mot20"] = False
                try: return BotCls(**kwargs)
                except TypeError:
                    try: return BotCls()
                    except Exception: return None

            tracker_kind = (getattr(app, "tracker_kind", None).get() if hasattr(app, "tracker_kind") else "bytetrack")
            if str(tracker_kind).lower() in ("botsort","bot-sort","bot","bts"):
                tracker = _make_botsort(conf, track_buffer, match_thresh, min_hits) or _make_bytetrack(conf, track_buffer, match_thresh, min_hits)
                tracker_name = "BoT-SORT" if tracker and tracker.__class__.__name__.lower().startswith("bot") else "ByteTrack (fallback)"
            else:
                tracker = _make_bytetrack(conf, track_buffer, match_thresh, min_hits)
                tracker_name = "ByteTrack"

            app._log(
                "Tracker: {tn} | Selected classes: {cls} | Alert={onoff} {mode} {snd}; freeze={fz:.1f}s; zone_mode={zmode} | Sound backends: {bk}".format(
                    tn=tracker_name,
                    cls=", ".join(id2name[i] for i in sorted(selected_class_ids_set)) if selected_class_ids_set else "(none)",
                    onoff=("ON" if alert_enabled else "OFF"),
                    mode="(loop)" if alert_loop else "(ping)",
                    snd=f"(file: {Path(alert_sound_path).name})" if alert_sound_path else "(no file)",
                    fz=alert_freeze_ms/1000.0,
                    zmode=("INSIDE" if alert_when_inside else "OUTSIDE"),
                    bk=(sound_player.describe_backends() if sound_player else "none")
                )
            )

            last_anchor = {}
            line_states = [{} for _ in lines_cfg]
            line_counts = [{"ab":0,"ba":0} for _ in lines_cfg]
            zone_states = [{} for _ in zones_cfg]
            zone_counts = [{"in":0,"out":0} for _ in zones_cfg]
            events = []
            trails = {} if (getattr(app, "trace_enabled", None).get() if hasattr(app, "trace_enabled") else True) else None

            app._alert_state = {"last_ms": 0, "looping": False}

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
                    if cur_idx and cur_idx > 0 and fps > 0:
                        sec = float(cur_idx) / float(fps)
                    else:
                        sec = 0.0
                tc = _fmt_timecode(sec)
                return sec, tc, ""

            def _handle_frame(frame, frame_idx):
                res = app.model(frame, imgsz=imgsz, conf=conf, iou=iou,
                                device=device, classes=list(sorted(selected_class_ids_set)), verbose=False)[0]

                det_boxes, det_confs, det_cids, det_ids = [], [], [], []
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    if tracker is not None and len(xyxy) > 0 and sv is not None:
                        dets = sv.Detections(
                            xyxy=xyxy.astype(np.float32),
                            confidence=confs.astype(np.float32),
                            class_id=cls.astype(np.int32)
                        )
                        dets = tracker.update_with_detections(dets)
                        if dets.tracker_id is not None:
                            det_ids = dets.tracker_id.astype(int).tolist()
                            det_boxes = dets.xyxy.astype(float).tolist()
                            det_confs = dets.confidence.astype(float).tolist()
                            det_cids  = dets.class_id.astype(int).tolist()
                    if not det_ids:
                        det_boxes = xyxy.astype(float).tolist()
                        det_confs = confs.astype(float).tolist()
                        det_cids  = cls.astype(int).tolist()
                        det_ids   = list(range(1, len(det_boxes)+1))

                sec, tc, clk = _frame_timing(is_stream)

                anchors = _process_frame_counting(
                    app, frame_idx, fps, id2name,
                    lines_cfg, zones_cfg,
                    det_boxes, det_confs, det_cids, det_ids,
                    last_anchor, line_states, line_counts, zone_states, zone_counts, events,
                    line_min_gap, zone_min_gap, anchor_mode, ghost_margin,
                    alert_enabled, selected_class_ids_set,
                    alert_freeze_ms, alert_when_inside,
                    sec, tc, clk,
                    sound_player if alert_enabled and sound_player and sound_player.path else None,
                    alert_loop
                )

                ov = frame
                try:
                    draw_detections(ov, det_boxes, det_confs, det_cids, det_ids,
                                    id2name, (overlay_mode or "centroid"),
                                    None, True, anchors)
                except Exception:
                    pass

                draw_lines_zones(ov, lines_cfg, zones_cfg,
                                 frame_color=_parse_color(p.get("overlay_frame_color", None)),
                                 frame_thickness=int(p.get("overlay_frame_thickness", 2)) if str(p.get("overlay_frame_thickness","")).strip() != "" else 2)

                if trails is not None:
                    for tid, a in zip(det_ids, anchors):
                        dq = trails.get(tid)
                        if dq is None:
                            dq = deque(maxlen=max(2, int((getattr(app, "trace_len", None).get() if hasattr(app, "trace_len") else 24))))
                            trails[tid] = dq
                        dq.append((int(a[0]), int(a[1])))

                    draw_trails(ov, trails,
                                trace_color=_parse_color(p.get("trace_color", None)),
                                trace_thickness=int(p.get("trace_thickness", 2)) if str(p.get("trace_thickness","")).strip() != "" else 2)

                draw_counts_panel(ov, lines_cfg, line_counts, zones_cfg, zone_counts, anchor="br", app=app)
                return ov

            processed = 0
            if not is_stream and 'first_frame' in locals() and first_frame is not None:
                if (processed % stride) == 0:
                    ov = _handle_frame(first_frame, processed)
                    ov = _ensure_bgr(ov)
                    writer.write(ov)
                    _preview(app, ov, processed, fps, total_frames)
                processed += 1

            while True:
                if app.abort_event.is_set(): break
                ok, frame = cap.read()
                if not ok or frame is None: break
                if (processed % stride) != 0:
                    processed += 1
                    continue
                ov = _handle_frame(frame, processed)
                ov = _ensure_bgr(ov)
                writer.write(ov)
                _preview(app, ov, processed, fps, total_frames)
                processed += 1

            try:
                if sound_player: sound_player.stop()
            except Exception: pass

            cap.release(); writer.release()

            ev_df = pd.DataFrame(events)
            ev_path = save_csv_collision(ev_df, ev_dir / f"{base_stem}_events.csv")
            app._log(f"Saved events: {ev_path}")

            summary = {
                "source": src_name,
                "frames": int(processed),
                "fps": float(fps),
                "lines": [{"name": ln["name"], **line_counts[i]} for i,ln in enumerate(lines_cfg)],
                "zones": [{"name": zn["name"], **zone_counts[i]} for i,zn in enumerate(zones_cfg)],
                "advanced": {
                    "trace_color": p.get("trace_color", None),
                    "trace_thickness": int(p.get("trace_thickness", 2)),
                    "overlay_frame_color": p.get("overlay_frame_color", None),
                    "overlay_frame_thickness": int(p.get("overlay_frame_thickness", 2)),
                    "alert_zone_inside": int(p.get("alert_zone_inside", 1)),
                    "alert_sound": alert_sound_path,
                    "alert_loop": alert_loop,
                    "alert_freeze_s": int(alert_freeze_ms/1000)
                }
            }
            sum_path = save_json_collision(summary, summ_dir / f"{base_stem}_summary.json")
            app._log(f"Saved summary: {sum_path}")

        app._set_progress(100.0, "Done.")

    except Exception as e:
        try: app._log(f"[ERROR] {e}")
        except Exception: print(e)
    finally:
        try:
            app.worker_done.set()
            app.btn_start.config(state="normal")
            app.btn_abort.config(state="disabled")
        except Exception:
            pass
        app._log(f"\nDone in {time.time()-t0:.1f}s")

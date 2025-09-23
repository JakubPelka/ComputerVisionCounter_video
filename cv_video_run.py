# cv_video_run.py — trace/frame color+thickness (advanced) + zone alert invert (0/1)
from __future__ import annotations
from pathlib import Path
import time, re as _re, threading
import cv2, numpy as np, pandas as pd
from collections import deque

from cv_video_gui import CounterEditor
from cv_video_overlay import draw_detections, draw_counters
from cv_video_core import (
    ensure_dir, device_auto_str, open_video_writer_collision,
    save_json_collision, save_csv_collision,
    VIDEO_PRESETS, DEFAULT_QUALITY,
    LINE_MIN_GAP_FRAMES_DEFAULT, ZONE_MIN_GAP_FRAMES_DEFAULT,
)

try:
    import supervision as sv
except Exception:
    sv = None

# ---------- Beep (Windows only) ----------
try:
    import winsound
    def _beep(freq, dur):
        threading.Thread(target=lambda: winsound.Beep(int(freq), int(dur)), daemon=True).start()
except Exception:
    def _beep(freq, dur):  # no-op poza Windows
        pass

# ---------- Geometry ----------
def line_side(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    return (bx-ax)*(py-ay) - (by-ay)*(px-ax)

def segments_intersect(p1, p2, q1, q2):
    def _orient(a,b,c):
        v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        return 1 if v>0 else (-1 if v<0 else 0)
    def _on(a,b,c):
        return (min(a[0],b[0]) - 1e-6 <= c[0] <= max(a[0],b[0]) + 1e-6 and
                min(a[1],b[1]) - 1e-6 <= c[1] <= max(a[1],b[1]) + 1e-6)
    o1 = _orient(p1,p2,q1); o2 = _orient(p1,p2,q2)
    o3 = _orient(q1,q2,p1); o4 = _orient(q1,q2,p2)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and _on(p1,p2,q1): return True
    if o2 == 0 and _on(p1,p2,q2): return True
    if o3 == 0 and _on(q1,q2,p1): return True
    if o4 == 0 and _on(q1,q2,p2): return True
    return False

def point_in_polygon(p, poly):
    x, y = p; inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1)
        if cond: inside = not inside
    return inside

# ---------- Utils ----------
_URL_RE = _re.compile(r'^\s*(rtsp|rtsps|rtmp|http|https)://', flags=_re.I)
def _is_stream_source(src):
    if isinstance(src, (int,)): return True
    if isinstance(src, str) and _URL_RE.match(src): return True
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
            app._update_preview(frame_bgr, frame_idx, fps, total_frames); return
        if hasattr(app, "_show_preview_bgr"):
            app._show_preview_bgr(frame_bgr); return
        if hasattr(app, "update_preview"):
            app.update_preview(frame_bgr, frame_idx, fps, total_frames); return
        if hasattr(app, "show_preview"):
            app.show_preview(frame_bgr); return
    except Exception:
        pass  # brak preview nie przerywa run

def _parse_color(val):
    """Zwraca kolor w BGR lub None. Akceptuje: None/''/'auto', '#RRGGBB' (RGB), 'B,G,R'."""
    if val is None: return None
    if isinstance(val, (tuple, list)) and len(val) == 3:
        b,g,r = val; return (int(b), int(g), int(r))
    s = str(val).strip()
    if not s or s.lower() == "auto":
        return None
    if s.startswith("#") and len(s) == 7:
        # hex RGB -> BGR
        r = int(s[1:3], 16); g = int(s[3:5], 16); b = int(s[5:7], 16)
        return (b, g, r)
    # B,G,R
    try:
        parts = [int(x.strip()) for x in s.replace(";",",").split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    except Exception:
        pass
    return None

# ---------- Tracker ----------
def _make_bytetrack(conf: float, track_buffer: int, match_thresh: float, min_hits: int):
    if sv is None:
        return None
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
    try:
        return sv.ByteTrack(**kwargs)
    except TypeError:
        try: return sv.ByteTrack()
        except Exception: return None

# ---------- Drawing helpers ----------
def _draw_lines_zones(frame, lines_cfg, zones_cfg, frame_color, frame_thickness):
    """Rysuje same krawędzie (bez etykiet) – by uniknąć duplikatów."""
    if frame_thickness is not None and frame_thickness <= 0:
        return
    th = int(frame_thickness or 2)
    # Linie
    for ln in lines_cfg or []:
        a = tuple(map(int, ln["a"])); b = tuple(map(int, ln["b"]))
        col = frame_color if frame_color is not None else tuple(int(c) for c in ln.get("color", (0,255,255)))
        cv2.line(frame, a, b, col, th, cv2.LINE_AA)
    # Strefy
    for zn in zones_cfg or []:
        pts = np.array(zn["pts"], dtype=np.int32)
        if len(pts) >= 3:
            col = frame_color if frame_color is not None else tuple(int(c) for c in zn.get("color", (0,200,255)))
            cv2.polylines(frame, [pts], True, col, th, cv2.LINE_AA)

def _draw_trails(frame, trails, trace_color, trace_thickness):
    """Prosty renderer śladów. trace_thickness=0 → brak śladów."""
    th = int(trace_thickness if trace_thickness is not None else 2)
    if th <= 0 or not trails:
        return
    for tid, dq in trails.items():
        if len(dq) < 2:
            continue
        if trace_color is None:
            # stabilny „auto” kolor na bazie id
            r = (37 * tid) % 256; g = (91 * tid) % 256; b = (157 * tid) % 256
            col = (int(b), int(g), int(r))
        else:
            col = trace_color
        pts = np.array(dq, dtype=np.int32)
        cv2.polylines(frame, [pts], False, col, th, cv2.LINE_AA)

def _safe_draw_counters(frame, lines_cfg, line_counts, zones_cfg, zone_counts):
    ok = False
    try:
        draw_counters(frame, lines_cfg, line_counts, zones_cfg, zone_counts, None)
        ok = True
    except TypeError:
        try:
            draw_counters(frame, lines_cfg, line_counts, zones_cfg, zone_counts)
            ok = True
        except Exception:
            ok = False
    if ok:
        return
    # Fallback HUD
    x, y = 12, 24
    cv2.rectangle(frame, (6,6), (420, max(36, y + 18*(len(lines_cfg)+len(zones_cfg))+12)), (0,0,0), -1)
    cv2.putText(frame, "Counters", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    y += 24
    for i, ln in enumerate(lines_cfg or []):
        txt = f"[L{i+1}:{ln.get('name','')}] AB:{line_counts[i]['ab']}  BA:{line_counts[i]['ba']}"
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,255,200), 2, cv2.LINE_AA)
        y += 18
    for i, zn in enumerate(zones_cfg or []):
        txt = f"[Z{i+1}:{zn.get('name','')}] IN:{zone_counts[i]['in']}  OUT:{zone_counts[i]['out']}"
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,220,255), 2, cv2.LINE_AA)
        y += 18

# ---------- Anchors & events ----------
def _anchor_from_box(b, mode: str):
    if mode == "bottom": return 0.5*(b[0]+b[2]), b[3]
    return 0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])

def _process_frame_counting(app, frame_idx, fps, names,
                            lines_cfg, zones_cfg,
                            det_boxes, det_confs, det_cids, det_ids,
                            last_anchor, line_states, line_counts, zone_states, zone_counts, events,
                            line_min_gap, anchor_mode,
                            alert_enabled, alert_classes_set, alert_freq, alert_dur,
                            alert_state, alert_freeze_ms,
                            alert_when_inside):
    anchors = [_anchor_from_box(b, anchor_mode) for b in det_boxes]

    for (tid, b, s, cid, (cx,cy)) in zip(det_ids, det_boxes, det_confs, det_cids, anchors):
        # Linie: pojedynczy beep przy przekroczeniu
        for li, ln in enumerate(lines_cfg or []):
            a = (ln["a"][0], ln["a"][1]); b2 = (ln["b"][0], ln["b"][1])
            st = line_states[li].get(tid, {"last_side": None, "last_frame": -9999})
            prev_side = st["last_side"]; cur_side = line_side(a, b2, (cx,cy))
            crossed = False; direction = None
            if prev_side is not None:
                prev_c = last_anchor.get(tid, (cx,cy))
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
                    "time_sec": float(frame_idx / max(1.0, fps)),
                    "track_id": int(tid),
                    "class_id": int(cid),
                    "class_name": (names[cid] if isinstance(names, dict) else names[cid]),
                    "event_type": f"line_{direction}",
                    "counter_name": ln["name"],
                    "conf": float(s)
                })
                if alert_enabled:
                    cname = (names[cid] if isinstance(names, dict) else names[cid]).lower()
                    if (not alert_classes_set) or (cname in alert_classes_set):
                        _beep(alert_freq, alert_dur)
            else:
                line_states[li][tid] = st

        # Strefy: ciągły beep wg trybu (inside/outside)
        for zi, zn in enumerate(zones_cfg or []):
            sstate = zone_states[zi].get(tid, {"inside": False, "last_change": -9999})
            inside_now = point_in_polygon((cx,cy), zn["pts"])
            if inside_now != sstate["inside"]:
                if frame_idx - sstate["last_change"] >= line_min_gap:
                    sstate["inside"] = inside_now
                    sstate["last_change"] = frame_idx
                    zone_states[zi][tid] = sstate
                    ev = "zone_in" if inside_now else "zone_out"
                    if inside_now: zone_counts[zi]["in"] += 1
                    else: zone_counts[zi]["out"] += 1
                    events.append({
                        "frame": int(frame_idx),
                        "time_sec": float(frame_idx / max(1.0, fps)),
                        "track_id": int(tid),
                        "class_id": int(cid),
                        "class_name": (names[cid] if isinstance(names, dict) else names[cid]),
                        "event_type": ev,
                        "counter_name": zn["name"],
                        "conf": float(s)
                    })
            else:
                zone_states[zi][tid] = sstate

            # Beep ciągły wg trybu
            if alert_enabled:
                want_inside = bool(alert_when_inside)  # 1=inside, 0=outside
                cond = (sstate.get("inside", False) is True) if want_inside else (sstate.get("inside", False) is False)
                if cond:
                    cname = (names[cid] if isinstance(names, dict) else names[cid]).lower()
                    if (not alert_classes_set) or (cname in alert_classes_set):
                        now_ms = int(time.time()*1000)
                        if now_ms - alert_state.get("last_ms", 0) >= int(alert_freeze_ms):
                            _beep(alert_freq, alert_dur)
                            alert_state["last_ms"] = now_ms

        last_anchor[tid] = (cx,cy)

    return anchors

# ---------- Main ----------
def run(app, sources, outp: Path, selected_idx):
    t0 = time.time()
    try:
        vids_dir = ensure_dir(outp / "videos")
        ev_dir   = ensure_dir(outp / "events")
        summ_dir = ensure_dir(outp / "summary")
        cnt_dir  = ensure_dir(outp / "counters")
        ensure_dir(outp / "temp")

        p = VIDEO_PRESETS.get(int(app.quality.get()), DEFAULT_QUALITY).copy()
        if getattr(app, "advanced_override", False):
            p.update(app.adv_params)

        imgsz = int(p["imgsz"]); conf = float(p["conf"]); iou = float(p["iou"])
        frame_skip = int(p["frame_skip"]); stride = max(1, frame_skip + 1)
        track_buffer = int(p["track_buffer"]); match_thresh = float(p["match_thresh"]); min_hits = int(p["min_hits"])
        line_min_gap = int(p.get("line_min_gap", LINE_MIN_GAP_FRAMES_DEFAULT))
        device = device_auto_str()

        names = app.model.names
        id2name = names if isinstance(names, dict) else {i:nm for i,nm in enumerate(names)}
        select_names = [id2name[i] for i in selected_idx]
        anchor_mode = getattr(app, "anchor_mode", None).get() if hasattr(app, "anchor_mode") else "bottom"

        # --- TRACE & FRAME (ADVANCED) ---
        trace_on = getattr(app, "trace_enabled", None).get() if hasattr(app, "trace_enabled") else True
        trace_len = int(getattr(app, "trace_len", None).get() if hasattr(app, "trace_len") else 24)
        trace_color = _parse_color(p.get("trace_color", None))
        trace_thickness = int(p.get("trace_thickness", 2)) if str(p.get("trace_thickness","")).strip() != "" else 2

        frame_color = _parse_color(p.get("overlay_frame_color", None))
        frame_thickness = int(p.get("overlay_frame_thickness", 2)) if str(p.get("overlay_frame_thickness","")).strip() != "" else 2

        # --- ALERTS ---
        alert_enabled = getattr(app, "alert_enabled", None).get() if hasattr(app, "alert_enabled") else False
        raw_classes = (getattr(app, "alert_classes", None).get() if hasattr(app, "alert_classes") else "person")
        alert_classes_set = set([c.strip().lower() for c in raw_classes.split(",") if c.strip()])
        alert_freq = int(getattr(app, "alert_freq", None).get() if hasattr(app, "alert_freq") else 880)
        alert_dur  = int(getattr(app, "alert_dur", None).get() if hasattr(app, "alert_dur") else 180)
        alert_freeze_ms = int(getattr(app, "alert_freeze", None).get() if hasattr(app, "alert_freeze") else 1500)
        alert_when_inside = int(p.get("alert_zone_inside", 1))  # 1=in zone (default), 0=outside

        app._log(
            "Param: imgsz=%s, conf=%s, iou=%s, frame_skip=%s, track_buffer=%s, match=%s, hits=%s, device=%s" %
            (imgsz, conf, iou, frame_skip, track_buffer, match_thresh, min_hits, device)
        )
        app._log(
            "Tracker: bytetrack | Klasy: %s | Alert=%s (%s, freeze=%sms, zone_mode=%s)" %
            (', '.join(select_names),
             'ON' if alert_enabled else 'OFF',
             ', '.join(sorted(alert_classes_set)) if alert_classes_set else '*',
             alert_freeze_ms,
             'INSIDE' if alert_when_inside else 'OUTSIDE')
        )

        for vi, source in enumerate(sources):
            src_name = (str(source) if not isinstance(source, (int,)) else f"cam_{source}")
            app._log(f"\n=== {vi+1}/{len(sources)}: {src_name} ===")

            is_stream = _is_stream_source(source)
            cap = cv2.VideoCapture(source if is_stream or isinstance(source, (int,)) else str(source))
            if not cap or not cap.isOpened():
                app._log(f"[WARN] Nie można otworzyć: {src_name}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (not is_stream and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0) else None
            fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps>1e-3 else 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            # --- Edytor liczników ---
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
                    app._log(f"[ERR] Brak pierwszej klatki: {src_name}")
                    cap.release(); continue
                base_stem = (Path(src_name).stem if isinstance(source, (str,Path)) else f"cam_{source}")
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, frame_bgr=first_frame, default_cfg_path=default_cfg_path, live_cap=None)
                app.wait_window(editor)
                lines_cfg = editor.lines[:]; zones_cfg = editor.zones[:]

            if not lines_cfg and not zones_cfg:
                app._log("[WARN] Brak linii i stref — pomijam.")
                cap.release(); continue

            # --- Writer ---
            fps_out = max(1.0, fps / float(stride))
            writer, out_path = open_video_writer_collision(vids_dir / f"{base_stem}_annotated.mp4", W, H, fps_out)
            if not writer or not writer.isOpened():
                app._log(f"[ERR] Nie można otworzyć VideoWriter: {src_name}")
                cap.release(); continue

            tracker = _make_bytetrack(conf, track_buffer, match_thresh, min_hits)

            last_anchor = {}
            line_states = [{ } for _ in lines_cfg]
            line_counts = [{"ab":0,"ba":0} for _ in lines_cfg]
            zone_states = [{ } for _ in zones_cfg]
            zone_counts = [{"in":0,"out":0} for _ in zones_cfg]
            events = []
            alert_state = {"last_ms": 0}  # globalny cooldown beep
            trails = {} if trace_on else None  # tid -> deque[(x,y)]

            def _handle_frame(frame, frame_idx):
                # Inference
                res = app.model(frame, imgsz=imgsz, conf=conf, iou=iou,
                                device=device, classes=selected_idx, verbose=False)[0]

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

                # Liczenie + alerty
                anchors = _process_frame_counting(
                    app, frame_idx, fps, id2name,
                    lines_cfg, zones_cfg,
                    det_boxes, det_confs, det_cids, det_ids,
                    last_anchor, line_states, line_counts, zone_states, zone_counts, events,
                    line_min_gap, anchor_mode,
                    alert_enabled, alert_classes_set, alert_freq, alert_dur,
                    alert_state, alert_freeze_ms,
                    alert_when_inside
                )

                # Aktualizacja śladów
                if trails is not None:
                    for tid, a in zip(det_ids, anchors):
                        dq = trails.get(tid)
                        if dq is None:
                            dq = deque(maxlen=max(2, int(trace_len)))
                            trails[tid] = dq
                        dq.append((int(a[0]), int(a[1])))

                # Rysunek in-place (preview + zapis)
                overlay = frame
                # pudełka/ID (zgodne z różnymi sygnaturami)
                try:
                    draw_detections(overlay, det_boxes, det_confs, det_cids, det_ids, id2name, lines_cfg, zones_cfg, None)
                except TypeError:
                    try:
                        draw_detections(overlay, det_boxes, det_confs, det_cids, det_ids, id2name)
                    except Exception:
                        pass
                except Exception:
                    pass

                # Linie/strefy (bez etykiet, nadpisywalny kolor/grubość)
                _draw_lines_zones(overlay, lines_cfg, zones_cfg, frame_color, frame_thickness)
                # Ślady (nadpisywalny kolor/grubość)
                _draw_trails(overlay, trails, trace_color, trace_thickness)
                # Liczniki (fallback HUD, jeśli Twoja funkcja ma inną sygnaturę)
                _safe_draw_counters(overlay, lines_cfg, line_counts, zones_cfg, zone_counts)

                return overlay

            processed = 0
            if not is_stream and 'first_frame' in locals() and first_frame is not None:
                if (processed % stride) == 0:
                    ov = _handle_frame(first_frame, processed)
                    ov = _ensure_bgr(ov)
                    writer.write(ov)
                    _preview(app, ov, processed, fps, total_frames)
                processed += 1

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
                ov = _ensure_bgr(ov)
                writer.write(ov)
                _preview(app, ov, processed, fps, total_frames)
                processed += 1

            cap.release()
            writer.release()

            # Zapis zdarzeń + podsumowanie
            ev_df = pd.DataFrame(events)
            ev_path = save_csv_collision(ev_df, ev_dir / f"{base_stem}_events.csv")
            app._log(f"Zapisano events: {ev_path}")

            summary = {
                "source": src_name,
                "frames": int(processed),
                "fps": float(fps),
                "lines": [{"name": ln["name"], **line_counts[i]} for i,ln in enumerate(lines_cfg)],
                "zones": [{"name": zn["name"], **zone_counts[i]} for i,zn in enumerate(zones_cfg)],
                "advanced": {
                    "trace_color": p.get("trace_color", None),
                    "trace_thickness": trace_thickness,
                    "overlay_frame_color": p.get("overlay_frame_color", None),
                    "overlay_frame_thickness": frame_thickness,
                    "alert_zone_inside": alert_when_inside
                }
            }
            sum_path = save_json_collision(summary, summ_dir / f"{base_stem}_summary.json")
            app._log(f"Zapisano summary: {sum_path}")

        app._set_progress(100.0, "Gotowe.")

    except Exception as e:
        try: app._log(f"[BŁĄD] {e}")
        except Exception: print(e)
    finally:
        try:
            app.worker_done.set()
            app.btn_start.config(state="normal")
            app.btn_abort.config(state="disabled")
        except Exception:
            pass
        app._log(f"\nDone in {time.time()-t0:.1f}s")

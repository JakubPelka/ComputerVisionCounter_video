# cv_video_run.py — unified loop for files & streams + LIVE editor + ghost + zone alerts
from __future__ import annotations
from pathlib import Path
import time, re as _re, threading
import cv2, numpy as np, pandas as pd

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

# ---- Beep (Windows) ----
try:
    import winsound
    def _beep(freq, dur):
        threading.Thread(target=lambda: winsound.Beep(int(freq), int(dur)), daemon=True).start()
except Exception:
    def _beep(freq, dur):  # no-op
        pass

# ---- Geometry helpers ----
def line_side(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
def segments_intersect(p1, p2, q1, q2):
    def orient(a,b,c):
        v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if v > 0: return 1
        if v < 0: return -1
        return 0
    def on_seg(a,b,c):
        return (min(a[0],b[0]) - 1e-6 <= c[0] <= max(a[0],b[0]) + 1e-6 and
                min(a[1],b[1]) - 1e-6 <= c[1] <= max(a[1],b[1]) + 1e-6)
    o1 = orient(p1,p2,q1); o2 = orient(p1,p2,q2)
    o3 = orient(q1,q2,p1); o4 = orient(q1,q2,p2)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_seg(p1,p2,q1): return True
    if o2 == 0 and on_seg(p1,p2,q2): return True
    if o3 == 0 and on_seg(q1,q2,p1): return True
    if o4 == 0 and on_seg(q1,q2,p2): return True
    return False
def point_in_polygon(pt, poly):
    poly_np = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, pt, False) >= 0
def dist_point_to_segment(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    ab2 = abx*abx + aby*aby
    if ab2 <= 1e-9: return float(np.hypot(px-ax, py-ay))
    t = max(0.0, min(1.0, (apx*abx + apy*aby)/ab2))
    cx, cy = ax + t*abx, ay + t*aby
    return float(np.hypot(px-cx, py-cy))

# ---- ByteTrack factory (kompat) ----
def _make_bytetrack(conf: float, track_buffer: int, match_thresh: float, min_hits: int):
    if sv is None:
        return None
    import inspect
    try:
        params = inspect.signature(sv.ByteTrack.__init__).parameters
    except Exception:
        try:
            return sv.ByteTrack()
        except Exception:
            return None
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

# ---- Anchor + counting ----
def _anchor_from_box(b, mode: str):
    if mode == "bottom": return 0.5*(b[0]+b[2]), b[3]
    return 0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])

def _process_frame_counting(app, frame_idx, fps, names,
                            lines_cfg, zones_cfg,
                            det_boxes, det_confs, det_cids, det_ids,
                            last_anchor, line_states, line_counts, zone_states, zone_counts, events,
                            line_min_gap, anchor_mode,
                            alert_enabled, alert_classes_set, alert_freq, alert_dur,
                            alert_state, alert_freeze_ms):
    anchors = [_anchor_from_box(b, anchor_mode) for b in det_boxes]

    for (tid, b, s, cid, (cx,cy)) in zip(det_ids, det_boxes, det_confs, det_cids, anchors):
        # Linie
        for li, ln in enumerate(lines_cfg):
            a = (ln["a"][0], ln["a"][1]); b2 = (ln["b"][0], ln["b"][1])
            st = line_states[li].get(tid, {"last_side": None, "last_frame": -9999})
            prev_side = st["last_side"]
            cur_side = line_side(a, b2, (cx,cy))
            crossed = False; direction = None
            if prev_side is not None:
                prev_c = last_anchor.get(tid, (cx,cy))
                if segments_intersect(prev_c, (cx,cy), a, b2):
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
            else:
                line_states[li][tid] = st

        # Strefy + ALERT
        for zi, zn in enumerate(zones_cfg):
            sstate = zone_states[zi].get(tid, {"inside": False, "last_change": -9999})
            inside_now = point_in_polygon((cx,cy), zn["pts"])
            if inside_now != sstate["inside"]:
                if frame_idx - sstate["last_change"] >= line_min_gap:
                    sstate["inside"] = inside_now
                    sstate["last_change"] = frame_idx
                    zone_states[zi][tid] = sstate
                    ev = "zone_in" if inside_now else "zone_out"
                    if inside_now:
                        zone_counts[zi]["in"] += 1
                        if alert_enabled:
                            cname = (names[cid] if isinstance(names, dict) else names[cid]).lower()
                            if cname in alert_classes_set:
                                now_ms = int(time.time()*1000)
                                if now_ms - alert_state["last_ms"] >= int(alert_freeze_ms):
                                    _beep(alert_freq, alert_dur)
                                    alert_state["last_ms"] = now_ms
                    else:
                        zone_counts[zi]["out"] += 1
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

        last_anchor[tid] = (cx,cy)

    return anchors

def run(app, sources, outp: Path, selected_idx):
    t0 = time.time()
    try:
        vids_dir = ensure_dir(outp / "videos")
        ev_dir   = ensure_dir(outp / "events")
        summ_dir = ensure_dir(outp / "summary")
        cnt_dir  = ensure_dir(outp / "counters")
        temp_dir = ensure_dir(outp / "temp")

        p = VIDEO_PRESETS.get(int(app.quality.get()), DEFAULT_QUALITY).copy()
        if app.advanced_override:
            p.update(app.adv_params)
        imgsz = int(p["imgsz"]); conf = float(p["conf"]); iou = float(p["iou"])
        frame_skip = int(p["frame_skip"]); stride = max(1, frame_skip + 1)
        track_buffer = int(p["track_buffer"]); match_thresh = float(p["match_thresh"]); min_hits = int(p["min_hits"])
        line_min_gap = int(p.get("line_min_gap", LINE_MIN_GAP_FRAMES_DEFAULT))
        device = device_auto_str()

        id2name = app.model.names if isinstance(app.model.names, dict) else {i:nm for i,nm in enumerate(app.model.names)}
        select_names = [id2name[i] for i in selected_idx]
        anchor_mode = getattr(app, "anchor_mode", None).get() if hasattr(app, "anchor_mode") else "bottom"
        trace_on = getattr(app, "trace_enabled", None).get() if hasattr(app, "trace_enabled") else True
        trace_len = int(getattr(app, "trace_len", None).get() if hasattr(app, "trace_len") else 24)
        ghost_margin = int(getattr(app, "ghost_margin", None).get() if hasattr(app, "ghost_margin") else 12)

        alert_enabled = getattr(app, "alert_enabled", None).get() if hasattr(app, "alert_enabled") else False
        raw_classes = (getattr(app, "alert_classes", None).get() if hasattr(app, "alert_classes") else "person")
        alert_classes_set = set([c.strip().lower() for c in raw_classes.split(",") if c.strip()])
        alert_freq = int(getattr(app, "alert_freq", None).get() if hasattr(app, "alert_freq") else 880)
        alert_dur  = int(getattr(app, "alert_dur", None).get() if hasattr(app, "alert_dur") else 180)
        alert_freeze_ms = int(getattr(app, "alert_freeze", None).get() if hasattr(app, "alert_freeze") else 1500)

        app._log(f"Param: imgsz={imgsz}, conf={conf}, iou={iou}, frame_skip={frame_skip} (stride={stride}), buf={track_buffer}, match={match_thresh}, hits={min_hits}, device={device}")
        app._log(f"Tracker: bytetrack | Klasy: {', '.join(select_names)} | Anchor: {anchor_mode} | Trace: {trace_on} (len={trace_len}) | Ghost: {ghost_margin}px | Alert: {alert_enabled} ({', '.join(sorted(alert_classes_set))}, freeze={alert_freeze_ms}ms)")

        for vi, source in enumerate(sources):
            if app.abort_event.is_set(): break
            src_name = (str(source) if not isinstance(source, Path) else source.name)
            app._log(f"► Źródło {vi+1}/{len(sources)}: {src_name}")
            is_stream = not isinstance(source, Path)

            cap = cv2.VideoCapture(source if is_stream else str(source))
            if not cap.isOpened():
                app._log(f"[WARN] Nie można otworzyć: {src_name}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
            fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps>1e-3 else 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            if is_stream:
                # === TRYB LIVE: edytor pracuje na tym samym cap (auto-pauza przy rysowaniu) ===
                base_stem = _re.sub(r'[^A-Za-z0-9_]+','_', src_name if isinstance(source, str) else f"cam_{source}")
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, frame_bgr=None, default_cfg_path=default_cfg_path, live_cap=cap)
                app.wait_window(editor)
                lines_cfg = editor.lines[:]
                zones_cfg = editor.zones[:]
                first_ready = True   # nie przetwarzamy oddzielnie „pierwszej klatki”
            else:
                # === Plik: zamrożona pierwsza klatka ===
                ret, first_frame = cap.read()
                if not ret or first_frame is None:
                    app._log(f"[WARN] Brak pierwszej klatki: {src_name}")
                    cap.release(); continue
                base_stem = (source.stem if isinstance(source, Path) else _re.sub(r'[^A-Za-z0-9_]+','_', src_name))
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, first_frame, default_cfg_path=default_cfg_path, live_cap=None)
                app.wait_window(editor)
                lines_cfg = editor.lines[:]
                zones_cfg = editor.zones[:]
                first_ready = False  # przetworzymy tę klatkę niżej

            fps_out = max(1.0, fps / float(stride))
            writer, out_video_path = open_video_writer_collision(vids_dir / f"{base_stem}_annotated.mp4", W, H, fps_out)
            if not writer or not writer.isOpened():
                app._log(f"[ERR] Nie można otworzyć VideoWriter dla: {src_name}")
                cap.release(); continue

            tracker = _make_bytetrack(conf, track_buffer, match_thresh, min_hits)

            last_anchor = {}
            line_states = [{ } for _ in lines_cfg]
            line_counts = [{"ab":0,"ba":0} for _ in lines_cfg]
            zone_states = [{ } for _ in zones_cfg]
            zone_counts = [{"in":0,"out":0} for _ in zones_cfg]
            events = []

            from collections import deque
            trails = {} if trace_on else None
            missed = {}
            last2 = {}
            last_class = {}
            last_conf  = {}
            RESET_MISSED = max(3, stride * 2)

            alert_state = {"last_ms": 0}  # globalny cooldown

            def _handle_frame(frame, frame_idx):
                nonlocal tracker, trails, missed
                res = app.model(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, classes=selected_idx, verbose=False)[0]

                det_boxes, det_confs, det_cids, det_ids = [], [], [], []
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    if tracker is not None and len(xyxy) > 0 and sv is not None:
                        dets = sv.Detections(xyxy=xyxy.astype(np.float32),
                                             confidence=confs.astype(np.float32),
                                             class_id=cls.astype(int))
                        dets = tracker.update_with_detections(dets)
                        tids = dets.tracker_id if dets.tracker_id is not None else np.full(len(dets), -1)
                        for b, s, c, tid in zip(dets.xyxy, dets.confidence, dets.class_id, tids):
                            tid = int(tid)
                            if tid < 0: continue
                            det_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                            det_confs.append(float(s)); det_cids.append(int(c)); det_ids.append(tid)
                            last_class[tid] = int(c); last_conf[tid] = float(s)
                    else:
                        for i,(b,s,c) in enumerate(zip(xyxy, confs, cls)):
                            det_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                            det_confs.append(float(s)); det_cids.append(int(c)); det_ids.append(int(i))
                            last_class[int(i)] = int(c); last_conf[int(i)] = float(s)

                anchors = _process_frame_counting(
                    app, frame_idx, fps, app.names,
                    lines_cfg, zones_cfg,
                    det_boxes, det_confs, det_cids, det_ids,
                    last_anchor, line_states, line_counts, zone_states, zone_counts, events,
                    line_min_gap, anchor_mode,
                    alert_enabled, alert_classes_set, alert_freq, alert_dur,
                    alert_state, alert_freeze_ms
                )

                # ślady / last-two
                present = set(det_ids)
                for tid, b, ap in zip(det_ids, det_boxes, anchors):
                    dq2 = last2.get(tid)
                    if dq2 is None: dq2 = deque(maxlen=2); last2[tid] = dq2
                    dq2.append((ap[0], ap[1]))
                    if trails is not None:
                        dqt = trails.get(tid)
                        if dqt is None:
                            dqt = deque(maxlen=max(1, int(trace_len)))
                            trails[tid] = dqt
                        dqt.append((ap[0], ap[1]))
                    missed[tid] = 0
                for tid in list(missed.keys()):
                    if tid not in present: missed[tid] += 1
                for tid, m in list(missed.items()):
                    if m == RESET_MISSED and ghost_margin > 0:
                        for li, ln in enumerate(lines_cfg):
                            a = (ln["a"][0], ln["a"][1]); b2 = (ln["b"][0], ln["b"][1])
                            hist = last2.get(tid)
                            if not hist or len(hist) < 2: continue
                            p0, p1 = hist[0], hist[1]
                            d = dist_point_to_segment(a, b2, p1)
                            if d > ghost_margin: continue
                            vx, vy = p1[0]-p0[0], p1[1]-p0[1]
                            p_pred = (p1[0] + 2*vx, p1[1] + 2*vy)
                            s1 = line_side(a, b2, p1); sp = line_side(a, b2, p_pred)
                            if s1 != 0 and s1*sp < 0:
                                direction = "ab" if s1 < 0 and sp > 0 else ("ba" if s1 > 0 and sp < 0 else None)
                                if direction:
                                    st = line_states[li].get(tid, {"last_side": None, "last_frame": -9999})
                                    if frame_idx - st["last_frame"] >= line_min_gap:
                                        line_counts[li][direction] += 1
                                        st["last_frame"] = frame_idx; st["last_side"] = sp
                                        line_states[li][tid] = st
                                        cid = last_class.get(tid, -1); sconf = last_conf.get(tid, float("nan"))
                                        events.append({
                                            "frame": int(frame_idx),
                                            "time_sec": float(frame_idx / max(1.0, fps)),
                                            "track_id": int(tid),
                                            "class_id": int(cid),
                                            "class_name": (app.names[cid] if cid >=0 else "unknown"),
                                            "event_type": f"line_{direction}_ghost",
                                            "counter_name": ln["name"],
                                            "conf": float(sconf)
                                        })
                    if m >= RESET_MISSED:
                        for zi in range(len(zones_cfg)):
                            zone_states[zi][tid] = {"inside": False, "last_change": frame_idx}
                        for li in range(len(lines_cfg)):
                            line_states[li][tid] = {"last_side": None, "last_frame": -9999}
                        if tid in last_anchor: del last_anchor[tid]
                        if tid in last2: del last2[tid]
                        if trails is not None and tid in trails: del trails[tid]
                        missed[tid] = 0

                draw_detections(frame, det_boxes, det_confs, det_cids, det_ids,
                                app.names, app.overlay_mode.get(),
                                polygons=None,
                                show_anchor=(app.overlay_mode.get()=="centroid"),
                                anchor_points=anchors)
                draw_counters(frame, lines_cfg, line_counts, zones_cfg, zone_counts,
                              trails if trails is not None else None)
                writer.write(frame)
                app._show_preview_bgr(frame)

            # „pierwsza klatka” tylko dla pliku
            if not is_stream and not first_ready and 0 % stride == 0:
                _handle_frame(first_frame, 0)

            processed = 1
            start_time = time.time()
            est_total_processed = (total_frames // stride) if total_frames else None

            while True:
                if app.abort_event.is_set(): break
                ret, frame = cap.read()
                if not ret or frame is None: break
                processed += 1
                if (processed-1) % stride != 0: continue
                frame_idx = processed - 1
                _handle_frame(frame, frame_idx)

                if est_total_processed is not None:
                    done = min(est_total_processed, processed // stride)
                    frac = min(1.0, done/max(1, est_total_processed))
                    eta = app._eta(time.time()-start_time, max(1e-6, frac))
                    app._set_progress(frac*100.0, f"{src_name} — {done}/{est_total_processed} — stride {stride} — ETA {eta}")
                else:
                    app._set_progress(None, f"{src_name} — przetw. {processed} — stride {stride}")

            try: cap.release()
            except Exception: pass
            try: writer.release()
            except Exception: pass
            try: cv2.destroyAllWindows()
            except Exception: pass
            app._destroy_preview_window()

            df = pd.DataFrame(events)
            ev_path = save_csv_collision(df, ev_dir / f"{base_stem}_events.csv")
            summary = {
                "video": src_name,
                "lines": [{"name": ln["name"], "A_to_B": line_counts[i]["ab"], "B_to_A": line_counts[i]["ba"]} for i,ln in enumerate(lines_cfg)],
                "zones": [{"name": zn["name"], "IN": zone_counts[i]["in"], "OUT": zone_counts[i]["out"]} for i,zn in enumerate(zones_cfg)],
                "total_events": int(len(events))
            }
            sum_path = save_json_collision(summary, summ_dir / f"{base_stem}_counts.json")
            app._log(f"Zapisano: {out_video_path.name}, {ev_path.name}, {Path(sum_path).name}")

        save_json_collision({
            "params": {
                "quality": int(app.quality.get()) if not app.advanced_override else "ADV",
                **(VIDEO_PRESETS.get(int(app.quality.get()), DEFAULT_QUALITY)),
                **(app.adv_params if app.advanced_override else {}),
                "device_auto": device, "overlay_mode": app.overlay_mode.get(),
                "anchor_mode": anchor_mode,
                "trace": {"enabled": trace_on, "len": trace_len},
                "ghost_margin_px": ghost_margin,
                "alert": {
                    "enabled": alert_enabled,
                    "classes": sorted(list(alert_classes_set)),
                    "freq": alert_freq, "dur_ms": alert_dur, "freeze_ms": alert_freeze_ms
                },
            },
            "output_dir": str(outp)
        }, outp / "run_metadata.json")

        if app.abort_event.is_set():
            app._set_progress(None, "Przerwano."); app._log("=== PRZERWANO przez użytkownika ===")
        else:
            app._set_progress(100.0, "Gotowe."); app._log(f"Zakończono. Wyniki: {outp}")

    except Exception as e:
        app._log(f"[BŁĄD] {e}")
    finally:
        try:
            app.after(0, lambda: (
                app.progressbar.stop(),
                app.progressbar.config(mode="determinate"),
                setattr(app, "_progress_indeterminate", False)
            ))
        except Exception:
            pass
        app.worker_done.set()
        app.btn_start.config(state="normal")
        app.btn_abort.config(state="disabled")

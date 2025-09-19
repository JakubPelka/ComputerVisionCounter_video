# cv_video_run.py — pętla przetwarzania
from __future__ import annotations
from pathlib import Path
import time, re as _re
import cv2, numpy as np, pandas as pd

from cv_video_gui import CounterEditor                     # tylko GUI edytora
from cv_video_overlay import draw_detections, draw_counters
from cv_video_core import (                                # wspólne utilsy/stałe
    ensure_dir, device_auto_str, open_video_writer_collision,
    save_json_collision, save_csv_collision,
    VIDEO_PRESETS, DEFAULT_QUALITY,
    LINE_MIN_GAP_FRAMES_DEFAULT, ZONE_MIN_GAP_FRAMES_DEFAULT,
)

# --- geometria (krótka) ---
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

def run(app, sources, outp: Path, selected_idx):
    t0 = time.time()
    try:
        vids_dir = ensure_dir(outp / "videos")
        ev_dir   = ensure_dir(outp / "events")
        summ_dir = ensure_dir(outp / "summary")
        cnt_dir  = ensure_dir(outp / "counters")
        temp_dir = ensure_dir(outp / "temp")

        p = VIDEO_PRESETS.get(int(app.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY]).copy()
        if app.advanced_override:
            p.update(app.adv_params)
        imgsz = int(p["imgsz"]); conf = float(p["conf"]); iou = float(p["iou"])
        frame_skip = int(p["frame_skip"]); stride = max(1, frame_skip + 1)
        track_buffer = int(p["track_buffer"]); match_thresh = float(p["match_thresh"]); min_hits = int(p["min_hits"])
        line_min_gap = int(p.get("line_min_gap", LINE_MIN_GAP_FRAMES_DEFAULT))
        zone_min_gap = int(p.get("zone_min_gap", ZONE_MIN_GAP_FRAMES_DEFAULT))
        tracker_kind = app.tracker_kind.get()

        device = device_auto_str()
        id2name = app.model.names if isinstance(app.model.names, dict) else {i:nm for i,nm in enumerate(app.model.names)}
        select_names = [id2name[i] for i in selected_idx]

        app._log(f"Param: imgsz={imgsz}, conf={conf}, iou={iou}, frame_skip={frame_skip} (stride={stride}), buf={track_buffer}, match={match_thresh}, hits={min_hits}, device={device}")
        app._log(f"Tracker: {tracker_kind} | Klasy: {', '.join(select_names)}")
        app._log(f"Histereza: line_gap={line_min_gap}, line_sep=—, zone_gap={zone_min_gap}")

        for vi, source in enumerate(sources):
            src_name = (str(source) if not isinstance(source, Path) else source.name)
            if app.abort_event.is_set(): break
            app._log(f"► Źródło {vi+1}/{len(sources)}: {src_name}")

            cap = cv2.VideoCapture(source if not isinstance(source, Path) else str(source))
            if not cap.isOpened():
                app._log(f"[WARN] Nie można otworzyć: {src_name}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
            fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps>1e-3 else 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            ret, first_frame = cap.read()
            cap.release()
            if not ret or first_frame is None:
                app._log(f"[WARN] Brak pierwszej klatki: {src_name}")
                continue

            base_stem = (source.stem if isinstance(source, Path) else _re.sub(r'[^A-Za-z0-9_]+','_', src_name))
            default_cfg_path = cnt_dir / f"{base_stem}.json"
            editor = CounterEditor(app, first_frame, default_cfg_path=default_cfg_path)
            app.wait_window(editor)
            lines_cfg = editor.lines[:]
            zones_cfg = editor.zones[:]

            fps_out = max(1.0, fps / float(stride))
            writer, out_video_path = open_video_writer_collision(vids_dir / f"{base_stem}_annotated.mp4", W, H, fps_out)
            if not writer or not writer.isOpened():
                app._log(f"[ERR] Nie można otworzyć VideoWriter dla: {src_name}")
                continue

            # --- tracker YAML ---
            tracker_yaml = (temp_dir / f"{tracker_kind}.yaml")
            with open(tracker_yaml, "w", encoding="utf-8") as f:
                if tracker_kind == "botsort":
                    f.write(
f"""tracker_type: botsort
track_high_thresh: {conf}
track_low_thresh: {max(conf-0.1, 0.05)}
new_track_thresh: {conf}
track_buffer: {track_buffer}
match_thresh: {match_thresh}
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.25
min_hits: {min_hits}
fuse_score: True
with_reid: False
"""
                    )
                else:
                    f.write(
f"""tracker_type: bytetrack
track_high_thresh: {conf}
track_low_thresh: {max(conf-0.1, 0.05)}
new_track_thresh: {conf}
track_buffer: {track_buffer}
match_thresh: {match_thresh}
min_box_area: 10
mot20: False
fuse_score: False
"""
                    )

            generator = app.model.track(
                source=(source if not isinstance(source, Path) else str(source)),
                stream=True, verbose=False,
                imgsz=imgsz, conf=conf, iou=iou, device=device,
                classes=selected_idx, tracker=str(tracker_yaml),
                persist=True, save=False, vid_stride=stride
            )

            last_centroid = {}
            line_states = [{ } for _ in lines_cfg]
            line_counts = [{"ab":0,"ba":0} for _ in lines_cfg]
            zone_states = [{ } for _ in zones_cfg]
            zone_counts = [{"in":0,"out":0} for _ in zones_cfg]
            events = []
            from collections import deque
            trails = {}

            processed = 0
            start_time = time.time()
            est_total_processed = (total_frames // stride) if total_frames else None

            for res in generator:
                if app.abort_event.is_set(): break
                processed += 1
                frame_idx = processed * stride - 1

                frame = res.orig_img.copy() if hasattr(res, "orig_img") and res.orig_img is not None else None
                if frame is None:
                    try: frame = res.plot()
                    except Exception: continue

                det_boxes, det_confs, det_cids, det_ids = [], [], [], []
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.array([-1]*len(xyxy))
                    for b,s,c,tid in zip(xyxy, confs, cls, ids):
                        if tid < 0:   # tylko trackowane obiekty
                            continue
                        det_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                        det_confs.append(float(s))
                        det_cids.append(int(c))
                        det_ids.append(int(tid))

                # centroidy + stany
                centroids = []
                for b in det_boxes:
                    cx = 0.5*(b[0]+b[2]); cy = 0.5*(b[1]+b[3]); centroids.append((cx,cy))

                for (tid, b, s, cid, (cx,cy)) in zip(det_ids, det_boxes, det_confs, det_cids, centroids):
                    # Linie
                    for li, ln in enumerate(lines_cfg):
                        a = (ln["a"][0], ln["a"][1]); b2 = (ln["b"][0], ln["b"][1])
                        st = line_states[li].get(tid, {"last_side": None, "last_frame": -9999})
                        prev_side = st["last_side"]
                        cur_side = line_side(a, b2, (cx,cy))
                        crossed = False; direction = None
                        if prev_side is not None:
                            prev_c = last_centroid.get(tid, (cx,cy))
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
                                "video": src_name,
                                "frame": int(frame_idx),
                                "time_sec": float(frame_idx / max(1.0, fps)),
                                "track_id": int(tid),
                                "class_id": int(cid),
                                "class_name": (app.names[cid] if isinstance(app.names, dict) else app.names[cid]),
                                "event_type": f"line_{direction}",
                                "counter_name": ln["name"],
                                "conf": float(s)
                            })
                        else:
                            line_states[li][tid] = st

                    # Strefy
                    for zi, zn in enumerate(zones_cfg):
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
                                    "video": src_name,
                                    "frame": int(frame_idx),
                                    "time_sec": float(frame_idx / max(1.0, fps)),
                                    "track_id": int(tid),
                                    "class_id": int(cid),
                                    "class_name": (app.names[cid] if isinstance(app.names, dict) else app.names[cid]),
                                    "event_type": ev,
                                    "counter_name": zn["name"],
                                    "conf": float(s)
                                })
                        else:
                            zone_states[zi][tid] = sstate

                    # trajektorie
                    last_centroid[tid] = (cx,cy)
                    dq = trails.get(tid)
                    if dq is None:
                        from collections import deque
                        dq = deque(maxlen=24)
                        trails[tid] = dq
                    dq.append((int(cx), int(cy)))

                # overlay
                draw_detections(frame, det_boxes, det_confs, det_cids, det_ids, app.names, app.overlay_mode.get())
                draw_counters(frame, lines_cfg, line_counts, zones_cfg, zone_counts, trails)

                writer.write(frame)

                # progres
                if est_total_processed:
                    frac = (processed)/max(1, est_total_processed)
                    eta = app._eta(time.time()-start_time, min(1.0, frac))
                    app._set_progress(frac*100.0, f"{src_name} — przetw. {processed}/{est_total_processed} — stride {stride} — ETA {eta}")
                else:
                    app._set_progress(None, f"{src_name} — przetw. {processed} — stride {stride}")

            writer.release()

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

        # meta
        save_json_collision({
            "params": {
                "quality": int(app.quality.get()) if not app.advanced_override else "ADV",
                **(VIDEO_PRESETS.get(int(app.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])),
                **(app.adv_params if app.advanced_override else {}),
                "device_auto": device, "tracker": app.tracker_kind.get(),
                "overlay_mode": app.overlay_mode.get()
            },
            "selected_classes": [app.names[i] if isinstance(app.names, dict) else app.names[i] for i in selected_idx],
            "output_dir": str(outp)
        }, outp / "run_metadata.json")

        if app.abort_event.is_set():
            app._set_progress(None, "Przerwano.")
            app._log("=== PRZERWANO przez użytkownika ===")
        else:
            app._set_progress(100.0, "Gotowe.")
            app._log(f"Zakończono. Wyniki: {outp}")

    except Exception as e:
        app._log(f"[BŁĄD] {e}")
    finally:
        app.worker_done.set()
        app.btn_start.config(state="normal")
        app.btn_abort.config(state="disabled")

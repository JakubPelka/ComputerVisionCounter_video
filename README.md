# YOLO Video Counter

> Windows-first desktop app (Tkinter + Ultralytics YOLOv11 + ByteTrack + optional Supervision) for counting line crossings and zone entries on videos or live camera/RTSP streams. Stores annotated MP4, CSV of events (with timestamps), JSON summary, and optional snapshots.

---

## 1) Quick start (Windows)

1. **Clone / unzip** this repository anywhere (avoid paths with non-ASCII if you hit issues).
2. Put model weights in `./models/` (e.g. `yolo11x.pt` or a `.zip` with a `.pt` inside).
3. (Preferred) Run via the bootstrap starter:

   ```bat
   start.py
   ```

   or:

   ```bat
   python start.py
   ```

   This prioritizes local packages in `./_pkgs/` and sets camera backends for stability.
4. The app starts as `cv_video.py` with the GUI.

> **Note**: You can also run `cv_video.py` directly; the bootstrap just helps with local wheels and Windows camera backend (DirectShow) to avoid MSMF warnings.

---

## 2) Folder layout

```
Yolo_video_counter/
├─ cv_video.py            # main launcher (GUI + wiring)
├─ cv_video_gui.py        # GUI widgets (AppUIMixin, CounterEditor, scrollable class grid, etc.)
├─ cv_video_run.py        # processing loop (tracking, counting, overlay, snapshots)
├─ cv_video_core.py       # constants, presets, I/O utils, geometry (lines/zones)
├─ models/                # put your YOLO .pt or zipped .pt here
├─ _pkgs/                 # optional local packages/wheels cache
├─ results/               # created on first run
│  ├─ videos/             # annotated MP4
│  ├─ events/             # CSV events; snapshots/ subfolder
│  ├─ summary/            # JSON summaries per source
│  └─ temp/               # tracker config, temp files
├─ start.py               # bootstrap launcher (Windows-friendly)
├─ bootstrap_env.py       # older bootstrap (still works)
└─ env_guard.py           # environment helper (optional)
```

---

## 3) GUI overview

* **Source**

  * *Files*: select one or more video files.
  * *Folder*: process all supported files in a directory.
  * *Camera / URL*: enter numeric camera index (0, 1, …) or RTSP/HTTP URL. Live preview can be shown during processing.

* **Model & classes**

  * Choose weights (`models/*.pt` or `.zip`). The app loads class names and builds a **responsive checkbox grid** that auto-fits to window width.
  * Tick one or more classes to track & count (e.g., `person`, `car`).

* **Quality preset**

  * Slider 1..5 mapping to preset dict (img size, conf, iou, frame stride, tracker buffer, etc.).
  * Use **Advanced** to override any parameter.

* **Overlay**

  * `centroid` (anchor point) or `boxes / boxes_conf` (+ optional Supervision `BoxAnnotator` labels).
  * **Trace**: trailing path length per track-id (0 disables). `Anchor`: bottom or center. `Ghost margin`: vertical offset for bottom-anchor.

* **Counters editor** (opens when you start processing)

  * Draw multiple **lines** (A→B) and **polygons** (zones). Name them and save/load from JSON in `results/counters/`.
  * For stream sources, the preview can be temporarily paused while drawing (configurable in GUI).

* **Sound alert**

  * Optional beep on **zone presence** for selected classes (frequency, duration, freeze-time between beeps are configurable). See §7.

* **Buttons**

  * **Start** launches the worker thread; **Abort** stops it and releases a camera.
  * Progress bar shows either percent (when total is known) or indeterminate animation.

---

## 4) Outputs

For each source (file / camera):

* `results/videos/<name>_annotated.mp4` — annotated result.
* `results/events/<name>_events.csv` — all events with **timestamps**:

  * `ts_local` (local timezone), `ts_utc` (UTC), `frame`, `time_sec`, `track_id`, `class_name`, `event_type` (`line_ab`, `line_ba`, `zone_in`, `zone_out`), `counter_name`, `conf`.
* `results/events/snapshots/*.jpg` — optional snapshots taken at event time.
* `results/summary/<name>_counts.json` — totals per line/zone.
* `results/run_metadata.json` — parameters used for the run.

---

## 5) Advanced options

* `imgsz`, `conf`, `iou`, `frame_skip` (stride = frame\_skip+1)
* Tracker tuning: `track_buffer`, `match_thresh`, `min_hits` (ByteTrack/BOTSort YAML is generated per run under `results/temp/`).
* Hysteresis for counters: `line_min_gap_frames`, `line_min_sep_px`, `zone_min_gap_frames`.
* Overlay/trace/anchor/ghost.
* **Alerts**:

  * enable/disable, class allow-list, frequency (Hz), duration (ms), **freeze** (ms) between beeps while an object **stays in a zone**.
* **Presets**: save/load your own advanced profiles (JSON).

---

## 6) Performance (CPU tips)

* Lower `imgsz` (e.g., 320–512), increase `frame_skip` (stride 2–4), lower confidence.
* Use ByteTrack with small `track_buffer` when objects move slowly and exits/re-entries should be re-counted quickly.
* Disable confidence text overlay; use centroid mode instead of boxes.
* Close the live preview if your laptop struggles.

---

## 7) Alerts: zone presence vs. line crossing

By default we support **zone presence** alerts: while an allowed class is **inside any zone**, the app will beep no more frequently than every `alert_freeze` milliseconds.

See the code note in §9 to ensure you are on presence-based alerting.

---

## 8) Troubleshooting (Windows)

* **MSMF `can't grab frame` warnings**: we force **DirectShow** backend. If you still see MSMF logs, start the app via `start.py` which sets environment variables *before* importing OpenCV/Ultralytics.
* **Torch / torchvision mismatch**: `start.py` prints a hint. Use the recommended pair (or update both).
* **Supervision API differences**: `BoxAnnotator.annotate(..., labels=...)` changed across versions. The code has a try/except fallback. Update Supervision if you want label rendering inside its annotator.

---

## 9) Code map

* `cv_video.py`
  GUI app entry-point; wires buttons/vars to the worker thread in `cv_video_run.run()`; maintains progress & preview.

* `cv_video_gui.py`
  `AppUIMixin`, `ScrollableFrame`, `CounterEditor` (line/zone editor), responsive class grid, advanced options dialog (save/load preset), alert settings.

* `cv_video_run.py`
  The processing loop: YOLO tracking stream, event logic (line/zones), overlay drawing, **zone presence alerting**, timestamps/snapshots, CSV/JSON/MP4 outputs.

* `cv_video_core.py`
  Presets & constants, file/dir utilities (writer, numbered paths, save CSV/JSON), and geometry helpers: `line_side`, `segments_intersect`, `dist_point_to_segment`, `point_in_polygon`.

---

## 10) License

MIT for the app code. Models follow their own licenses.

---

## 11) Notes for Polish users (PL)

* Aplikacja domyślnie uruchamia się na Windows i preferuje backend kamer **DirectShow** — to ogranicza problemy MSMF.
* **Alert dźwiękowy** dotyczy *przebywania w strefie* (nie tylko przecięcia). Częstotliwość wyzwalania reguluje `alert_freeze` (ms) w Opcjach zaawansowanych.
* Presety można zapisać/wczytać w oknie zaawansowanych ustawień.

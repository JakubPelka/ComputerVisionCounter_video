# INTERNAL README — ComputerVisionCounter (Video Edition) Readme_INT_2025_10_15

**Date:** 2025‑10‑15
**Audience:** Internal developers & maintainers
**Scope:** Desktop, Windows‑first, Tkinter UI. Single‑machine processing of video files and live streams.

---

## 0. Ground Rules

* **Small, incremental changes.** Keep file sizes ~**500–600 lines** each when possible.
* **Do not rewrite UI/UX**; extend without breaking existing user flows.
* **Prefix module names** with `cv_...`.
* **Every structural change** → update `architecture.md` and this document if relevant.
* **No background tasks** beyond the main processing thread; keep the app responsive.

---

## 1. Folder Structure (Commercial Build)

```
<root>/
├─ src/                          # all source code lives here
│  ├─ cv_video.py                # main GUI entry and orchestrator
│  ├─ cv_video_gui.py            # widgets, CounterEditor (lines/zones), preview helpers
│  ├─ cv_video_run.py            # frame-by-frame runner (detections, tracking, HUD, I/O)
│  ├─ cv_video_advanced_ui.py    # Advanced panel (Main & Extras tabs)
│  ├─ cv_video_hud.py            # RIGHT HUD (lines/zones + per-class breakdown + SUM)
│  ├─ cv_video_hud_extras.py     # LEFT HUD (Now/Max) with unified scaling
│  ├─ cv_video_heatmap.py        # heatmap engine and overlay utilities
│  ├─ cv_video_overlay.py        # boxes/centroids drawing, line/zone frames
│  ├─ cv_video_geom.py           # geometry utils (line side, PIP, intersections)
│  ├─ cv_video_sound.py          # alert sound player (loop/ping)
│  ├─ cv_video_core.py           # file I/O, writers, JSON/CSV helpers, presets
│  ├─ cv_video_stats.py          # runtime stats aggregation (Now/Max)
│  └─ (optional) cv_video_preview.py, paths.py  # if present; legacy helpers
├─ models/                       # .pt weights
├─ indata/                       # videos or stream .txt lists (optional)
├─ output/
│  ├─ videos/                    # annotated MP4
│  ├─ events/                    # CSV events (+ snapshots subfolder if enabled)
│  ├─ heatmap/                   # heatmap images (only when enabled)
│  └─ summary/                   # per-run JSON/CSV summaries
├─ presets/                      # saved advanced presets (.json)
├─ sounds/                       # alert audio files
├─ start.bat                     # launcher (calls Python on our entrypoint)
└─ README.md / architecture.md   # docs
```

---

## 2. Runtime Overview (High Level)

1. **User selects source & model** in `cv_video.py` → clicks **Start**.
2. We grab a first frame (or live capture) → open **CounterEditor** to draw **lines/zones** (AOI optional).
3. We enter the **runner** (`cv_video_run.run`):

   * Run YOLO, convert detections, feed tracker.
   * Compute **line crossings** (AB/BA) & **zone in/out** events.
   * Update **HUDs** (left: Now/Max; right: lines/zones + per‑class per‑zone + SUM).
   * Optionally **accumulate heatmap** and **play alerts**.
   * Save annotated frames to video writer; periodically dump CSV/JSON.

### Key Objects (runner)

* `line_states[]`, `zone_states[]` — per‑ID hysteresis structures.
* `line_counts[]`, `zone_counts[]` — totals rendered on HUD.
* `events[]` — append‑only event rows saved to CSV (also exposed for HUD fallback via `app._ev_ref`).
* `app._zone_class_totals_by_zone` — list of `{in:{}, out:{}}` per zone (per‑class counts).
* `app._zone_class_totals_sum` — global `{in:{}, out:{}}` across zones (per‑class sums).
* `DetectionHeatmap` — float32 grid with decay/window; renders overlay or saved PNG/RGBA.

---

## 3. Detection / Tracking

* **Detector:** Ultralytics YOLO (v8/v11 compatible) through `app.model(...)`.
* **Parser:** `_parse_ultra_results(res)` → `(boxes[N,4], scores[N], cids[N])`.
* **Tracker:** Supervision ByteTrack when available; graceful fallback to identity IDs.
* **Quality Preset:** Main slider synced into Advanced fields (`imgsz`, `conf`, `iou`, `frame_skip`, `track_buffer`, `match_thresh`, `min_hits`). See `cv_video_advanced_ui._extract_quality_from_main`.

**Performance knobs** (Advanced → Main):

* `imgsz` (tile size): 320–960 typical.
* `frame_skip`: 1 (all frames), or n>1 to sub‑sample.
* `track_buffer`, `match_thresh`, `min_hits` — tracking stability vs responsiveness.

---

## 4. Counters: Lines & Zones

* **Lines:**

  * Either 2‑point line or polyline.
  * Crossing detection via side‑sign changes and segment intersection.
  * Direction → **`ab` / `ba`**; debounce by `line_min_gap` frames.
  * Events appended with `event_type = "line_ab|line_ba"` and **`AB/BA` flags**.
* **Zones:**

  * Polygon PIP; hysteresis by `zone_min_gap` frames.
  * Events appended with `event_type = zone_in|zone_out` and `AB=0/BA=0`.
* **Anchor point:** `center` or `bottom` (with `ghost_margin`) via `_anchor_from_box`.

**UI:** Counters are drawn/edited in `CounterEditor` (in `cv_video_gui`), saved as JSON in `output/counters/` (per source).

---

## 5. Alerts (Sound)

* Controlled by **Advanced → Sound**.
* `alert_mode`:

  * `1` = **inside zone** (active when inside),
  * `0` = **outside zone** (active when outside),
  * `2` = **line crossing** (fires only on crossing event).
* **Looping:** applies to zone modes only. Line mode is always **single ping**.
* **Cooldown:** `alert_freeze_s` prevents frequent pings/restarts.
* **Players:** `ffplay/afplay/aplay/winsound/PowerShell/simpleaudio` (auto‑select). All abstracted by `SoundPlayer`.

---

## 6. Heatmaps

* **Create heatmap = Off** → **no files written**.
* **On** → saves **only** under `output/heatmap/` (periodic + final).
* **Overlay toggle:** press **`m`** during run.
* **Parameters:** `alpha` (opacity), `gamma` (contrast), `sigma` (point radius), `gain` (per‑hit), `decay` (per‑frame), `window_minutes` (rolling window), `save_interval_s`.
* **Zero‑as‑transparent:** `heat_zero_thresh` makes low values fully transparent in RGBA.

Implementation: `cv_video_heatmap.DetectionHeatmap`

* Core buffer: `float32` H×W.
* Decay: multiplicative; window mode converts target minutes to **per‑frame decay** (`~1/e` at window length).
* Saving helpers: `save_png(...)`, `save_png_rgba(...)`, `render_overlay_masked(...)`.

---

## 7. HUDs (Unified Scaling)

* **Left HUD** (`cv_video_hud_extras.draw_run_counters`) — Now & Max per class.
* **Right HUD** (`cv_video_hud.draw_counts_panel`) — Lines (AB/BA), Zones (IN/OUT per zone), **per‑class breakdown per zone**, and **global SUM**.
* Both use **the same scaling formula**: `s = clamp(H/720, 0.6..2.2) * hud_scale`.
* Right HUD reconstructs per‑class maps from `app._ev_ref` if live accumulators missing.

---

## 8. Outputs

* **Video:** `output/videos/<basename>_annotated.mp4` (writer via `cv_video_core.open_video_writer_collision`).
* **CSV (events):** `output/events/<basename>_<run>_events.csv` — columns:

  * `frame, time_sec, timecode, clock, track_id, class_id, class_name, event_type, counter_name, AB, BA, conf`
* **Summary JSON/CSV:** per source totals (lines AB/BA, zones IN/OUT, advanced params snapshot).
* **Heatmaps:** `output/heatmap/` (opaque, RGBA and overlay snapshots), only when enabled.
* **Snapshots on events:** optional (Extras → Snapshot on events). Saved per event with structured filenames.

---

## 9. Advanced Panel — Key Bindings & Sync

* Quality slider ↔ Advanced fields synced via robust parser (`imgsz=.. conf=.. iou=.. skip=..`).
* Overlay selection **removed** from Advanced; controlled in Main UI.
* Help panel auto‑wraps; same width for **Main** and **Extras**.
* Sound ▶/⏹ test buttons — use `SoundPlayer` directly.

---

## 10. Coding Conventions

* **File naming:** `cv_<domain>.py`.
* **Keep modules focused**; split when over ~600 lines.
* **No hardcoded absolute paths**; use `cv_video_core.ensure_dir` and root‑relative conventions.
* **Log via** `app._log(...)`; never print spam to console in production.
* **Avoid tight coupling** across UI/runner layers; communicate via `app` fields and plain dicts.

---

## 11. Extension Points

* **New detectors or trackers:** wrap outputs to `(boxes, scores, cids)` and return `det_ids` in `_track_update`.
* **New HUD fields:** extend rows assembly in `cv_video_hud.py` and/or `cv_video_hud_extras.py`; use the unified scale.
* **New outputs:** add save blocks in runner after frame loop to keep I/O serialized and consistent.
* **New alerts:** route through `_update_counts_and_alerts(...)`; keep cooldown & loop semantics.

---

## 12. QA Checklist (Pre‑release)

1. **Quality slider sync** → open Advanced; values match; Apply/Save/Load work.
2. **Drawing editor** → lines & zones saved/loaded; names appear in HUD & CSV.
3. **Line crossings** → AB/BA counted; ping sound on crossing; cooldown respected.
4. **Zones** → IN/OUT stable with hysteresis; loop alert starts/stops correctly.
5. **Right HUD** → shows per‑zone per‑class breakdown and global SUM.
6. **Left HUD** → Now & Max per class scale identically to right HUD.
7. **Heatmap OFF** → no files in `output/heatmap/`.
8. **Heatmap ON** → only writes to `output/heatmap/`; overlay toggles with `m`.
9. **CSV events** → schema present; AB/BA flags; counter names correct.
10. **Summary JSON/CSV** → contains alert_mode, heatmap params, counts.
11. **Snapshots** (if enabled) → correct timestamps, names, safe filenames.
12. **Performance** → test with `imgsz` small/large; `frame_skip` 1/2/3.
13. **Sounds** → ▶ test works; line vs zone behave per design on Windows.

---

## 13. Troubleshooting

* **No camera / stream:** verify source string; MSMF vs DirectShow; try file first.
* **No audio:** ensure a supported player is available; try `.wav`; test ▶ in Advanced.
* **Heatmap too faint / too fast decay:** increase `gain`; reduce `decay`; or enable rolling `window_minutes`.
* **Right HUD lacks classes:** ensure we set `app._ev_ref = events`; fallback rebuild will populate.
* **Large outputs:** increase `frame_skip` or disable snapshots.

---


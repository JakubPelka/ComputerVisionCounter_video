# ComputerVisionCounter VIDEO – Architecture Overview (2025-10-14)

## 1. Entry & Boot

### **start.py**

* Application launcher; initializes environment, checks folders and dependencies.
* Runs `src/cv_video.py` as main entrypoint.

### **paths.py**

* Defines common directories: `INPUT`, `OUTPUT`, `MODELS`, `SOUNDS`, `PRESETS`.
* Used by all modules for consistent I/O.

---

## 2. Main GUI Layer

### **cv_video.py**

* Central orchestrator for the whole app.
* Builds main Tkinter window and menu layout.
* Handles:

  * Model and source (video/camera/stream) selection.
  * Quality slider (affects detection & tracking presets).
  * Overlay (HUD visibility) mode.
  * Start / Abort workflow.
* Launches processing thread using `cv_video_run.run()`.
* Opens **Advanced panel** from `cv_video_advanced_ui.py`.
* Exposes app-level attributes (`adv_params`, `v_quality`, `trace_enabled`, `alert_enabled`, etc.) used across modules.

### **cv_video_gui.py**

* Contains GUI helper classes:

  * `ScrollableFrame` – vertical scrolling frame.
  * `CounterEditor` – handles line/zone creation and AOI drawing.
  * `AppUIMixin` – shared mixin for preview management and event handling.
* Synchronizes Quality slider with advanced parameters.
* Manages drawing mode windows (edit lines/zones, cancel, save & close).

### **cv_video_preview.py**

* Handles live preview window display.
* ESC or close-window → aborts current analysis gracefully.
* Allows frame overlay toggling for HUD preview.

---

## 3. Advanced Settings

### **cv_video_advanced_ui.py**

* Two-tab interface: **Main** and **Extras**.
* Syncs current Quality slider preset to advanced variables (`imgsz`, `conf`, `iou`, `skip`, etc.).
* **Main tab:**

  * Detection / Tracking / Hysteresis settings.
  * Trace & anchor setup.
  * Frame & trace colors and thickness.
  * Full sound alert configuration.
  * Live preview and ghost margin control.
* **Extras tab:**

  * HUD size (affects both left and right panels).
  * Snapshot-on-event toggle.
  * **Heatmap section:** create, overlay toggle, AOI restriction, alpha, gamma, sigma, decay, window, gain, memory multiplier, save interval.
* Adds full Help system with wrapped text (consistent width between tabs).
* Preset system: Apply / Save preset / Load preset / Close.
* Sound test (▶ / ⏹) for alert sound.

---

## 4. Processing Core

### **cv_video_run.py**

* Main loop controlling frame-by-frame video processing.
* Responsibilities:

  * Load YOLO model and track objects.
  * Manage detections, IDs, classes, and confidence scores.
  * Check crossings for **lines** and **zones**.
  * Update internal states for event counters.
  * Call HUD renderers and overlay functions.
  * Save annotated frames or snapshots.
  * Manage sound alerts and heatmap accumulation.
* **Sound Alerts:**

  * Supports both *Zone alerts* (loop or ping) and *Line crossing* (single ping).
  * All modes controlled by `alert_mode` (inside / outside / line crossing).
* **Heatmap Logic:**

  * Controlled from Advanced panel.
  * When disabled → no files are saved.
  * When enabled → all saves go to `/output/heatmap/` only.
  * Overlay toggle (‘m’) enables/disables live preview of heatmap.
* **Event Logging:**

  * Builds `events[]` list (frame, class, event_type, counter_name, etc.).
  * Saves to `CSV_events` with AB/BA columns for line crossings.
* **Per-class Zone Statistics:**

  * Maintains `app._zone_class_totals_by_zone` and `_zone_class_totals_sum`.
  * Records IN/OUT counts per zone & per class.
  * Exposes `app._ev_ref` to HUD for fallback reconstruction.
* **HUD Scaling:** unified logic (`auto_scale * hud_scale`).

---

## 5. Visualization & Overlays

### **cv_video_hud.py**

* Responsible for drawing:

  * Per-line counters (AB/BA).
  * Per-zone counters (IN/OUT).
  * **Right panel:** detailed zone breakdown – per-class IN/OUT and global SUM.
  * **Left panel:** Now & Max counts.
* Uses unified scaling formula (auto × hud_scale).
* Rebuilds per-class summaries on the fly from `app._ev_ref` if live data unavailable.

### **cv_video_hud_extras.py**

* Contains helper functions for HUD rendering:

  * `draw_run_counters()` for left-side stats.
  * Unified font, padding, and scaling to match right panel.

### **cv_video_overlay.py**

* Draws detection boxes, centroids, and lines/zones.
* Integrates with supervision or OpenCV overlays.
* Handles alpha blending and line/zone color customization.

---

## 6. Support Modules

### **cv_video_geom.py**

* All geometric calculations:

  * Line intersection checks.
  * Side of line determination.
  * Polygon hit test (point-in-polygon).

### **cv_video_heatmap.py**

* Manages heatmap creation and visualization.
* `DetectionHeatmap` accumulates detection centroids with Gaussian kernel.
* Supports both time decay and rolling window accumulation.
* Transparent overlay via alpha channel.
* Two export types:

  * `*_heatmap.png` – opaque normalized.
  * `*_heatmap_rgba.png` – transparent zeros.
* Works with: `heat_enabled`, `heat_decay`, `heat_window_minutes`, `heat_gain`, etc.

### **cv_video_sound.py**

* Platform-independent audio alert player.
* Supports: ffplay / afplay / paplay / aplay / PowerShell / winsound / simpleaudio.
* Functions:

  * `play_once()` – single ping.
  * `start_loop()` / `stop()` – continuous loop for active state.
* Used by both line and zone events.

### **cv_video_core.py**

* Shared backend utilities:

  * File saving and path management.
  * Annotated frame output.
  * CSV/JSON export helpers.

### **cv_video_stats.py**

* Statistical and CSV aggregation functions.
* Formats results for `CSV_events` and summary JSON outputs.

---

## 7. Data Flow Overview

```
start.py
 └─ cv_video.py                (main app)
      ├─ cv_video_gui.py       (UI / preview controls)
      ├─ cv_video_advanced_ui.py (advanced settings)
      ├─ cv_video_run.py       (frame loop)
      │    ├─ cv_video_hud.py          (HUD rendering)
      │    ├─ cv_video_hud_extras.py   (HUD helpers)
      │    ├─ cv_video_overlay.py      (visual overlays)
      │    ├─ cv_video_geom.py         (geometry math)
      │    ├─ cv_video_sound.py        (sound alerts)
      │    ├─ cv_video_core.py         (utilities)
      │    └─ cv_video_heatmap.py      (heatmaps)
      └─ cv_video_preview.py    (live view / control)
```

---

## 8. Key Functional Highlights

* ✅ **Heatmap Control:** full enable/disable logic; saves exclusively to `/output/heatmap/` when active.
* ✅ **Sound Alerts:** both zone and line crossing triggers; loop and single ping modes.
* ✅ **HUD Scaling:** consistent formula across left & right panels.
* ✅ **Right HUD:** per-zone + per-class IN/OUT counts + global summary.
* ✅ **Quality Sync:** automatic propagation between main slider and Advanced parameters.
* ✅ **Help System:** wrapped, resizable, identical width between tabs.
* ✅ **CSV Events:** unified schema (frame, timecode, event_type, AB/BA, class_id/name, counter_name, conf).
* ✅ **Fallback Rebuild:** per-class zone stats reconstructed from events if accumulators absent.
* ✅ **Snapshot-on-Event:** optional auto-save of trigger frames.
* ✅ **Unified Font & Scale:** both HUDs visually balanced, auto-adjust with frame height.

---

## 9. Future-Proofing / Next Steps

* 🧩 Add optional metrics export (JSON summary for per-class, per-zone, per-line performance).
* 🧩 Add persistent user presets (auto-load last used settings).
* 🧩 Optional heatmap normalization toggle (sticky vs dynamic scale).
* 🧩 UI: multi-source configuration & preview layout improvements.

---

## 10. Change Log (since previous iteration)

* Added: line crossing alert mode (single ping).
* Added: global and per-zone per-class IN/OUT tracking.
* Added: HUD right panel class breakdowns + SUM.
* Improved: heatmap saving logic (off = no output, on = `/output/heatmap/` only).
* Improved: unified HUD scale for both panels.
* Improved: Advanced UI (help wrapping, numeric inputs, sound test, preset management).
* Improved: synchronization of Quality slider → Advanced panel.
* Fixed: overlay duplication & out-of-sync scaling issues.
* Fixed: empty heatmap writes when disabled.

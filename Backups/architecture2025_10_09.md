start.py
  └─ boots environment, then launches:  src/cv_video.py  (GUI app)

paths.py
  └─ Defines project paths (INPUTS, OUTPUTS, MODELS, SOUNDS, etc.) used by UI and runner.

cv_video.py   (main GUI orchestrator)
  ├─ builds main window, loads model, picks sources/classes, shows logs/progress
  ├─ opens Advanced panel via:  build_advanced_settings(...)  (from cv_video_advanced_ui.py)
  ├─ manages preview via: preview_* helpers (from cv_video_preview.py)
  └─ starts processing thread calling: run(app, sources, out_dir, selected_idx) (cv_video_run.py)

cv_video_advanced_ui.py
  └─ builds the Advanced panel (vertical layout), updates app.adv_params, and wires help.

cv_video_preview.py
  └─ creates and updates the live preview window (Esc/close == Abort), resizes video to window.

cv_video_run.py   (processing/runner)
  ├─ reads frames; runs YOLO (ultralytics) + tracker (ByteTrack/BoT-SORT)
  ├─ computes line/zone events (using helpers below)
  ├─ draws overlays & HUD; writes annotated video + CSV/JSON summaries
  ├─ pushes frames to GUI preview via app._show_preview_bgr(...)
  ├─ manages sound alerts via SoundPlayer (cv_video_sound.py)
  └─ imports helpers:
       • cv_video_geom.py   – geometry (line/polyline, point-in-polygon, crossing)
       • cv_video_hud.py    – draw lines/zones/trails + results panel (BR black box)
       • cv_video_overlay.py– draw detections (boxes/centroids/polygons)
       • cv_video_sound.py  – SoundPlayer (ffplay/afplay/paplay/aplay/winsound/PowerShell/simpleaudio)
       • cv_video_core.py   – IO helpers (writers, saving CSV/JSON, device pick, presets)

cv_video_geom.py
  └─ line/polyline helpers, side tests, intersections, point-in-polygon, etc.

cv_video_hud.py
  └─ paint lines/zones/trails, and the bottom-right black “results panel”.

cv_video_overlay.py
  └─ detection overlays using supervision (boxes/labels/centroids/polygons). Reuses HUD helpers.

cv_video_sound.py
  └─ cross-platform sound playback (loop/ping), used by runner and the “▶ Test” button.

cv_video_gui.py
  └─ re-usable GUI pieces (ScrollableFrame, CounterEditor for drawing lines/zones/polylines).

cv_video_core.py
  └─ common utilities (ensure_dir, open_video_writer_collision, save_json/csv, presets).

covert.py / env_guard.py
  └─ misc helpers & environment checks (kept untouched).

README.md
  └─ docs (to update later with /src layout).

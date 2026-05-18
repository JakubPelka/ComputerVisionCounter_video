# Architecture overview

This document summarizes the current application structure. It intentionally avoids changing code.

## Entry point

```text
start.bat
└─ src/start.py
   └─ src/cv_video.py
```

`start.bat` is the Windows launcher.  
`src/start.py` prepares the local dependency environment and starts the main application.

## Main application layers

```text
src/cv_video.py
```

Main Tkinter GUI and application orchestrator.

Responsibilities:

- main window layout,
- model/source selection,
- class selection,
- quality slider,
- overlay options,
- Start/Abort workflow,
- launching the processing runner,
- opening the Advanced panel.

```text
src/cv_video_gui.py
```

GUI helpers and counter editor.

Responsibilities:

- line drawing,
- zone drawing,
- AOI/counter editing,
- GUI helper widgets,
- preview-related UI helpers.

```text
src/cv_video_advanced_ui.py
```

Advanced settings window.

Responsibilities:

- detection and tracking parameters,
- hysteresis settings,
- sound options,
- HUD size,
- snapshot settings,
- heatmap settings,
- preset loading/saving,
- help text.

```text
src/cv_video_run.py
```

Main frame-by-frame processing loop.

Responsibilities:

- loading/using YOLO model,
- detection parsing,
- tracking,
- line crossing logic,
- zone in/out logic,
- event list creation,
- HUD rendering calls,
- overlay rendering calls,
- writing outputs,
- sound alert calls,
- heatmap accumulation.

This file is a candidate for later careful refactoring, but not during the repository hygiene step.

## Support modules

```text
src/cv_video_core.py
```

Shared I/O and output helpers.

```text
src/cv_video_geom.py
```

Geometry utilities for lines, zones and intersections.

```text
src/cv_video_heatmap.py
```

Heatmap accumulation, rendering and saving.

```text
src/cv_video_hud.py
src/cv_video_hud_extras.py
```

HUD rendering for counters, current counts and summaries.

```text
src/cv_video_overlay.py
```

Detection overlays, boxes, centroids, traces, line/zone drawing.

```text
src/cv_video_preview.py
```

Preview window handling. ESC or closing the preview should abort gracefully.

```text
src/cv_video_sound.py
```

Sound alert handling.

```text
src/cv_video_stats.py
```

Runtime statistics and summary-related helpers.

## Data flow

```text
User selects source + model
        ↓
Counter editor opens for lines/zones
        ↓
Processing runner reads frames
        ↓
YOLO detects objects
        ↓
Tracker assigns IDs
        ↓
Line/zone logic creates events
        ↓
HUD/overlay/alerts/heatmap update
        ↓
Results are saved locally under output/
```

## Local folders

```text
models/
```

Local model weights. Ignored by Git.

```text
indata/
```

Local input videos. Ignored by Git.

```text
output/
```

Generated results. Ignored by Git.

```text
presets/
```

Local Advanced settings presets. Ignored by Git except README/placeholders.

```text
sounds/
```

Local alert sounds. Ignored by Git except README/placeholders.

```text
wheels/
```

Optional local wheels for offline setup. Ignored by Git except README/placeholders.

## Refactoring rule

Do not refactor large files until the repository hygiene step is complete.

When code refactoring starts:

1. preserve current behavior,
2. change one module boundary at a time,
3. test after every small step,
4. update this document after structural changes.

# ComputerVisionCounter – Video Edition (Public README)

> AI‑powered counting and event analytics for videos and live streams. Draw lines and zones, detect objects (e.g., people, cars), hear alerts, export CSVs, and create heatmaps — all without coding.

---

## Who is this for?

* Teams that need **quick, local** analytics of entrances, corridors, intersections, or shop floors.
* Users who want results (counts, CSVs, annotated videos) **without cloud services** or programming.

**Runs offline.** Your videos stay on your machine.

---

## System Requirements

* **OS:** Windows 10/11 (64‑bit)
* **CPU:** Recent Intel/AMD (AVX2 recommended). A discrete **NVIDIA GPU** speeds up analysis but is **not required**.
* **RAM:** 8 GB minimum (16 GB recommended for long videos)
* **Disk:** ~2 GB free + space for outputs

---

## What’s in the package

```
ComputerVisionCounter_video/
├─ src/                  # program files (no need to edit)
├─ models/               # place your YOLO .pt models here
├─ indata/               # put your input videos here
├─ output/               # results will be written here
├─ presets/              # optional saved presets
├─ sounds/               # alert audio files (wav/mp3)
└─ start.bat             # double‑click to start
```

> **Tip:** You can keep your videos anywhere, but `indata/` is the simplest place.

---

## Quick Start (5 minutes)

1. **Copy your model** (YOLO `.pt`) into `models/`.
2. **Put a video** into `indata/` (or use a camera/RTSP URL).
3. **Double‑click `start.bat`**.
4. In the app:

   * Pick your **model** and **classes** (person, car, …).
   * Choose a **source** (file, folder, or camera/URL).
   * Click **Start**.
5. The **Counters editor** opens — draw **Lines** (for crossings) and/or **Zones** (for inside/outside). Save & close.
6. Watch the **live preview**; the app will generate results in `output/`.

You can stop anytime with **Abort**.

---

## What you’ll see

* **Left panel (HUD):** current and maximum counts per class.
* **Right panel (HUD):**

  * **Lines:** AB / BA crossing totals for each line.
  * **Zones:** IN / OUT totals per zone.
  * **Per‑class breakdowns** for each zone, plus a **global summary**.
* **Overlay mode:** choose centroids or boxes; optional motion traces.

**Heatmap overlay:** press **`m`** to show/hide the heatmap during a run (if enabled).

---

## Outputs

All results are written to the **`output/`** folder. Main subfolders:

* **videos/** – annotated MP4s (if enabled).
* **events/** – CSV log of all events (lines & zones).
* **snapshot/** – optional still images captured on events (enable in Extras).
* **heatmap/** – heatmaps (only when **Create heatmap** = On).
* **summary/** – per‑run JSON/CSV summaries.
* **counters/** – saved line/zone layouts for the source.
* **temp/** – internal working files; safe to ignore.

**Naming pattern** (illustrative):

* We use the input **basename** (e.g., `myclip_1920x1080_30fps`) and a **run timestamp** `YYYYMMDD_HHMMSS`.
* Examples (no exact filenames):

  * `events/<basename>_<timestamp>_events.csv`
  * `videos/<basename>_annotated.mp4`
  * `summary/<basename>_<timestamp>_summary.(json|csv)`
  * `heatmap/<basename>_<timestamp>_heatmap[_overlay|_rgba].png`
  * `snapshot/<basename>_<timestamp>/*.jpg`

**CSV columns:** `frame, time_sec, timecode, track_id, class_id, class_name, event_type, counter_name, AB, BA, conf`

---er **Frame Skip** (>1) in Advanced to process fewer frames.

---

## Advanced Settings (overview)

* **Main tab:** detection thresholds, tracker tuning, overlay colors, trace length, anchor mode.
* **Extras tab:** HUD size (%), snapshot on events, heatmap controls.
* **Help** shows explanations for each option when focused.

> You don’t need to change these to get good results — the presets are sensible. Tweak only if necessary.

---

## Tips & Good Practice

* Draw lines cleanly across the flow you want to count; name them clearly (e.g., `Entrance`, `Exit`).
* For zones, cover the relevant area tightly — not the whole frame.
* Use **class filters** (checkboxes) to remove classes you don’t care about.
* If alerts feel too frequent, raise the **cooldown** or disable **loop**.
* If heatmaps look faint, increase **gain** or **window length**.

---

## Privacy & Security

* Everything runs **locally** on your machine.
* Your videos and outputs are **not uploaded**.

---

## Troubleshooting

* **No detections?** Try a higher Quality level or a more appropriate model.
* **Too slow?** Lower Quality, increase Frame Skip, or disable traces.
* **No sound?** Try a `.wav` file; use ▶ test in Advanced. Check system volume.
* **Heatmap not saving?** Ensure it’s enabled in Extras. Files appear under `output/heatmap/` only.
* **Right panel shows totals but no classes?** Run a bit longer; class breakdowns update as events happen.

If you get stuck, re‑run the app and watch the log area for hints.

---

## License (Commercial)

This software is licensed for **single‑user commercial use**. Redistribution, sharing, or repackaging of the **source code** is prohibited. For multi‑user/site licensing, contact the vendor.

You may use, publish, and monetize the **outputs** you create (videos, images, CSVs, heatmaps).

---

## Support

* Questions, licensing, or custom builds: **Lynx IT Solutions**
* Please include your Windows version and a short description of your use case.

---

## Recent Highlights

* Line crossing alert (single ping), plus zone alerts.
* Per‑zone **per‑class** IN/OUT breakdown with global summary in HUD.
* Heatmap toggle with **output isolation** to `output/heatmap/`.
* Synchronized Quality slider ↔ Advanced.
* Transparent heatmaps with true no‑data.
* Presets: Apply / Save / Load for repeatable runs.

## Folder layout (where files go)

```
Project root /
├─ start.bat                 # Double‑click to launch the app (Windows)
├─ indata/                   # Your input files (videos, images); you can point the GUI to any other folder as well
├─ models/                   # YOLO .pt (or .zip that contains a .pt)
├─ output/                   # Results for each run
│  ├─ videos/                # Annotated MP4s (if enabled in your build)
│  ├─ events/                # CSV_events.csv + optional snapshots/ per run
│  ├─ summary/               # Run summaries (JSON + CSV)
│  ├─ heatmap/               # Heatmaps (only when **Create heatmap** = On)
│  └─ temp/                  # Temporary files created on demand
├─ presets/                  # Your saved Advanced-options presets (.json)
├─ sounds/                   # .wav files for alerts (Test/Stop inside Advanced → Sound)
├─ src/                      # Application code (for reference; no need to edit to use the app)
├─ wheels/ , _pkgs/          # Bundled Python wheels/deps (don’t modify)
├─ Backups/ , reklam/ , temp/# Internal/support folders; safe to ignore
└─ README*.md                # This guide
```

**Notes**

* Heatmaps are created **only** if **Extras → Heatmap → Create heatmap = On**. When On, all heatmap files are written to `output/heatmap/` (overlay, raw, and RGBA); when Off, nothing is written there.
* Event CSV now includes direction for line crossings (AB/BA) and per‑event `counter_name` so you can filter by line or zone.
* HUD size (Extras → HUD) scales both the left "Now/Max" block and the right results panel.

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

## Outputs (in `output/`)

| File/Folder         | What it is                                                 |
| ------------------- | ---------------------------------------------------------- |
| `videos/*.mp4`      | Annotated video with overlays                              |
| `events/*.csv`      | Detailed events log (time, class, counter name, direction) |
| `events/snapshots/` | Optional still images when events happen                   |
| `heatmap/`          | Heatmaps (only when heatmap is enabled)                    |
| `summary/*.json`    | Summary of counts for the run                              |

**CSV columns include:** `frame, time_sec, timecode, track_id, class_id, class_name, event_type, counter_name, AB, BA, conf`

---

## Sound Alerts

* **Zone alerts:** play while objects are **inside** or **outside** a selected zone (looping or single ping with cooldown).
* **Line alerts:** **single ping** each time a line is crossed (AB/BA).
* Choose your sound file in **Advanced → Sound** and use the ▶ / ⏹ test buttons.

---

## Heatmaps (optional)

* Turn on in **Advanced → Extras → Heatmap**.
* When **off**: no heatmap files are written.
* When **on**: heatmaps are saved **only** in `output/heatmap/`.
* Options: intensity (alpha), contrast (gamma), point size (sigma), decay, rolling window length, save interval.
* Transparent output treats true zero as **no‑data** (not blue).

---

## Quality vs. Speed

* Use the **Quality slider (1–5)** on the main screen:

  * Lower = faster; Higher = more accurate.
* The slider also updates **Advanced** parameters (confidence, NMS, tile size, tracker settings).
* For long videos, consider **Frame Skip** (>1) in Advanced to process fewer frames.

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

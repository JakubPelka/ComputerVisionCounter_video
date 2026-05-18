# ComputerVisionCounter VIDEO

**Status:** EXPERIMENT / WORK IN PROGRESS  
**License:** MIT  
**Platform focus:** Windows 10/11, local desktop use  
**Main idea:** count objects in videos or live streams using YOLO-based detection, lines, zones, alerts and exportable results.

ComputerVisionCounter VIDEO is a local desktop tool for object counting and event analytics in video material. It is designed for practical experiments with computer vision workflows: entrance counting, zone occupancy, line crossings, simple movement statistics, snapshots and heatmap-style visual outputs.

The project is public and can be used, studied, modified and adapted under the MIT License. Model weights and input/output data are not included in the repository.

---

## What the application does

The application can:

- open video files, folders of videos, camera sources or stream-like sources,
- use YOLO-compatible model weights placed locally in `models/`,
- detect selected classes, for example people, cars or other objects supported by the model,
- let the user draw counting lines and polygon zones,
- count line crossings in both directions,
- count zone entries and exits,
- show a live preview with HUD overlays,
- optionally play sound alerts,
- optionally save event snapshots,
- optionally create heatmap outputs,
- export events and summaries to local output folders.

Everything runs locally. Videos, model weights and results are not uploaded by the application.

---

## Repository scope

This repository should contain source code, documentation and small configuration files only.

It should **not** contain:

- private videos,
- generated outputs,
- backup archives,
- old ZIP packages,
- local dependency folders,
- model weights,
- API keys, tokens or `.env` files,
- private/local paths,
- large test datasets.

Model files, videos and outputs should stay local and are ignored by Git.

---

## Recommended folder layout

```text
ComputerVisionCounter_video/
├─ README.md
├─ LICENSE
├─ CHANGELOG.md
├─ ROADMAP.md
├─ requirements.txt
├─ start.bat
├─ lista_filer.bat
├─ src/
│  └─ application source files
├─ docs/
│  ├─ architecture.md
│  ├─ development.md
│  └─ repository_hygiene.md
├─ models/
│  └─ README.md
├─ indata/
│  └─ README.md
├─ output/
│  └─ generated locally, ignored by Git
├─ presets/
│  └─ README.md
├─ sounds/
│  └─ README.md
└─ wheels/
   └─ README.md
```

`src/` contains the application code. This hygiene patch does not modify `src/`.

---

## Quick start

### 1. Install / prepare Python

Use Python 3.12 on Windows.

The current launcher uses `src/start.py`, which prepares a local `_pkgs/` dependency folder and then starts the application. This is useful for a self-contained local setup.

### 2. Add a model locally

Place your YOLO-compatible model file in:

```text
models/
```

Example:

```text
models/yolo11x.pt
```

Model files are ignored by Git and should not be committed.

### 3. Add input data locally

Put test videos in:

```text
indata/
```

You can also select files from another folder in the GUI.

Input videos are ignored by Git and should not be committed.

### 4. Start the app

On Windows, double-click:

```text
start.bat
```

The launcher calls:

```text
src/start.py
```

---

## Typical workflow

1. Select a local model from `models/`.
2. Select a video, folder, camera or stream source.
3. Choose the classes to detect.
4. Start the analysis.
5. Draw lines and/or zones in the counter editor.
6. Watch the live preview.
7. Stop the run when finished.
8. Review local results in `output/`.

---

## Outputs

Generated files are written locally, usually under `output/`.

Typical output types:

- annotated videos,
- event CSV files,
- JSON/CSV summaries,
- snapshots on events,
- heatmap images,
- saved counter layouts.

Outputs are ignored by Git.

---

## Models and licensing

This repository does not include model weights.

You are responsible for checking the license and usage conditions of any model you use, for example YOLO or custom-trained weights.

Do not commit model files to this repository.

---

## Development notes

The application is currently under active experimental development. Some features may be incomplete, unstable or awaiting verification.

Known current priorities:

- repository hygiene and removal of backup/generated files,
- clearer documentation,
- keeping model weights local,
- later review of large source files and GUI synchronization issues,
- later functional cleanup after the repository is safe and understandable.

See:

```text
ROADMAP.md
docs/architecture.md
docs/development.md
docs/repository_hygiene.md
```

---

## License

This project is licensed under the MIT License. See `LICENSE`.

---

## Disclaimer

This is an experimental computer vision tool. Detection and counting results depend on model quality, camera angle, image quality, object visibility, frame rate, occlusion and counting geometry. Always validate results before using them for decisions.

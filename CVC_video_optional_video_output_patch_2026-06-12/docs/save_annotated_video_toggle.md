# Save annotated video toggle

## Purpose

Long high-resolution videos and stream recordings can create very large annotated MP4 files. In many workflows, the user only needs processing results such as CSV/JSON summaries, event logs, zone metrics, snapshots or heatmaps.

This patch adds a UI checkbox:

```text
Save annotated video (large files)
```

The checkbox is enabled by default to preserve current behaviour.

## Behaviour

When enabled:

- the application opens a `VideoWriter`,
- annotated video is saved as before,
- CSV/JSON/snapshots/heatmaps/preview continue as before.

When disabled:

- no annotated MP4 is written,
- `VideoWriter` is not opened,
- frame processing still runs,
- live preview still works,
- event CSV and summary JSON/CSV still work,
- zone dwell/peak metrics still work,
- snapshots and heatmaps still work if enabled.

## Why this matters

For long videos or stream-like sources, annotated output can easily create tens of gigabytes of data. Disabling video output makes batch processing lighter and safer for disk space.

## Test checklist

- Start the app with the checkbox enabled and verify annotated video is created.
- Start the app with the checkbox disabled and verify no MP4 is created in `output/videos/`.
- Verify that existing CSV/JSON summaries still appear.
- Verify that zone metrics still appear when polygon zones are used.
- Verify that preview still works.
- Verify a short stream/camera source if available.

## Suggested commit

```text
fix: add option to disable annotated video output
```

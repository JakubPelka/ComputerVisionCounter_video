# Optimization audit — ComputerVisionCounter VIDEO

Status: post-`v0.1.0` planning and safe first patch.

## Current hotspot

The main hotspot is `src/cv_video_run.py`.

It currently handles too many domains in one file:

- Ultralytics result parsing,
- detection drawing fallback,
- writer safety,
- ByteTrack construction and update,
- line/zone counting,
- sound alerts,
- heatmap handling,
- video processing loop,
- snapshots and outputs.

This is not wrong for a working prototype, but it makes the application harder to optimize and harder to extend safely.

## Safe optimization targets

### 1. Tracker isolation

Move tracker creation/update logic out of `cv_video_run.py`.

Reason:

- current ByteTrack logic can stay stable,
- future OC-SORT can be tested behind the same interface,
- tracking experiments will not require editing the main loop every time.

### 2. Model inference context

Wrap direct model calls in `torch.inference_mode()` when torch is available.

Reason:

- avoids unnecessary autograd overhead,
- safe for inference-only video processing,
- should not change detection results.

### 3. Per-frame class filter

Avoid recreating `np.fromiter(selected_class_ids_set, dtype=int)` on every frame.

Reason:

- small but free performance gain,
- reduces avoidable per-frame allocations.

### 4. OpenCV runtime setup

Enable `cv2.setUseOptimized(True)`.

Reason:

- safe OpenCV optimization flag,
- does not change UI or counting logic.

Thread count should remain optional via environment variable, because forcing OpenCV threads can sometimes hurt responsiveness.

### 5. Later: detector adapter

Current parser `_parse_ultra_results()` handles boxes only.

For segmentation and RF-DETR, the project will need a normalized internal detection object:

```text
boxes:   Nx4 xyxy
scores:  N
cids:    N
masks:   optional N/H/W or polygon contours
ids:     optional tracker IDs
```

Do not implement this until the ByteTrack/default detection path is stable after the first refactor.

## Not included in this patch

- No UI changes.
- No segmentation drawing.
- No RF-DETR backend.
- No OC-SORT activation in UI.
- No rewrite of counting logic.
- No rewrite of heatmap logic.

## Recommended next steps

1. Apply this safe runtime patch.
2. Test one short video.
3. Commit if stable.
4. Then split additional code from `cv_video_run.py` gradually:
   - `cv_video_detection.py`,
   - `cv_video_tracking.py`,
   - `cv_video_counting.py`,
   - `cv_video_outputs.py`,
   - `cv_video_runtime.py`.

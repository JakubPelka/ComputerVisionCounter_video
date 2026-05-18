# Future computer vision engines — notes

These are future directions, not part of the safe runtime patch.

## OC-SORT

OC-SORT is interesting as a tracker candidate for crowded scenes, occlusion and less linear movement.

Roboflow describes OC-SORT as an extension of SORT with:

- Observation-Centric Re-Update,
- Observation-Centric Momentum,
- better handling of occlusion, erratic motion and uniform-looking objects.

Suggested project strategy:

1. Keep ByteTrack as default.
2. Add an internal tracker adapter.
3. Later expose tracker choice in Advanced settings:

```text
Tracker: ByteTrack / OC-SORT / none
```

4. Test on the same video and compare:

- ID switches,
- duplicate counts,
- missed zone events,
- line crossing stability,
- speed.

## RF-DETR

RF-DETR is interesting because it supports detection and instance segmentation through the `rfdetr` package.

Relevant notes:

- `pip install rfdetr`
- Apache 2.0 for the open-source package and Apache-designated model weights.
- Some Plus components and larger detection models have a different license.
- It returns `supervision.Detections` in examples, which is promising because this project already uses Supervision/ByteTrack concepts.

Suggested project strategy:

1. Add a detector adapter layer first.
2. Normalize outputs to internal arrays: boxes, scores, class IDs, optional masks.
3. Do not mix RF-DETR directly into `cv_video_run.py`.
4. Test RF-DETR detection first, segmentation later.

## SAM 3

SAM 3 is now integrated into Ultralytics and supports concept segmentation using text prompts, image exemplars and video tracking.

This is conceptually exciting, but it is a different workflow from standard YOLO detection:

- prompt-driven,
- potentially open-vocabulary,
- segmentation-first,
- likely heavier than simple box detection.

Suggested project strategy:

1. Treat SAM 3 as a separate experimental backend.
2. Start with offline video/frame experiments, not the main UI.
3. Test whether text-prompt segmentation is useful for real counting cases.
4. Only integrate after a normalized segmentation data structure exists.

## YOLO26 / YOLO26-seg

YOLO26 is relevant because it supports detection, segmentation, pose and OBB model variants.

For this project, the most relevant variants are:

```text
yolo26x.pt       # detection — already available locally for testing
yolo26x-seg.pt   # segmentation candidate
```

Important license note:

- Ultralytics YOLO26 is documented under AGPL-3.0 / Enterprise licensing.
- Keep model weights local and never commit them to the repository.
- For public or commercial distribution, licensing must be checked before bundling anything.

Suggested project strategy:

1. Test `yolo26x.pt` in the current detection workflow first.
2. Confirm that `_parse_ultra_results()` still handles YOLO26 outputs.
3. Add mask parsing later for `*-seg.pt` models.
4. Add mask overlay as optional mode after box counting remains stable.

## Segmentation roadmap

A practical segmentation path:

1. Keep current box detection/counting stable.
2. Add optional mask extraction from Ultralytics results.
3. Store masks internally but still count with existing anchor logic.
4. Add visual mask overlay.
5. Add optional mask-based zone overlap counting.
6. Add segmentation-based heatmap accumulation.

Do not jump directly to mask-based counting before the current box/anchor counting is tested after refactor.

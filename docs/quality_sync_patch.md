# Quality slider / Advanced settings sync patch

## Goal

Stabilize the relation between the main **Quality** slider and the **Advanced options** window without changing counting, tracking, heatmaps, alerts, geometry or output logic.

## Behavior after patch

- Moving the main Quality slider updates:
  - the preset label,
  - `app.adv_params`,
  - already-open Advanced fields: `imgsz`, `conf`, `iou`, `frame_skip`, `track_buffer`, `match_thresh`, `min_hits`.
- Pressing **Apply** in Advanced makes Advanced values the active source.
- Loading a preset applies it through the same path as **Apply**.
- Opening Advanced does not overwrite manually applied Advanced values unless the Quality slider is moved again.

## Files touched

- `src/cv_video.py`
- `src/cv_video_advanced_ui.py`

## Manual test checklist

1. Start app with `start.bat`.
2. Move Quality slider to 1, 3, 5.
3. Open Advanced and verify:
   - `imgsz`, `conf`, `iou`, `frame_skip`, `track_buffer`, `match_thresh`, `min_hits` match the main label.
4. Change one Advanced value manually, for example `conf`.
5. Click **Apply**.
6. Close and reopen Advanced.
7. Verify the manual value remains.
8. Move Quality slider again.
9. Open Advanced and verify the fields now match the selected preset again.
10. Run a short video to confirm the app still starts and processes normally.

## Suggested commit

```text
fix: synchronize quality slider and advanced parameters
```

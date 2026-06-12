# Patch: optional annotated video output

Urgent UI/runtime patch for `ComputerVisionCounter_video`.

## Goal

Add a checkbox that controls whether annotated video is saved.

Default remains ON, so current behaviour is preserved.

## Files modified by the patch script

```text
src/cv_video.py
src/cv_video_run.py
```

## Files added by this patch package

```text
tools/apply_optional_video_output_patch.py
docs/save_annotated_video_toggle.md
```

## Apply

Copy the patch package into the repository root and run:

```bat
python tools\apply_optional_video_output_patch.py
```

The script creates backups in:

```text
TEMP/patch_backups/
```

Do not commit `TEMP/`.

## Test

1. Run with `Save annotated video` enabled. Confirm MP4 is created.
2. Run with `Save annotated video` disabled. Confirm MP4 is not created.
3. Confirm event CSV and summary JSON/CSV still appear.
4. Confirm zone metrics still appear if polygon zones are used.
5. Confirm live preview still works.

## Suggested commit

```text
fix: add option to disable annotated video output
```

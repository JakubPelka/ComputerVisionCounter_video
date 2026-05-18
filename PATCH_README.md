# CVC VIDEO patch — Quality slider / Advanced settings sync

This patch is intended for `ComputerVisionCounter_video` after release `v0.1.1`.

## What to copy

Copy these folders/files into the repository root:

```text
tools/apply_quality_sync_patch.py
docs/quality_sync_patch.md
```

## Apply

From the repository root:

```bat
python tools\apply_quality_sync_patch.py
```

The script modifies:

```text
src/cv_video.py
src/cv_video_advanced_ui.py
```

It creates local backups under:

```text
TEMP/patch_backups/
```

Do not commit `TEMP/`.

## Test

Use the checklist in:

```text
docs/quality_sync_patch.md
```

## Suggested commit

```text
fix: synchronize quality slider and advanced parameters
```

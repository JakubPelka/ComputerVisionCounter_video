# Hotfix — `selected_class_ids_arr` NameError

This is a small repair patch for the previous safe runtime optimization package.

It fixes:

```text
[ERROR] name 'selected_class_ids_arr' is not defined
```

## What it changes

Only this file is modified:

```text
src/cv_video_run.py
```

The script either:

1. adds the missing definition immediately after:

```python
selected_class_ids_set = set(selected_idx or [])
```

or, if that anchor is not found, restores the previous inline class-filter expression.

It does **not** change UI, counting logic, tracking, heatmaps, sounds, output structure or model loading.

## How to apply

Copy the `tools/` folder into the repository root and run:

```bat
python tools\fix_selected_class_ids_arr.py
```

Then test with the same video and settings that produced the error.

## Suggested commit message

```text
fix: define selected class filter array in video runner
```

## If something still fails

The script creates a backup next to `src/cv_video_run.py`, for example:

```text
src/cv_video_run.py.bak_selected_class_ids_20260518_083000
```

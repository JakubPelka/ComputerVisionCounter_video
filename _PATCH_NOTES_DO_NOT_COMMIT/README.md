# Patch notes — live event CSV

Do not commit this folder.

## Apply

```bat
python tools\apply_live_events_patch.py
```

## Commit after successful test

```text
fix: write events incrementally during processing
```

## Keep in repo

```text
src/cv_video_event_log.py
docs/live_event_logging.md
```

## Optional cleanup after applying

You may remove this helper after successful commit:

```text
tools/apply_live_events_patch.py
```

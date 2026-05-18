# Zone metrics output

This document describes the additional zone analytics added after `v0.1.1`.

The feature adds two CSV outputs under `output/summary/` when the run contains at least one polygon zone.

## Files

```text
<source>_<run_tag>_zone_dwell_times.csv
<source>_<run_tag>_zone_class_peaks.csv
```

## Dwell time per track and zone

`zone_dwell_times.csv` contains one row per tracked object and zone.

Columns:

```text
source, run_tag, zone_id, zone_name, track_id, class_id, class_name,
first_frame, last_frame, first_time_sec, last_time_sec,
seconds_inside, frames_inside, visits,
first_timecode, last_timecode, first_clock, last_clock
```

Notes:

- `frames_inside` counts processed frames, not necessarily every original video frame when frame skipping is enabled.
- `seconds_inside` is based on video timestamps / stream clock deltas between processed frames.
- If a tracked object leaves a zone and later re-enters, the gap is not counted. The `visits` column is incremented.
- Accuracy depends on tracking quality. Track ID switches will split dwell time between IDs.

## Peak concurrent objects per class and zone

`zone_class_peaks.csv` contains the maximum number of simultaneous tracked objects per class and zone.

Columns:

```text
source, run_tag, zone_id, zone_name, class_id, class_name,
max_concurrent, first_peak_frame, first_peak_time_sec,
first_peak_timecode, first_peak_clock
```

Only the first frame/time where the maximum was reached is stored.

## Scope

This feature does not change:

- existing event CSV output,
- existing summary JSON/CSV output,
- line counting,
- zone in/out counting,
- HUD behavior,
- alert behavior,
- model loading,
- tracker configuration.

It only adds additional analytics files based on the already-tracked detections and the already-defined polygon zones.

# Live event CSV logging

This patch adds incremental event logging for long runs and stream processing.

## Why

Previous behavior saved the events CSV at the end of the run. This is fine for short video files, but risky for:

- long high-resolution runs,
- camera/RTSP streams,
- outdoor monitoring,
- unstable hardware or capture devices,
- future integrations with tools such as InfluxDB/Grafana.

If the app crashed before the final save, the in-memory event list could be lost.

## What changed

A live event CSV file is now created as soon as a source run starts:

```text
output/events/<source>_<run_tag>_events_live.csv
```

New events are appended during processing, directly after line/zone event detection.

The existing final events CSV is still created at the end:

```text
output/events/<source>_<run_tag>_events.csv
```

## CSV columns

```text
frame,time_sec,timecode,clock,track_id,class_id,class_name,event_type,counter_name,conf,AB,BA
```

## Notes

- The live CSV is meant as a crash-resilient event log.
- The final CSV remains the regular end-of-run export.
- The live CSV opens, writes, flushes and closes each batch of new events.
- `fsync` is enabled by default in the live writer to improve crash resilience.
- For very event-heavy workflows this may add some I/O overhead, but normal line/zone events should be sparse enough.
- Future InfluxDB integration should read/forward the same event records or write from the same event hook.

## Test checklist

1. Run with line counting and check that `_events_live.csv` appears immediately after the first event.
2. Run with polygon zone events and check that `zone_in` / `zone_out` are appended.
3. Abort a run after events have occurred and verify that live CSV still contains records.
4. Check that final `_events.csv` is still created at the end.
5. Confirm that normal summary and zone metrics outputs still work.

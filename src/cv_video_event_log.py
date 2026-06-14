# cv_video_event_log.py
# Incremental / crash-resilient event CSV writer for ComputerVisionCounter VIDEO.
from __future__ import annotations

from pathlib import Path
import csv
import os
from typing import Iterable, Mapping, Any


class LiveEventCsvWriter:
    """Append event rows to CSV as they happen.

    The existing end-of-run events CSV is still useful as a final summary/export.
    This writer is intentionally simple and robust:
    - creates the output file immediately,
    - writes a stable header,
    - appends only new event rows,
    - opens/closes/flushed each batch so data survives most crashes.
    """

    COLUMNS = [
        "frame",
        "time_sec",
        "timecode",
        "clock",
        "track_id",
        "class_id",
        "class_name",
        "event_type",
        "counter_name",
        "conf",
        "AB",
        "BA",
    ]

    def __init__(self, path: str | Path, *, fsync: bool = True):
        self.path = self._collision_path(Path(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = bool(fsync)
        self._init_file()

    @staticmethod
    def _collision_path(path: Path) -> Path:
        """Return a non-existing path by adding _001, _002, ... if needed."""
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        for i in range(1, 10000):
            candidate = parent / f"{stem}_{i:03d}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not create collision-free live events CSV path: {path}")

    def _init_file(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            writer.writeheader()
            f.flush()
            if self.fsync:
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass

    def write_events(self, events: Iterable[Mapping[str, Any]]) -> int:
        rows = list(events or [])
        if not rows:
            return 0
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            for ev in rows:
                writer.writerow(self._clean_event(ev))
            f.flush()
            if self.fsync:
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        return len(rows)

    def _clean_event(self, ev: Mapping[str, Any]) -> dict:
        out = {}
        for key in self.COLUMNS:
            val = ev.get(key, "")
            out[key] = "" if val is None else val
        return out

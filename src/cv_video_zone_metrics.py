# cv_video_zone_metrics.py
# Additional per-zone analytics for ComputerVisionCounter VIDEO.
# Safe helper module: no UI code, no model code, no tracker code.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple


@dataclass
class _DwellRecord:
    zone_id: int
    zone_name: str
    track_id: int
    class_id: int
    class_name: str
    first_frame: int
    last_frame: int
    first_time_sec: float
    last_time_sec: float
    first_timecode: str = ""
    last_timecode: str = ""
    first_clock: str = ""
    last_clock: str = ""
    frames_inside: int = 1
    seconds_inside: float = 0.0
    visits: int = 1

    def touch(self, frame: int, time_sec: float, timecode: str, clock: str, was_active: bool) -> None:
        if was_active:
            dt = float(time_sec) - float(self.last_time_sec)
            if dt > 0:
                self.seconds_inside += dt
        else:
            self.visits += 1
        self.last_frame = int(frame)
        self.last_time_sec = float(time_sec)
        self.last_timecode = str(timecode or "")
        self.last_clock = str(clock or "")
        self.frames_inside += 1

    def to_row(self, source: str, run_tag: str) -> Dict[str, Any]:
        return {
            "source": source,
            "run_tag": run_tag,
            "zone_id": int(self.zone_id),
            "zone_name": self.zone_name,
            "track_id": int(self.track_id),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "first_frame": int(self.first_frame),
            "last_frame": int(self.last_frame),
            "first_time_sec": float(self.first_time_sec),
            "last_time_sec": float(self.last_time_sec),
            "seconds_inside": round(float(self.seconds_inside), 3),
            "frames_inside": int(self.frames_inside),
            "visits": int(self.visits),
            "first_timecode": self.first_timecode,
            "last_timecode": self.last_timecode,
            "first_clock": self.first_clock,
            "last_clock": self.last_clock,
        }


class ZoneMetrics:
    """Track zone dwell-time and peak concurrent occupancy.

    The class is intentionally small and passive. The main runner feeds it one
    processed frame at a time. It does not change counting logic, alerts, HUD,
    tracker behavior, model inference or existing event CSV output.

    Notes:
    - `frames_inside` counts processed frames, not original video frames when
      frame skipping is active.
    - `seconds_inside` is based on timestamp deltas between consecutive
      processed frames where the same track remains inside the same zone.
    - If a track leaves and later re-enters a zone, the gap is not counted.
    """

    DWELL_COLUMNS = [
        "source",
        "run_tag",
        "zone_id",
        "zone_name",
        "track_id",
        "class_id",
        "class_name",
        "first_frame",
        "last_frame",
        "first_time_sec",
        "last_time_sec",
        "seconds_inside",
        "frames_inside",
        "visits",
        "first_timecode",
        "last_timecode",
        "first_clock",
        "last_clock",
    ]

    PEAK_COLUMNS = [
        "source",
        "run_tag",
        "zone_id",
        "zone_name",
        "class_id",
        "class_name",
        "max_concurrent",
        "first_peak_frame",
        "first_peak_time_sec",
        "first_peak_timecode",
        "first_peak_clock",
    ]

    def __init__(self) -> None:
        self._records: Dict[Tuple[int, int], _DwellRecord] = {}
        self._active_keys: Set[Tuple[int, int]] = set()
        self._peaks: Dict[Tuple[int, int, str], Dict[str, Any]] = {}

    @staticmethod
    def _safe_class_name(names: Any, cid: int) -> str:
        try:
            if isinstance(names, dict):
                return str(names.get(int(cid), str(cid)))
            if isinstance(names, (list, tuple)):
                return str(names[int(cid)]) if 0 <= int(cid) < len(names) else str(cid)
        except Exception:
            pass
        return str(cid)

    @staticmethod
    def _zone_name(zone_cfg: Dict[str, Any], zone_idx: int) -> str:
        try:
            name = str(zone_cfg.get("name", "")).strip()
            return name if name else f"zone_{zone_idx}"
        except Exception:
            return f"zone_{zone_idx}"

    def update_frame(
        self,
        *,
        frame_idx: int,
        time_sec: float,
        timecode: str,
        clock: str,
        zones_cfg: Sequence[Dict[str, Any]],
        det_ids: Iterable[Any],
        det_cids: Iterable[Any],
        det_confs: Iterable[Any] | None,
        anchors: Sequence[Tuple[float, float]],
        names: Any,
        point_in_polygon: Callable[[Tuple[float, float], Sequence[Tuple[float, float]]], bool],
    ) -> None:
        if not zones_cfg:
            self._active_keys = set()
            return

        current_active: Set[Tuple[int, int]] = set()
        current_counts: Dict[Tuple[int, int, str], int] = {}

        # Convert to lists once to tolerate numpy arrays, generators and lists.
        ids = list(det_ids) if det_ids is not None else []
        cids = list(det_cids) if det_cids is not None else []
        pts = list(anchors) if anchors is not None else []

        for tid_raw, cid_raw, anchor in zip(ids, cids, pts):
            try:
                tid = int(tid_raw)
                cid = int(cid_raw)
                cx, cy = float(anchor[0]), float(anchor[1])
            except Exception:
                continue

            class_name = self._safe_class_name(names, cid)

            for zi, zone in enumerate(zones_cfg or []):
                try:
                    zpts = zone.get("pts", [])
                    inside = bool(point_in_polygon((cx, cy), zpts))
                except Exception:
                    inside = False
                if not inside:
                    continue

                zone_name = self._zone_name(zone, zi)
                key = (int(zi), int(tid))
                current_active.add(key)

                rec = self._records.get(key)
                if rec is None:
                    self._records[key] = _DwellRecord(
                        zone_id=int(zi),
                        zone_name=zone_name,
                        track_id=int(tid),
                        class_id=int(cid),
                        class_name=class_name,
                        first_frame=int(frame_idx),
                        last_frame=int(frame_idx),
                        first_time_sec=float(time_sec),
                        last_time_sec=float(time_sec),
                        first_timecode=str(timecode or ""),
                        last_timecode=str(timecode or ""),
                        first_clock=str(clock or ""),
                        last_clock=str(clock or ""),
                    )
                else:
                    rec.touch(
                        frame=int(frame_idx),
                        time_sec=float(time_sec),
                        timecode=str(timecode or ""),
                        clock=str(clock or ""),
                        was_active=key in self._active_keys,
                    )

                peak_key = (int(zi), int(cid), class_name)
                current_counts[peak_key] = int(current_counts.get(peak_key, 0)) + 1

        for (zi, cid, class_name), count in current_counts.items():
            zone_name = self._zone_name(zones_cfg[zi], zi)
            peak = self._peaks.get((zi, cid, class_name))
            if peak is None or count > int(peak.get("max_concurrent", 0)):
                self._peaks[(zi, cid, class_name)] = {
                    "zone_id": int(zi),
                    "zone_name": zone_name,
                    "class_id": int(cid),
                    "class_name": class_name,
                    "max_concurrent": int(count),
                    "first_peak_frame": int(frame_idx),
                    "first_peak_time_sec": float(time_sec),
                    "first_peak_timecode": str(timecode or ""),
                    "first_peak_clock": str(clock or ""),
                }

        self._active_keys = current_active

    def dwell_rows(self, *, source: str, run_tag: str) -> List[Dict[str, Any]]:
        rows = [rec.to_row(source=source, run_tag=run_tag) for rec in self._records.values()]
        return sorted(rows, key=lambda r: (r["zone_id"], r["class_name"], r["track_id"]))

    def peak_rows(self, *, source: str, run_tag: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for peak in self._peaks.values():
            row = {"source": source, "run_tag": run_tag}
            row.update(peak)
            rows.append(row)
        return sorted(rows, key=lambda r: (r["zone_id"], r["class_name"]))

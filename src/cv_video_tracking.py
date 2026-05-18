# cv_video_tracking.py — tracker adapter for ComputerVisionCounter VIDEO
from __future__ import annotations

import inspect
from typing import Any

import numpy as np

try:
    import supervision as sv
except Exception:  # pragma: no cover - optional runtime dependency
    sv = None


def make_tracker(
    tracker_name: str = "bytetrack",
    *,
    conf: float = 0.5,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    min_hits: int = 2,
    fps: float = 30.0,
) -> Any | None:
    """Create a tracker while keeping ByteTrack as the stable default.

    Current behaviour is preserved when called with tracker_name="bytetrack".
    OC-SORT is prepared as an optional future backend, but only activates if the
    relevant third-party package is installed and explicitly selected later.
    """
    name = (tracker_name or "bytetrack").lower().replace("-", "").replace("_", "")
    if name in {"ocsort", "ocsorttracker"}:
        tr = _make_ocsort(track_buffer=track_buffer, match_thresh=match_thresh, min_hits=min_hits, fps=fps)
        if tr is not None:
            return tr
    return _make_bytetrack(conf=conf, track_buffer=track_buffer, match_thresh=match_thresh, min_hits=min_hits, fps=fps)


def _make_bytetrack(conf: float, track_buffer: int, match_thresh: float, min_hits: int, fps: float) -> Any | None:
    if sv is None or not hasattr(sv, "ByteTrack"):
        return None
    try:
        params = inspect.signature(sv.ByteTrack.__init__).parameters
    except Exception:
        try:
            return sv.ByteTrack()
        except Exception:
            return None

    kwargs: dict[str, Any] = {}
    if "track_thresh" in params:
        kwargs["track_thresh"] = max(0.05, min(float(conf), 0.99))
    if "track_buffer" in params:
        kwargs["track_buffer"] = int(track_buffer)
    if "match_thresh" in params:
        kwargs["match_thresh"] = float(match_thresh)
    if "min_hits" in params:
        kwargs["min_hits"] = int(min_hits)
    if "mot20" in params:
        kwargs["mot20"] = False
    if "frame_rate" in params:
        kwargs["frame_rate"] = float(max(1.0, fps))

    for candidate in (
        kwargs,
        {k: kwargs[k] for k in ("track_thresh", "track_buffer", "match_thresh") if k in kwargs},
        {},
    ):
        try:
            return sv.ByteTrack(**candidate)
        except TypeError:
            continue
        except Exception:
            continue
    return None


def _make_ocsort(track_buffer: int, match_thresh: float, min_hits: int, fps: float) -> Any | None:
    """Best-effort optional OC-SORT factory.

    This is intentionally defensive because OC-SORT implementations expose slightly
    different class names and constructor parameters. It is not used by default.
    """
    candidates = [
        ("trackers", "OCSORT"),
        ("trackers", "OCSort"),
        ("trackers.ocsort", "OCSORT"),
        ("trackers.ocsort", "OCSort"),
        ("boxmot", "OCSORT"),
        ("boxmot", "OCSort"),
    ]
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except Exception:
            continue
        try:
            params = inspect.signature(cls.__init__).parameters
        except Exception:
            params = {}
        kwargs: dict[str, Any] = {}
        if "track_buffer" in params:
            kwargs["track_buffer"] = int(track_buffer)
        if "max_age" in params:
            kwargs["max_age"] = int(track_buffer)
        if "match_thresh" in params:
            kwargs["match_thresh"] = float(match_thresh)
        if "iou_threshold" in params:
            kwargs["iou_threshold"] = float(match_thresh)
        if "min_hits" in params:
            kwargs["min_hits"] = int(min_hits)
        if "frame_rate" in params:
            kwargs["frame_rate"] = float(max(1.0, fps))
        try:
            return cls(**kwargs)
        except Exception:
            try:
                return cls()
            except Exception:
                continue
    return None


def track_update(tracker: Any | None, boxes: np.ndarray, scores: np.ndarray, cids: np.ndarray):
    """Update tracker and return boxes, scores, class IDs and track IDs.

    Return contract matches the existing cv_video_run._track_update function:
    (boxes, scores, cids, det_ids_or_None)
    """
    if tracker is None or boxes is None or boxes.shape[0] == 0:
        return boxes, scores, cids, None

    # Supervision-compatible trackers: ByteTrack and similar.
    if sv is not None:
        try:
            dets = sv.Detections(
                xyxy=np.asarray(boxes, dtype=float),
                class_id=np.asarray(cids, dtype=int) if cids is not None else None,
                confidence=np.asarray(scores, dtype=float) if scores is not None else None,
            )
        except Exception:
            dets = None
        if dets is not None:
            out = _update_supervision_like_tracker(tracker, dets, boxes, scores, cids)
            if out is not None:
                return out

    # Array/list-based trackers, used by many SORT/OC-SORT implementations.
    out = _update_array_like_tracker(tracker, boxes, scores, cids)
    if out is not None:
        return out

    return boxes, scores, cids, None


def _update_supervision_like_tracker(tracker: Any, dets: Any, boxes, scores, cids):
    for method_name in ("update_with_detections", "update"):
        method = getattr(tracker, method_name, None)
        if method is None:
            continue
        try:
            tracked = method(dets)
        except Exception:
            continue
        parsed = _parse_supervision_result(tracked, boxes, scores, cids)
        if parsed is not None:
            return parsed
    return None


def _parse_supervision_result(tracked: Any, boxes, scores, cids):
    if hasattr(tracked, "xyxy") and getattr(tracked, "tracker_id", None) is not None:
        t_boxes = np.asarray(tracked.xyxy, dtype=int)
        t_ids = np.asarray(tracked.tracker_id, dtype=int)
        t_cids = _safe_take_attr(tracked, "class_id", cids, len(t_ids), int)
        t_scores = _safe_take_attr(tracked, "confidence", scores, len(t_ids), float)
        return t_boxes, t_scores, t_cids, t_ids

    if isinstance(tracked, (list, tuple)) and tracked and hasattr(tracked[0], "track_id"):
        t_boxes, t_ids = [], []
        for item in tracked:
            if hasattr(item, "tlbr"):
                xyxy = item.tlbr
            elif hasattr(item, "to_tlbr"):
                xyxy = item.to_tlbr()
            else:
                continue
            t_boxes.append([int(v) for v in xyxy])
            t_ids.append(int(getattr(item, "track_id")))
        if t_boxes:
            t_ids_arr = np.asarray(t_ids, dtype=int)
            return (
                np.asarray(t_boxes, dtype=int),
                _safe_take(scores, len(t_ids_arr), float),
                _safe_take(cids, len(t_ids_arr), int),
                t_ids_arr,
            )
    return None


def _update_array_like_tracker(tracker: Any, boxes, scores, cids):
    arr = _detections_to_sort_array(boxes, scores, cids)
    for method_name in ("update", "update_with_detections"):
        method = getattr(tracker, method_name, None)
        if method is None:
            continue
        for payload in (arr, boxes):
            try:
                tracked = method(payload)
            except Exception:
                continue
            parsed = _parse_array_tracker_result(tracked, scores, cids)
            if parsed is not None:
                return parsed
    return None


def _detections_to_sort_array(boxes, scores, cids):
    boxes_f = np.asarray(boxes, dtype=float)
    if boxes_f.size == 0:
        return np.empty((0, 6), dtype=float)
    scores_f = np.asarray(scores, dtype=float).reshape(-1, 1) if scores is not None else np.ones((len(boxes_f), 1))
    cids_f = np.asarray(cids, dtype=float).reshape(-1, 1) if cids is not None else -np.ones((len(boxes_f), 1))
    return np.concatenate([boxes_f[:, :4], scores_f, cids_f], axis=1)


def _parse_array_tracker_result(tracked, scores, cids):
    if tracked is None:
        return None
    try:
        arr = np.asarray(tracked)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 5:
        return None
    t_boxes = arr[:, :4].astype(int)
    # Common formats: x1,y1,x2,y2,track_id or x1,y1,x2,y2,score,track_id,class_id
    if arr.shape[1] >= 7:
        t_scores = arr[:, 4].astype(float)
        t_ids = arr[:, 5].astype(int)
        t_cids = arr[:, 6].astype(int)
    else:
        t_ids = arr[:, 4].astype(int)
        t_scores = _safe_take(scores, len(t_ids), float)
        t_cids = _safe_take(cids, len(t_ids), int)
    return t_boxes, t_scores, t_cids, t_ids


def _safe_take_attr(obj: Any, attr: str, fallback, n: int, dtype):
    val = getattr(obj, attr, None)
    if val is not None:
        try:
            return np.asarray(val, dtype=dtype)
        except Exception:
            pass
    return _safe_take(fallback, n, dtype)


def _safe_take(arr, n: int, dtype):
    if arr is None:
        return None
    try:
        return np.asarray(arr[:n], dtype=dtype)
    except Exception:
        return None

# cv_video_performance.py — small safe runtime helpers for ComputerVisionCounter VIDEO
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any


def configure_opencv_runtime(app: Any | None = None) -> None:
    """Enable safe OpenCV optimizations without changing application behaviour.

    This function intentionally does not force a thread count by default. If needed,
    set environment variable CVC_OPENCV_THREADS to a positive integer.
    """
    try:
        import cv2
    except Exception:
        return

    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass

    raw_threads = os.environ.get("CVC_OPENCV_THREADS", "").strip()
    if raw_threads:
        try:
            n_threads = int(raw_threads)
            if n_threads > 0:
                cv2.setNumThreads(n_threads)
                _log_once(app, f"[PERF] OpenCV threads set to {n_threads}")
        except Exception:
            _log_once(app, f"[PERF] Ignored invalid CVC_OPENCV_THREADS={raw_threads!r}")


def inference_context():
    """Return torch.inference_mode() when torch is available, otherwise no-op.

    Ultralytics often handles this internally, but wrapping the call is safe and can
    reduce autograd overhead for direct model calls.
    """
    try:
        import torch
        return torch.inference_mode()
    except Exception:
        return nullcontext()


def _log_once(app: Any | None, message: str) -> None:
    if app is None:
        return
    seen = getattr(app, "_perf_log_seen", None)
    if seen is None:
        seen = set()
        try:
            setattr(app, "_perf_log_seen", seen)
        except Exception:
            return
    if message in seen:
        return
    seen.add(message)
    try:
        app._log(message)
    except Exception:
        pass

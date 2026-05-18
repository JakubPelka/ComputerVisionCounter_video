"""Apply a small, reversible runtime optimization patch to ComputerVisionCounter VIDEO.

Run from repository root:
    python tools/apply_safe_runtime_patch.py

The script:
1. adds a backup: src/cv_video_run.py.bak_before_runtime_patch
2. replaces inline tracker helper functions with thin wrappers around cv_video_tracking.py
3. wraps model inference in torch.inference_mode() when torch is available
4. precomputes the selected-class numpy array outside the per-frame loop
5. enables safe OpenCV runtime optimization through cv_video_performance.py
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ROOT / "src" / "cv_video_run.py"
BACKUP_PATH = ROOT / "src" / "cv_video_run.py.bak_before_runtime_patch"

TRACKER_BLOCK_RE = re.compile(
    r"# [─-]+\n# tracker helpers \(ByteTrack via Supervision\)\n# [─-]+\n"
    r"def _make_bytetrack\(conf, track_buffer, match_thresh, min_hits, fps\):.*?"
    r"def _track_update\(tracker, boxes, scores, cids\):.*?"
    r"(?=\n# [─-]+\n# counting \+ alerts)",
    re.S,
)

TRACKER_REPLACEMENT = '''# ──────────────────────────────────────────────────────────────────────────────
# tracker helpers (delegated to cv_video_tracking)
# ──────────────────────────────────────────────────────────────────────────────
def _make_bytetrack(conf, track_buffer, match_thresh, min_hits, fps):
    # Preserve current behaviour: ByteTrack remains the default tracker.
    try:
        from cv_video_tracking import make_tracker
        return make_tracker(
            "bytetrack",
            conf=conf,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            min_hits=min_hits,
            fps=fps,
        )
    except Exception:
        return None


def _track_update(tracker, boxes, scores, cids):
    try:
        from cv_video_tracking import track_update
        return track_update(tracker, boxes, scores, cids)
    except Exception:
        return boxes, scores, cids, None

'''


def main() -> int:
    if not RUN_PATH.exists():
        print(f"ERROR: {RUN_PATH} not found. Run this script from repository root.")
        return 1

    text = RUN_PATH.read_text(encoding="utf-8")
    original = text

    if "cv_video_tracking import track_update" not in text:
        text, n = TRACKER_BLOCK_RE.subn(TRACKER_REPLACEMENT, text, count=1)
        if n != 1:
            print("WARNING: Tracker helper block was not patched. Pattern not found.")

    if "configure_opencv_runtime(app)" not in text:
        text = re.sub(
            r"(def run\(app, sources, outp: Path, selected_idx\):\s*\n\s*t0 = time\.time\(\)\s*\n)",
            r"\1    try:\n        from cv_video_performance import configure_opencv_runtime\n        configure_opencv_runtime(app)\n    except Exception:\n        pass\n",
            text,
            count=1,
        )

    if "selected_class_ids_arr =" not in text:
        text = re.sub(
            r"(\n\s*cur_time_sec = 0\.0 # updated every frame\s*\n)",
            r"\n    selected_class_ids_arr = np.fromiter(selected_class_ids_set, dtype=int) if selected_class_ids_set else None\n\1",
            text,
            count=1,
        )

    text = text.replace(
        "mask_keep = np.isin(cids, np.fromiter(selected_class_ids_set, dtype=int))",
        "mask_keep = np.isin(cids, selected_class_ids_arr)",
    )

    if "with inference_context():" not in text:
        text = re.sub(
            r"(?m)^(\s*)res = app\.model\(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False\)",
            r"\1from cv_video_performance import inference_context\n\1with inference_context():\n\1    res = app.model(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)",
            text,
            count=1,
        )

    if text == original:
        print("No changes applied. File may already be patched.")
        return 0

    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(original, encoding="utf-8")
        print(f"Backup written: {BACKUP_PATH}")
    RUN_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"Patched: {RUN_PATH}")
    print("Next: run start.bat and test one short video before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

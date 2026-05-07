# cv_video_core.py — shared constants and utilities
from __future__ import annotations
from pathlib import Path
import json, zipfile, math, time   # ← add math, time
import cv2
import pandas as pd
import torch

# ====== CONSTANTS ======
SUPPORTED_VID_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".ts")
MODEL_DIRNAME = "models"

VIDEO_PRESETS = {
    1: {"imgsz": 320,  "conf": 0.50, "iou": 0.60, "frame_skip": 2, "track_buffer": 5,  "match_thresh": 0.80, "min_hits": 2},
    2: {"imgsz": 640,  "conf": 0.55, "iou": 0.55, "frame_skip": 2, "track_buffer": 30, "match_thresh": 0.80, "min_hits": 2},
    3: {"imgsz": 960,  "conf": 0.60, "iou": 0.50, "frame_skip": 1, "track_buffer": 60, "match_thresh": 0.78, "min_hits": 2},
    4: {"imgsz": 1280, "conf": 0.65, "iou": 0.50, "frame_skip": 1, "track_buffer": 75, "match_thresh": 0.75, "min_hits": 3},
    5: {"imgsz": 1280, "conf": 0.70, "iou": 0.45, "frame_skip": 0, "track_buffer": 90, "match_thresh": 0.75, "min_hits": 3},
}
DEFAULT_QUALITY = 1
DEFAULT_TRACKER = "bytetrack"

LINE_MIN_GAP_FRAMES_DEFAULT  = 8
LINE_MIN_SEP_PX_DEFAULT      = 12
ZONE_MIN_GAP_FRAMES_DEFAULT  = 6

# ====== UTIL ======
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def device_auto_str() -> str:
    return "0" if torch.cuda.is_available() else "cpu"

def score_weight_name(p: Path) -> int:
    name = p.name.lower(); score = 0
    try: score += int(p.stat().st_size // (1024*1024))
    except Exception: pass
    if "x.pt" in name: score += 100
    if "l.pt" in name: score += 80
    if "m.pt" in name: score += 60
    return score

def find_best_weights(models_dir: Path) -> Path | None:
    cands = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.zip"))
    if not cands: return None
    cands.sort(key=score_weight_name, reverse=True); return cands[0]

def resolve_weights_to_pt(path: Path, extract_dir: Path) -> Path:
    if path.suffix.lower() == ".pt": return path
    if path.suffix.lower() == ".zip":
        ensure_dir(extract_dir)
        with zipfile.ZipFile(path, "r") as z: z.extractall(extract_dir)
        pts = list(extract_dir.rglob("*.pt"))
        if not pts: raise RuntimeError("No .pt file found inside the .zip archive")
        pts.sort(key=score_weight_name, reverse=True); return pts[0]
    raise ValueError("Select a .pt or .zip file")

def numbered_path(path: Path) -> Path:
    if not path.exists(): return path
    stem, suf = path.stem, path.suffix; i = 1
    while True:
        cand = path.with_name(f"{stem}_{i}{suf}")
        if not cand.exists(): return cand
        i += 1

def _safe_fps(val, default=30.0):
    try:
        f = float(val)
    except Exception:
        return float(default)
    if not math.isfinite(f) or f <= 0 or f > 240:
        return float(default)
    return float(f)

def open_video_writer_collision(out_dir: Path, base_name: str,
                                size_wh: tuple[int, int],
                                fps: float,
                                fourcc: str = "mp4v") -> tuple[Path, cv2.VideoWriter]:
    """
    Create an mp4 writer in out_dir/videos with a unique filename if needed.
    FPS is respected exactly as passed (validated by _safe_fps).
    """
    out_dir = Path(out_dir)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    ext = ".mp4"
    stem = f"{base_name}_annotated"
    path = vid_dir / f"{stem}{ext}"
    if path.exists():
        # collision-safe naming
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = vid_dir / f"{stem}_{ts}{ext}"

    W, H = int(size_wh[0]), int(size_wh[1])
    fcc = cv2.VideoWriter_fourcc(*fourcc)
    fps = _safe_fps(fps)
    writer = cv2.VideoWriter(str(path), fcc, fps, (W, H))
    if not writer or not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path} (fps={fps}, size={(W,H)})")
    return path, writer

def save_json_collision(obj, path: Path) -> Path:
    try:
        if path.exists():
            try: path.unlink()
            except Exception: pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        alt = numbered_path(path)
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return alt

def save_csv_collision(df: pd.DataFrame, path: Path) -> Path:
    try:
        if path.exists():
            try: path.unlink()
            except Exception: pass
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except Exception:
        alt = numbered_path(path)
        df.to_csv(alt, index=False, encoding="utf-8")
        return alt

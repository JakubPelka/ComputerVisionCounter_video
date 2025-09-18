# cv_video_core.py
# Minimalny rdzeń przetwarzania wideo (detekcja YOLO + opcjonalne trackowanie + eventy)
# Cel: logika niezależna od GUI/Tkinter. Wywoływana z cv_video.py (App).
#
# Zakres (Faza 1):
# - Ładowanie modelu (Ultralytics YOLO)
# - Pętla przetwarzania wideo (źródło: plik lub kamera)
# - Opcjonalne trackowanie (YOLO .track -> ByteTrack) z id
# - Callbacki: on_progress, on_detection, on_frame (do podglądu/overlay w GUI)
# - Prosty licznik przekroczeń linii (jeśli przekazane linie)
#
# W kolejnych krokach można tu przenieść bardziej złożone funkcje (ROI/poligony,
# konfigurację presetów, kompletne metryki, zapis CSV/Video itd.).

from __future__ import annotations
import sys, time, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Dict, Any, Union

import cv2
import numpy as np

# YOLO (Ultralytics) – opcjonalny; rdzeń ma działać gdy pakiet jest w ./_pkgs lub globalnie
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _ultra_err = e

# Supervision – opcjonalnie; na razie nie wymagamy (kolejne fazy)
try:
    import supervision as sv  # noqa: F401
except Exception:
    sv = None

# Typy
Point = Tuple[int, int]
Line = Tuple[Point, Point]

@dataclass
class VideoSettings:
    weights: Union[str, Path]
    imgsz: int = 640
    conf: float = 0.5
    iou: float = 0.6
    use_tracker: bool = True           # użyj YOLO.track (ByteTrack) aby uzyskać id torów
    tracker_config: Optional[str] = None  # np. 'bytetrack.yaml' (w ultralytics)
    frame_skip: int = 1                # co ile klatek wykonywać detekcję
    class_filter: Optional[List[int]] = None  # np. [0] dla 'person'
    device: Optional[str] = None       # 'cpu' | 'cuda' | None -> auto
    max_frames: Optional[int] = None   # ograniczenie liczby klatek (debug)
    overlay_mode: str = "centroid"     # "centroid"|"boxes"|"boxes_conf"

@dataclass
class LineCounter:
    lines: List[Line] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.counts or len(self.counts) != len(self.lines):
            self.counts = [0] * len(self.lines)

    @staticmethod
    def _side_of_line(a: Point, b: Point, p: Point) -> float:
        # zwraca znak położenia punktu względem linii AB (iloczyn wektorowy)
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

    def update_crossings(self, prev: Dict[int, Point], curr: Dict[int, Point]) -> List[Tuple[int, int]]:
        """Zlicz przekroczenia: jeśli tor (id) zmienił stronę względem linii.
        Zwraca listę (line_idx, track_id) dla nowych przekroczeń."""
        events = []
        for tid, cp in curr.items():
            if tid not in prev: 
                continue
            pp = prev[tid]
            for i, (A, B) in enumerate(self.lines):
                s1 = self._side_of_line(A, B, pp)
                s2 = self._side_of_line(A, B, cp)
                if s1 == 0 or s2 == 0:
                    continue
                # różne znaki -> przekroczenie
                if s1 * s2 < 0:
                    self.counts[i] += 1
                    events.append((i, tid))
        return events

def _xyxy_to_centroid(xyxy: np.ndarray) -> Point:
    x1, y1, x2, y2 = xyxy
    cX = int((x1 + x2) / 2)
    cY = int((y1 + y2) / 2)
    return (cX, cY)

def _draw_overlay(frame: np.ndarray, boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray, ids: Optional[np.ndarray], mode: str) -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]
    if boxes is None:
        return img
    for i, box in enumerate(boxes):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        if mode in ("boxes", "boxes_conf"):
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        if mode == "boxes_conf":
            label = f"{int(cls[i])}:{conf[i]:.2f}"
            cv2.putText(img, label, (p1[0], max(0, p1[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        if mode.startswith("centroid"):
            c = _xyxy_to_centroid(box)
            cv2.circle(img, c, 3, (255, 0, 0), -1)
            if ids is not None:
                cv2.putText(img, f"ID {int(ids[i])}", (c[0]+4, c[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return img

def _filter_classes(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray, ids: Optional[np.ndarray], allowed: Optional[List[int]]):
    if allowed is None:
        return boxes, conf, cls, ids
    mask = np.isin(cls.astype(int), np.array(allowed, dtype=int))
    idx = np.where(mask)[0]
    if ids is not None:
        return boxes[idx], conf[idx], cls[idx], ids[idx]
    return boxes[idx], conf[idx], cls[idx], None

def _iter_cap_frames(cap: cv2.VideoCapture) -> Iterable[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame

def run_video_pipeline(
    source: Union[str, int],
    settings: VideoSettings,
    lines: Optional[List[Line]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    on_detection: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
) -> Dict[str, Any]:
    """Główny rdzeń. Zwraca słownik z prostym podsumowaniem."""
    if YOLO is None:
        raise ImportError(f"Ultralytics nie jest dostępny: {_ultra_err if '_ultra_err' in globals() else 'brak pakietu'}")

    model = YOLO(str(settings.weights))

    # Otwarcie źródła
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        total_frames = None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = cv2.VideoCapture(str(source))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if not cap.isOpened():
        raise RuntimeError(f"Nie udało się otworzyć źródła: {source}")

    # Liczniki i struktury pomocnicze
    frame_idx = 0
    line_counter = LineCounter(lines or [])
    prev_centroids: Dict[int, Point] = {}
    summary = {
        "frames": 0,
        "detected": 0,
        "crossings_per_line": [0]*len(line_counter.lines),
        "duration_s": 0.0,
        "fps": fps,
    }

    t0 = time.time()

    # Tryb trackujący Ultralytics – zwraca results z .boxes.id
    # Uwaga: model.track sam otwiera źródło, więc gdy używamy go, nie iterujemy po cap.
    if settings.use_tracker:
        tracker_cfg = settings.tracker_config or "bytetrack.yaml"
        stream = model.track(
            source=source,
            stream=True,
            imgsz=settings.imgsz,
            conf=settings.conf,
            iou=settings.iou,
            tracker=tracker_cfg,
            device=settings.device,
            verbose=False,
        )
        for res in stream:
            # res: ultralytics.engine.results.Results
            boxes = res.boxes
            if boxes is None or boxes.id is None or len(boxes) == 0:
                frame = res.orig_img if hasattr(res, "orig_img") else None
                if frame is not None and on_frame:
                    on_frame(frame)
                frame_idx += 1
                if on_progress and total_frames:
                    on_progress(min(100.0, 100.0 * frame_idx / max(1, total_frames)))
                continue

            b = boxes.xyxy.cpu().numpy()
            c = boxes.conf.cpu().numpy()
            k = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)

            # filtrowanie klas
            b, c, k, ids = _filter_classes(b, c, k, ids, settings.class_filter)

            # centroidy aktualne
            curr_centroids = {int(ids[i]): _xyxy_to_centroid(b[i]) for i in range(len(b))}

            # przekroczenia linii
            events = line_counter.update_crossings(prev_centroids, curr_centroids)
            summary["crossings_per_line"] = line_counter.counts
            if events and on_detection:
                for li, tid in events:
                    on_detection({"type": "line_cross", "line_idx": li, "track_id": int(tid)})

            # overlay i callback z klatką
            frame = res.orig_img if hasattr(res, "orig_img") else None
            if frame is not None:
                overlay = _draw_overlay(frame, b, c, k, ids, settings.overlay_mode)
                if on_frame:
                    on_frame(overlay)

            prev_centroids = curr_centroids
            summary["frames"] += 1
            summary["detected"] += len(b)

            frame_idx += 1
            if on_progress and total_frames:
                on_progress(min(100.0, 100.0 * frame_idx / max(1, total_frames)))

            if settings.max_frames and summary["frames"] >= settings.max_frames:
                break

    else:
        # Ręczna pętla: detekcja co N klatek; bez id (brak trackera)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % max(1, settings.frame_skip) == 0:
                res = model.predict(frame, imgsz=settings.imgsz, conf=settings.conf, iou=settings.iou, device=settings.device, verbose=False)[0]
                boxes = res.boxes
                if boxes is not None and len(boxes) > 0:
                    b = boxes.xyxy.cpu().numpy()
                    c = boxes.conf.cpu().numpy()
                    k = boxes.cls.cpu().numpy().astype(int)
                    b, c, k, _ = _filter_classes(b, c, k, None, settings.class_filter)
                    overlay = _draw_overlay(frame, b, c, k, None, settings.overlay_mode)
                else:
                    overlay = frame
                if on_frame:
                    on_frame(overlay)
                summary["detected"] += 0 if boxes is None else len(boxes)
            else:
                if on_frame:
                    on_frame(frame)

            summary["frames"] += 1
            frame_idx += 1
            if on_progress and total_frames:
                on_progress(min(100.0, 100.0 * frame_idx / max(1, total_frames)))
            if settings.max_frames and summary["frames"] >= settings.max_frames:
                break

    cap.release()
    summary["duration_s"] = time.time() - t0
    return summary

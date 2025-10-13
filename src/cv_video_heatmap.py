# cv_video_heatmap.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Iterable, List, Dict, Tuple, Optional

def _gaussian_kernel2d(sigma: int) -> np.ndarray:
    sigma = max(1, int(sigma))
    r = int(3 * sigma)
    k = 2 * r + 1
    x = cv2.getGaussianKernel(k, sigma)
    ker = (x @ x.T).astype(np.float32)
    ker /= (ker.max() + 1e-6)
    return ker

class DetectionHeatmap:
    """Per-frame centroid “splat” heatmap with optional decay and AOI mask."""
    def __init__(self, width: int, height: int, sigma: int = 8, decay: float = 0.0):
        self.W = int(width)
        self.H = int(height)
        self.decay = float(max(0.0, min(decay, 1.0)))
        self.kernel = _gaussian_kernel2d(int(sigma))
        self.r = self.kernel.shape[0] // 2
        self.HM = np.zeros((self.H, self.W), np.float32)
        self.mask_bool: Optional[np.ndarray] = None

    def set_sigma(self, sigma: int):
        self.kernel = _gaussian_kernel2d(int(sigma))
        self.r = self.kernel.shape[0] // 2

    def set_decay(self, per_frame_decay: float):
        self.decay = float(max(0.0, min(per_frame_decay, 1.0)))

    def set_mask(self, mask: np.ndarray | None):
        if mask is None:
            self.mask_bool = None
            return
        m = (mask.astype(np.uint8) > 0)
        if m.shape != self.HM.shape:
            m = cv2.resize(m.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(bool)
        self.mask_bool = m

    def add_points(self, pts: Iterable[Tuple[int, int]]):
        if pts is None:
            return
        if self.decay > 0.0:
            self.HM *= (1.0 - self.decay)

        r = self.r
        H, W = self.H, self.W
        k = self.kernel

        for (x, y) in pts:
            x = int(x); y = int(y)
            if x < 0 or y < 0 or x >= W or y >= H:
                continue
            x0 = max(0, x - r); x1 = min(W, x + r + 1)
            y0 = max(0, y - r); y1 = min(H, y + r + 1)
            kx0 = r - (x - x0); kx1 = kx0 + (x1 - x0)
            ky0 = r - (y - y0); ky1 = ky0 + (y1 - y0)

            roi = self.HM[y0:y1, x0:x1]
            add = k[ky0:ky1, kx0:kx1]
            if self.mask_bool is not None:
                mroi = self.mask_bool[y0:y1, x0:x1]
                if not mroi.any():
                    continue
                roi[mroi] += add[mroi]
            else:
                roi += add

    def _colorize(self, normalize: bool = True, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        hm = self.HM
        if normalize:
            mx = float(hm.max())
            if mx < 1e-6:
                norm = np.zeros_like(hm, np.uint8)
            else:
                norm = np.clip(255.0 * (hm / mx), 0, 255).astype(np.uint8)
        else:
            norm = np.clip(hm, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, colormap)

    def render_overlay(self, frame_bgr: np.ndarray, alpha: float = 0.5,
                       normalize: bool = True, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        alpha = float(max(0.0, min(alpha, 1.0)))
        cm = self._colorize(normalize=normalize, colormap=colormap)
        if cm.shape[:2] != frame_bgr.shape[:2]:
            cm = cv2.resize(cm, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(frame_bgr, 1.0 - alpha, cm, alpha, 0.0)

    def save_png(self, path, normalize: bool = True, colormap: int = cv2.COLORMAP_JET) -> bool:
        cm = self._colorize(normalize=normalize, colormap=colormap)
        return bool(cv2.imwrite(str(path), cm))

def build_mask_from_zones(zones: List[Dict], width: int, height: int) -> np.ndarray:
    H, W = int(height), int(width)
    mask = np.zeros((H, W), np.uint8)
    if not zones:
        return mask
    for z in zones:
        pts = np.array(z.get("points", []), np.int32).reshape(-1, 1, 2)
        mode = str(z.get("mode", "polygon")).lower().strip()
        if pts.shape[0] < 2:
            continue
        if mode == "polygon" and pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 1)
        else:
            thick = int(z.get("thickness", 8))
            p = pts.reshape(-1, 2)
            for i in range(len(p) - 1):
                x1, y1 = map(int, p[i].tolist())
                x2, y2 = map(int, p[i + 1].tolist())
                cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=max(1, thick))
    return mask

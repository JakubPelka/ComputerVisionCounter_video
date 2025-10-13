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

    # --- knobs ---------------------------------------------------------------
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

    # --- accumulate ----------------------------------------------------------
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

    # --- rendering -----------------------------------------------------------
    def _normalized(self) -> np.ndarray:
        mx = float(self.HM.max())
        if mx < 1e-6:
            return np.zeros_like(self.HM, np.float32)
        return (self.HM / mx).astype(np.float32)

    def colorize_bgr(self, normalize: bool = True, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Classic BGR heatmap (opaque)."""
        if normalize:
            norm = (self._normalized() * 255.0).astype(np.uint8)
        else:
            norm = np.clip(self.HM, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, colormap)

    def colorize_rgba(self, alpha_scale: float = 1.0, gamma: float = 1.0,
                      thresh: float = 1e-6, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Transparent zeros: returns BGRA where A = alpha_scale * pow(norm, gamma).
        thresh: values <= thresh become alpha=0 (true 'no data').
        """
        norm = self._normalized()
        if gamma != 1.0:
            norm = np.power(norm, float(max(1e-6, gamma)))
        a = np.where(norm > float(thresh), np.clip(norm * float(alpha_scale), 0, 1.0), 0.0).astype(np.float32)

        cm = cv2.applyColorMap(np.clip((norm * 255.0).astype(np.uint8), 0, 255), colormap)
        # Build BGRA with alpha in 0..255
        a8 = (a * 255.0).astype(np.uint8)
        return np.dstack([cm, a8])  # BGRA

    def render_overlay_masked(self, frame_bgr: np.ndarray, alpha: float = 0.5,
                              gamma: float = 1.0, thresh: float = 1e-6,
                              colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Per-pixel alpha blend: only pixels with heat > thresh affect the frame.
        No blue cast on the background.
        """
        h, w = frame_bgr.shape[:2]
        rgba = self.colorize_rgba(alpha_scale=float(alpha), gamma=gamma, thresh=thresh, colormap=colormap)
        if rgba.shape[0] != h or rgba.shape[1] != w:
            rgba = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_LINEAR)

        rgb = rgba[:, :, :3].astype(np.float32)
        a = (rgba[:, :, 3:4].astype(np.float32) / 255.0)  # HxWx1
        base = frame_bgr.astype(np.float32)
        out = base * (1.0 - a) + rgb * a
        return np.clip(out, 0, 255).astype(np.uint8)

    # --- save ----------------------------------------------------------------
    def save_png(self, path, normalize: bool = True, colormap: int = cv2.COLORMAP_JET) -> bool:
        """Opaque BGR PNG (kept for backward compatibility)."""
        bgr = self.colorize_bgr(normalize=normalize, colormap=colormap)
        return bool(cv2.imwrite(str(path), bgr))

    def save_png_rgba(self, path, alpha_scale: float = 1.0, gamma: float = 1.0, thresh: float = 1e-6,
                      colormap: int = cv2.COLORMAP_JET) -> bool:
        """Transparent zeros PNG (BGRA)."""
        rgba = self.colorize_rgba(alpha_scale=alpha_scale, gamma=gamma, thresh=thresh, colormap=colormap)
        return bool(cv2.imwrite(str(path), rgba))

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
                x2, y2 = map(int, p[i + 1]).tolist()
                cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=max(1, thick))
    return mask

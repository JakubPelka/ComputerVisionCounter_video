from __future__ import annotations
import tkinter as tk
import cv2
from PIL import Image, ImageTk

# ---- suppression helpers -----------------------------------------------------
def preview_suppress(self, ms: int = 2000):
    import time
    self._preview_suppress_until = time.monotonic() + (ms / 1000.0)

def preview_is_suppressed(self) -> bool:
    import time
    return getattr(self, "_preview_suppress_until", 0.0) > time.monotonic()

# ---- window lifecycle --------------------------------------------------------
def preview_ensure(self):
    """Create/activate the LIVE preview window. ESC / close => abort."""
    if preview_is_suppressed(self):
        return

    def _abort_from_preview(_evt=None):
        try:
            preview_suppress(self, 2000)
            preview_destroy(self)
            self.abort()
        except Exception:
            try: preview_destroy(self)
            except Exception: pass

    if getattr(self, "_preview_win", None) and self._preview_win.winfo_exists():
        try:
            w = self._preview_win
            w.deiconify(); w.lift(); w.focus_force()
            if getattr(self, "_preview_lbl", None):
                self._preview_lbl.focus_set()
            w.attributes("-topmost", True)
            w.after(250, lambda: w.attributes("-topmost", False))
        except Exception:
            pass
        return

    win = tk.Toplevel(self)
    win.title("Preview (LIVE)")
    win.geometry("960x620")

    win.protocol("WM_DELETE_WINDOW", _abort_from_preview)
    win.bind("<Escape>", _abort_from_preview)
    win.bind("<Control-w>", _abort_from_preview)

    lbl = tk.Label(win, anchor="center", bg="#111")
    lbl.pack(fill="both", expand=True)

    self._preview_win = win
    self._preview_lbl = lbl
    self._preview_last_bgr = None
    self._preview_after_id = None

    win.bind("<Configure>", lambda e: preview_on_resize(self))

    try:
        win.update_idletasks()
        win.deiconify(); win.lift(); win.focus_force(); lbl.focus_set()
        win.attributes("-topmost", True)
        win.after(250, lambda: win.attributes("-topmost", False))
    except Exception:
        pass

def preview_on_resize(self):
    try:
        if preview_is_suppressed(self):
            return
        if getattr(self, "_preview_last_bgr", None) is not None and \
           getattr(self, "_preview_win", None) and self._preview_win.winfo_exists():
            preview_show(self, self._preview_last_bgr)
    except Exception:
        pass

def preview_destroy(self):
    try:
        if getattr(self, "_preview_after_id", None):
            try: self.after_cancel(self._preview_after_id)
            except Exception: pass
            self._preview_after_id = None

        win = getattr(self, "_preview_win", None)
        if win:
            try: win.unbind("<Configure>")
            except Exception: pass
            try:
                if win.winfo_exists():
                    win.destroy()
            except Exception:
                pass
    finally:
        self._preview_win = None
        self._preview_lbl = None
        self._preview_last_bgr = None

# ---- frame rendering ---------------------------------------------------------
def preview_show(self, frame_bgr):
    if not self.preview_enabled.get() or preview_is_suppressed(self):
        return

    def _do():
        try:
            if not (getattr(self, "_preview_win", None) and self._preview_win.winfo_exists()):
                preview_ensure(self)
                if not (getattr(self, "_preview_win", None) and self._preview_win.winfo_exists()):
                    return

            win = self._preview_win
            lbl = self._preview_lbl

            self._preview_last_bgr = frame_bgr

            win.update_idletasks()
            tw = max(100, win.winfo_width()  - 12)
            th = max(100, win.winfo_height() - 12)
            target_w = int(tw * 0.90)
            target_h = int(th * 0.90)

            H, W = frame_bgr.shape[:2]
            scale = min(target_w / float(W), target_h / float(H))
            nw = max(1, int(W * scale)); nh = max(1, int(H * scale))

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            disp = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
            rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._preview_imgtk = imgtk
            lbl.config(image=imgtk)
        except Exception:
            pass

    try:
        if getattr(self, "_preview_after_id", None):
            try: self.after_cancel(self._preview_after_id)
            except Exception: pass
            self._preview_after_id = None
        self._preview_after_id = self.after(0, _do)
    except Exception:
        pass

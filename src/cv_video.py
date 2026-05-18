# cv_video.py — compact UI + dynamic class grid + presets + interactive help
# Default folders + custom alert sound + robust entrypoint with error popup/log
from __future__ import annotations
import threading, sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cv_video_gui import ScrollableFrame, CounterEditor, AppUIMixin  # GUI utils
from cv_video_run import run as core_run
from cv_video_sound import SoundPlayer
from cv_video_core import (
    ensure_dir,
    score_weight_name, find_best_weights, resolve_weights_to_pt,
    SUPPORTED_VID_EXTS, MODEL_DIRNAME,
    VIDEO_PRESETS, DEFAULT_QUALITY, DEFAULT_TRACKER,
    LINE_MIN_GAP_FRAMES_DEFAULT, LINE_MIN_SEP_PX_DEFAULT, ZONE_MIN_GAP_FRAMES_DEFAULT,
)

# preview mixin
from cv_video_preview import (
    preview_ensure, preview_show, preview_on_resize, preview_destroy,
    preview_suppress, preview_is_suppressed,
)

# advanced UI builder (moved out)
from cv_video_advanced_ui import build_advanced_settings

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
try:
    from ultralytics.nn.modules import block as _ublock
    if not hasattr(_ublock, "C3k2"):
        raise RuntimeError("Ultralytics without YOLOv11 support (missing C3k2).")
except Exception as e:
    print("Ultralytics check:", e, file=sys.stderr)

# ---------- Branding & default paths ----------
from paths import REPO_ROOT, INPUTS as DEFAULT_IN_DIR, OUTPUTS as DEFAULT_OUT_DIR, MODELS as DEFAULT_MODELS_DIR, SOUNDS as DEFAULT_SOUNDS_DIR

APP_NAME = "ComputerVisionCounter VIDEO"
PROJECT_ROOT = REPO_ROOT


class App(AppUIMixin, tk.Tk):
    CLASS_COL_MIN = 2
    CLASS_COL_MAX = 12
    CLASS_CELL_PX = 160

    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} — line/zone counting (YOLO + ByteTrack)")
        self.geometry("980x720")

        # --- GUI variables (with better defaults) ---
        self.input_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value=str(DEFAULT_OUT_DIR))
        models_default = DEFAULT_MODELS_DIR
        self.weights_path = tk.StringVar(value=str(find_best_weights(models_default) or models_default))

        self.quality = tk.IntVar(value=DEFAULT_QUALITY)
        self.tracker_kind = tk.StringVar(value=DEFAULT_TRACKER)
        self.overlay_mode = tk.StringVar(value="centroid")

        self.model = None; self.names = None; self.class_vars = []
        self.selected_files = []
        self._class_cols = 5

        # --- advanced / preset-able ---
        self.advanced_override = False
        self.adv_params = {
            "imgsz": VIDEO_PRESETS[DEFAULT_QUALITY]["imgsz"],
            "conf": VIDEO_PRESETS[DEFAULT_QUALITY]["conf"],
            "iou": VIDEO_PRESETS[DEFAULT_QUALITY]["iou"],
            "frame_skip": VIDEO_PRESETS[DEFAULT_QUALITY]["frame_skip"],
            "track_buffer": VIDEO_PRESETS[DEFAULT_QUALITY]["track_buffer"],
            "match_thresh": VIDEO_PRESETS[DEFAULT_QUALITY]["match_thresh"],
            "min_hits": VIDEO_PRESETS[DEFAULT_QUALITY]["min_hits"],
            "line_min_gap": LINE_MIN_GAP_FRAMES_DEFAULT,
            "line_min_sep": LINE_MIN_SEP_PX_DEFAULT,
            "zone_min_gap": ZONE_MIN_GAP_FRAMES_DEFAULT,
            "preview_enabled": True,
            "trace_enabled": True,
            "trace_len": 24,
            "anchor_mode": "center",
            "ghost_margin": 24,
            "alert_enabled": False,
            "alert_freeze_s": 2,
            "alert_zone_inside": 1,
            "alert_sound": "",
            "alert_loop": True,
            "trace_color": "auto",
            "trace_thickness": 2,
            "overlay_frame_color": "auto",
            "overlay_frame_thickness": 2,
        }

        # bindings (used by run)
        self.preview_enabled = tk.BooleanVar(value=self.adv_params["preview_enabled"])
        self.trace_enabled   = tk.BooleanVar(value=self.adv_params["trace_enabled"])
        self.trace_len       = tk.IntVar(value=self.adv_params["trace_len"])
        self.anchor_mode     = tk.StringVar(value=self.adv_params["anchor_mode"])
        self.ghost_margin    = tk.IntVar(value=self.adv_params["ghost_margin"])
        self.alert_enabled   = tk.BooleanVar(value=self.adv_params["alert_enabled"])
        self.alert_freeze_s  = tk.IntVar(value=self.adv_params["alert_freeze_s"])
        self.alert_sound     = tk.StringVar(value=self.adv_params["alert_sound"])
        self.alert_loop      = tk.BooleanVar(value=self.adv_params["alert_loop"])

        # progress / worker / preview
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = tk.StringVar(value="Ready.")
        self.abort_event = threading.Event()
        self.worker_done = threading.Event()
        self.worker_thread = None
        self._progress_indeterminate = False

        self._preview_win = None
        self._preview_lbl = None
        self._preview_imgtk = None

        # build UI
        self.build_ui_compact()
        self._autoload_best_model()

        # Attach preview helpers as bound methods (mixin-style)
        App._ensure_preview_window = preview_ensure
        App._show_preview_bgr     = preview_show
        App._on_preview_resize    = preview_on_resize
        App._destroy_preview_window = preview_destroy
        App._preview_suppress     = preview_suppress
        App._preview_is_suppressed = preview_is_suppressed


    # small helper for test sound
    def _play_test_sound(self, path: str | None):
        """Play selected alert sound once (non-blocking)."""
        try:
            p = (path or "").strip()
            if not p:
                messagebox.showinfo("Alert sound", "Pick a sound file first.")
                return
            if not Path(p).exists():
                messagebox.showerror("Alert sound", f"File not found:\n{p}")
                return
            sp = SoundPlayer(p)
            try:
                sp.stop()
            except Exception:
                pass
            sp.play_once()
            self._log(f"[TEST] {Path(p).name} — backends: {sp.describe_backends()}")
        except Exception as e:
            try:
                messagebox.showerror("Alert sound", str(e))
            except Exception:
                pass

    # ========== UI (compact version) ==========
    def build_ui_compact(self):
        root = tk.Frame(self); root.pack(fill="both", expand=True, padx=8, pady=6)

        # Section: paths
        self._row_browse(root, "Video folder (input):", self.input_dir, self.browse_input, is_dir=True)
        f_files = tk.Frame(root); f_files.pack(fill="x", pady=2)
        tk.Button(f_files, text="Select video files…", command=self.browse_files).pack(side="left")
        self.files_label = tk.Label(f_files, text="— none —"); self.files_label.pack(side="left", padx=8)
        tk.Button(f_files, text="Clear selection", command=self.clear_files).pack(side="left", padx=(8,0))

        self._row_browse(root, "Output folder (default: ./output):", self.output_dir, self.browse_output, is_dir=True)
        self._row_browse(root, "Weights (.pt/.zip):", self.weights_path, self.browse_weights, is_dir=False)

        # Source
        srcf = tk.LabelFrame(root, text="Input source"); srcf.pack(fill="x", pady=4)
        self.src_mode = tk.StringVar(value="files"); self.cam_index = tk.StringVar(value="0"); self.url_input = tk.StringVar(value="")
        def _src_toggle(*_):
            mf = self.src_mode.get()
            cam_ent.config(state=("normal" if mf=="camera" else "disabled"))
            url_ent.config(state=("normal" if mf=="url" else "disabled"))
        tk.Radiobutton(srcf, text="Files",  variable=self.src_mode, value="files", command=_src_toggle).pack(side="left", padx=6)
        tk.Radiobutton(srcf, text="Camera", variable=self.src_mode, value="camera", command=_src_toggle).pack(side="left", padx=6)
        tk.Label(srcf, text="Index:").pack(side="left")
        cam_ent = tk.Entry(srcf, width=4, textvariable=self.cam_index); cam_ent.pack(side="left", padx=(0,8))
        tk.Radiobutton(srcf, text="RTSP/HTTP URL", variable=self.src_mode, value="url", command=_src_toggle).pack(side="left", padx=6)
        url_ent = tk.Entry(srcf, textvariable=self.url_input); url_ent.pack(side="left", fill="x", expand=True, padx=(0,6))
        _src_toggle()

        # Quality (slider) + label
        qf = tk.Frame(root); qf.pack(fill="x", pady=4)
        tk.Label(qf, text="Quality (1 = faster/lower, 5 = ULTRA)").pack(side="left")
        tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality, command=self._on_quality_slider_changed).pack(side="left", fill="x", expand=True, padx=8)
        self.preset_label = tk.Label(qf, text=""); self.preset_label.pack(side="left")
        self._update_preset_label()

        # Overlay + Tracker
        ot = tk.Frame(root); ot.pack(fill="x", pady=4)
        ov = tk.LabelFrame(ot, text="Overlay"); ov.pack(side="left", fill="x", expand=True, padx=(0,6))
        tk.Radiobutton(ov, text="Centroids", variable=self.overlay_mode, value="centroid").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boxes", variable=self.overlay_mode, value="boxes").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boxes + conf", variable=self.overlay_mode, value="boxes_conf").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Polygons", variable=self.overlay_mode, value="polygon").pack(side="left", padx=6)

        tr = tk.LabelFrame(ot, text="Tracker"); tr.pack(side="left", padx=(6,0))
        tk.Radiobutton(tr, text="ByteTrack", variable=self.tracker_kind, value="bytetrack").pack(side="left", padx=6)
        tk.Radiobutton(tr, text="BoT-SORT", variable=self.tracker_kind, value="botsort").pack(side="left", padx=6)

        # Class selection — scrollable + dynamic columns + TOGGLE strip
        lf = tk.LabelFrame(root, text="Class selection (after loading weights)")
        lf.pack(fill="both", expand=True, pady=4)

        # Toggle row
        tog = tk.Frame(lf); tog.pack(fill="x", pady=(4, 0))
        tk.Label(tog, text="Toggle:").pack(side="left")
        tk.Button(tog, text="All",   width=6, command=self._toggle_all_classes).pack(side="left", padx=(6, 0))
        tk.Button(tog, text="None",  width=6, command=self._toggle_none_classes).pack(side="left", padx=(6, 0))
        tk.Button(tog, text="Invert",width=6, command=self._toggle_invert_classes).pack(side="left", padx=(6, 0))

        # Scroll area for checkboxes
        self.classes_scroll = ScrollableFrame(lf, height=220)
        self.classes_scroll.pack(fill="both", expand=True)
        self.classes_scroll.canvas.bind("<Configure>", self._on_classes_canvas_config)

        # Controls + Progress
        bf = tk.Frame(root); bf.pack(fill="x", pady=6)
        self.btn_start = tk.Button(bf, text="START", command=self.start); self.btn_start.pack(side="left")
        tk.Button(bf, text="Advanced options…", command=self.open_advanced).pack(side="left", padx=8)
        self.btn_abort = tk.Button(bf, text="ABORT", state="disabled", command=self.abort); self.btn_abort.pack(side="left", padx=(8,0))

        pf = tk.Frame(root); pf.pack(fill="x", pady=(0,4))
        self.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=self.progress_var, mode="determinate")
        self.progressbar.pack(fill="x", side="left", expand=True)
        self._progress_indeterminate = False
        tk.Label(pf, textvariable=self.progress_label, width=28, anchor="w").pack(side="left", padx=6)

        # Log
        logf = tk.Frame(root); logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, height=8, state="normal"); self.log.pack(fill="both", expand=True)

    # ========== helpers (UI) ==========
    def _row_browse(self, parent, label, var, cmd, is_dir=True):
        f = tk.Frame(parent); f.pack(fill="x", pady=3)
        tk.Label(f, text=label, width=26, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(f, text="Browse…", command=cmd).pack(side="left")

    def _current_quality_preset(self) -> dict:
        """Return the currently selected main Quality preset as a plain dict."""
        try:
            q = int(self.quality.get())
        except Exception:
            q = DEFAULT_QUALITY
        return dict(VIDEO_PRESETS.get(q, VIDEO_PRESETS[DEFAULT_QUALITY]))

    def _sync_quality_preset_to_adv(self, *, mark_manual_override: bool = False) -> dict:
        """Copy the main Quality preset into runtime advanced params and open Advanced fields.

        The main Quality slider is treated as the source of truth when the user moves it.
        Manual Advanced values are preserved until the slider is moved again.
        """
        p = self._current_quality_preset()
        for k in ("imgsz", "conf", "iou", "frame_skip", "track_buffer", "match_thresh", "min_hits"):
            if k in p:
                self.adv_params[k] = p[k]

        # False means: use main Quality preset as the current source.
        # True is set by Advanced Apply/Load and means: preserve manual Advanced values.
        self.advanced_override = bool(mark_manual_override)

        # If the Advanced window is currently open, update its fields immediately.
        var_map = {
            "imgsz": "v_imgsz",
            "conf": "v_conf",
            "iou": "v_iou",
            "frame_skip": "v_frame_skip",
            "track_buffer": "v_track_buffer",
            "match_thresh": "v_match_thresh",
            "min_hits": "v_min_hits",
        }
        for key, attr in var_map.items():
            var = getattr(self, attr, None)
            if var is not None and hasattr(var, "set") and key in p:
                try:
                    var.set(str(p[key]))
                except Exception:
                    pass
        return p

    def _on_quality_slider_changed(self, *_):
        """Handle main Quality slider changes and keep Advanced values in sync."""
        self._sync_quality_preset_to_adv(mark_manual_override=False)
        self._update_preset_label()

    def _update_preset_label(self):
        p = self._current_quality_preset()
        try:
            self.preset_label.config(text=(f"imgsz={p['imgsz']} conf={p['conf']} iou={p['iou']} "
                                           f"skip={p['frame_skip']} buf={p['track_buffer']} "
                                           f"match={p['match_thresh']} hits={p['min_hits']}"))
        except Exception:
            pass

    def _toggle_all_classes(self):
        for _nm, var, _idx in getattr(self, "class_vars", []):
            try: var.set(True)
            except Exception: pass

    def _toggle_none_classes(self):
        for _nm, var, _idx in getattr(self, "class_vars", []):
            try: var.set(False)
            except Exception: pass

    def _toggle_invert_classes(self):
        for _nm, var, _idx in getattr(self, "class_vars", []):
            try: var.set(not var.get())
            except Exception: pass

    # ========== model loading logic ==========
    def browse_input(self):
        initdir = str(DEFAULT_IN_DIR if DEFAULT_IN_DIR.exists() else PROJECT_ROOT)
        d = filedialog.askdirectory(title="Choose input video folder", initialdir=initdir)
        if d: self.input_dir.set(d)

    def browse_files(self):
        initdir = str(DEFAULT_IN_DIR if DEFAULT_IN_DIR.exists() else PROJECT_ROOT)
        files = filedialog.askopenfilenames(
            title="Select video files",
            initialdir=initdir,
            filetypes=[("Video","*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.mpg *.mpeg *.ts")]
        )
        if files:
            self.selected_files = list(files); self.files_label.config(text=f"Selected {len(self.selected_files)} files")
        else:
            self.selected_files = []; self.files_label.config(text="— none —")

    def clear_files(self):
        self.selected_files = []; self.files_label.config(text="— none —")

    def browse_output(self):
        initdir = str(DEFAULT_OUT_DIR)
        d = filedialog.askdirectory(title="Choose output folder", initialdir=initdir)
        if d:
            self.output_dir.set(d)
            try: ensure_dir(Path(d))
            except Exception: pass

    def browse_weights(self):
        initdir = str(DEFAULT_MODELS_DIR)
        f = filedialog.askopenfilename(initialdir=initdir, title="Choose weights",
                                       filetypes=[("Weights",".pt .zip"), ("All","*.*")])
        if f:
            self.weights_path.set(f); self.load_model_and_classes()

    def _autoload_best_model(self):
        try:
            wp = self.weights_path.get().strip()
            if not wp or Path(wp).is_dir():
                best = find_best_weights(DEFAULT_MODELS_DIR)
                if best: self.weights_path.set(str(best))
            if self.weights_path.get(): self.load_model_and_classes()
        except Exception:
            pass

    def load_model_and_classes(self):
        try:
            out_dir = ensure_dir(Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else DEFAULT_OUT_DIR)
            temp_root = ensure_dir(out_dir / "temp"); extract_dir = ensure_dir(temp_root / "extracted_models")

            wp = Path(self.weights_path.get().strip())
            if wp.is_dir():
                best = find_best_weights(wp)
                if not best: raise FileNotFoundError(f"No .pt/.zip in {wp}")
                wp = best

            pt = resolve_weights_to_pt(wp, extract_dir)
            self._log(f"Loading model: {pt}")
            self.model = YOLO(str(pt)); self.names = self.model.names
            self._populate_classes(self.names)
            self._log("Weights and class list loaded.")
        except Exception as e:
            messagebox.showerror("Model", f"Cannot load weights:\n{e}")

    # ========== classes (dynamic grid) ==========
    def _populate_classes(self, names):
        container = self.classes_scroll.inner
        previously_selected = set(idx for (_nm, var, idx) in getattr(self, "class_vars", []) if var.get())
        for w in container.winfo_children():
            w.destroy()
        self.class_vars.clear()

        id2name = list(names.values()) if isinstance(names, dict) else list(names)

        try:
            avail = max(320, int(self.classes_scroll.canvas.winfo_width()))
        except Exception:
            avail = 800

        cols = int(round(avail / float(self.CLASS_CELL_PX)))
        cols = max(self.CLASS_COL_MIN, min(self.CLASS_COL_MAX, cols))
        self._class_cols = cols

        col_w = max(120, (avail // cols))

        for c in range(cols):
            container.grid_columnconfigure(c, minsize=col_w, weight=1, uniform="classes")

        for i, nm in enumerate(id2name):
            var = tk.BooleanVar(value=(i in previously_selected))
            r, c = divmod(i, cols)

            cell = tk.Frame(container, width=col_w)
            cell.grid(row=r, column=c, sticky="nw", padx=6, pady=3)
            cell.grid_propagate(False)

            cb = tk.Checkbutton(cell, text=nm, variable=var, anchor="w", justify="left",
                                wraplength=col_w - 12)
            cb.pack(fill="x", expand=True, anchor="w")
            self.class_vars.append((nm, var, i))

    def _calc_class_cols(self, width: int | None = None) -> int:
        try:
            if width is None:
                width = max(320, self.classes_scroll.canvas.winfo_width())
        except Exception:
            width = 800
        cols = int(round(width / float(self.CLASS_CELL_PX)))
        return max(self.CLASS_COL_MIN, min(self.CLASS_COL_MAX, cols))

    def _on_classes_canvas_config(self, event):
        try:
            avail = max(320, int(event.width))
        except Exception:
            avail = max(320, int(self.classes_scroll.canvas.winfo_width()))
        new_cols = self._calc_class_cols(avail)

        if new_cols != self._class_cols:
            self._class_cols = new_cols
            if self.names is not None:
                self._populate_classes(self.names)
        else:
            col_w = max(120, (avail // max(1, self._class_cols)))
            for c in range(self._class_cols):
                try:
                    self.classes_scroll.inner.grid_columnconfigure(c, minsize=col_w, weight=1, uniform="classes")
                except Exception:
                    pass

    def selected_class_indices(self):
        return [idx for (nm, v, idx) in self.class_vars if v.get()]

    # ========== ADVANCED OPTIONS ==========
    def open_advanced(self):
        """Advanced options window (delegate to external builder)."""
        win = tk.Toplevel(self)
        win.transient(self)
        win.title(f"Advanced options — {APP_NAME}")
        win.geometry("1000x780")
        win.lift(); win.focus_force()
        # Build the whole panel inside this window (free function call)
        frame = build_advanced_settings(win, self)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

    # ========== START / ABORT ==========
    def abort(self):
        self.abort_event.set()
        self._set_progress(None, "Aborting…")
        def _wait_and_reset():
            try:
                if self.worker_thread is not None:
                    self.worker_done.wait(timeout=3.0)
            finally:
                def _reset_ui():
                    try:
                        self.progressbar.stop()
                        self.progressbar.config(mode="determinate")
                        self._progress_indeterminate = False
                    except Exception:
                        pass
                    self.progress_var.set(0.0)
                    self.progress_label.set("Aborted. Ready.")
                    self.btn_start.config(state="normal")
                    self.btn_abort.config(state="disabled")
                    self._destroy_preview_window()
                self.after(0, _reset_ui)
        threading.Thread(target=_wait_and_reset, daemon=True).start()

    def start(self):
        try:
            if self.btn_start['state'] == "disabled": return
            self.abort_event.clear()
            self.btn_start.config(state="disabled")
            self.btn_abort.config(state="normal")
            self._set_progress(0.0, "Preparing…")

            sources = []
            base_in = None
            if hasattr(self, "src_mode") and self.src_mode.get() == "camera":
                try: cam_idx = int(self.cam_index.get().strip())
                except Exception:
                    messagebox.showerror("Camera", "Camera index must be an integer.")
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                sources = [cam_idx]
            elif hasattr(self, "src_mode") and self.src_mode.get() == "url":
                url = self.url_input.get().strip()
                if not url:
                    messagebox.showerror("URL", "Provide RTSP/HTTP stream URL.")
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                sources = [url]
            else:
                if self.selected_files:
                    sources = [Path(p) for p in self.selected_files]
                    base_in = sources[0].parent if sources and isinstance(sources[0], Path) else None
                else:
                    inp = Path(self.input_dir.get().strip()) if self.input_dir.get().strip() else (DEFAULT_IN_DIR if DEFAULT_IN_DIR.exists() else PROJECT_ROOT)
                    if not inp.exists():
                        messagebox.showerror("Input", "Select a valid folder or files.")
                        self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                    sources = sorted([p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED_VID_EXTS])
                    if not sources:
                        self._log("No video files in the folder.")
                        self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                    base_in = inp

            if self.model is None:
                self.load_model_and_classes()
                if self.model is None:
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return

            selected_idx = self.selected_class_indices()
            if not selected_idx:
                messagebox.showwarning("Classes", "Select at least one class.")
                self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return

            out_base = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else DEFAULT_OUT_DIR
            outp = ensure_dir(out_base)

            self.worker_done.clear()
            self.worker_thread = threading.Thread(target=core_run, args=(self, sources, outp, selected_idx), daemon=True)
            self.worker_thread.start()
        except Exception as e:
            self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")
            messagebox.showerror("Error", str(e))

    # ========== log / progress / ETA ==========
    def _log(self, msg):
        try: self.log.insert("end", msg+"\n"); self.log.see("end")
        except Exception: pass

    def _set_progress(self, percent: float|None, text: str):
        def _upd():
            try:
                if percent is None:
                    if not getattr(self, "_progress_indeterminate", False):
                        self.progressbar.config(mode="indeterminate")
                        self.progressbar.start(10)
                        self._progress_indeterminate = True
                else:
                    if getattr(self, "_progress_indeterminate", False):
                        self.progressbar.stop()
                        self.progressbar.config(mode="determinate")
                        self._progress_indeterminate = False
                    self.progress_var.set(max(0.0, min(100.0, percent)))
            except Exception:
                pass
            self.progress_label.set(text)
        try: self.after(0, _upd)
        except Exception: pass

    def _eta(self, elapsed_s: float, progress_frac: float) -> str:
        if progress_frac <= 1e-6: return "--:--"
        total = elapsed_s / progress_frac
        remain = max(0.0, total - elapsed_s)
        m = int(remain // 60); s = int(remain % 60)
        return f"{m:02d}:{s:02d}"


# ---- robust entrypoint with on-screen + file logging of startup errors ----
def _safe_main():
    import traceback, tkinter as _tk
    from pathlib import Path as _P
    from paths import OUTPUTS as _LOG_DIR
    log_dir = _LOG_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log_file = log_dir / "ui_error.log"

    try:
        app = App()
        app.mainloop()
    except Exception:
        traceback.print_exc()
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "="*60 + "\nFATAL UI ERROR\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        try:
            _r = _tk.Tk(); _r.withdraw()
            from tkinter import messagebox as _mb
            _mb.showerror(f"{APP_NAME} – startup error", traceback.format_exc())
            _r.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    _safe_main()

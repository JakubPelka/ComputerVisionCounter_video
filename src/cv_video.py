# cv_video.py — compact UI + dynamic class grid + presets + interactive help
# Default folders + custom alert sound + robust entrypoint with error popup/log
from __future__ import annotations
import threading, sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cv_video_gui import ScrollableFrame, CounterEditor, AppUIMixin  # GUI utils
from cv_video_run import run as core_run, SoundPlayer               # ⟵ added SoundPlayer import
from cv_video_core import (
    ensure_dir,
    score_weight_name, find_best_weights, resolve_weights_to_pt,
    SUPPORTED_VID_EXTS, MODEL_DIRNAME,
    VIDEO_PRESETS, DEFAULT_QUALITY, DEFAULT_TRACKER,
    LINE_MIN_GAP_FRAMES_DEFAULT, LINE_MIN_SEP_PX_DEFAULT, ZONE_MIN_GAP_FRAMES_DEFAULT,
)

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
from paths import REPO_ROOT, INPUTS as DEFAULT_IN_DIR, OUTPUTS as DEFAULT_OUT_DIR, MODELS as DEFAULT_MODELS_DIR, SOUNDS as DEFAULT_SOUNDS_DIR  # ⟵ nowość

APP_NAME = "ComputerVisionCounter VIDEO"
PROJECT_ROOT = REPO_ROOT

# -------------------- Context help (right panel) --------------------
ADV_HELP_INTRO = (
    "Click or hover any field to see a short explanation and tips.\n\n"
    "Quick tips:\n"
    "• Missing objects → lower conf a bit, raise imgsz, or reduce frame_skip.\n"
    "• Too many false detections → raise conf.\n"
    "• Two boxes on one object → raise iou slightly.\n"
    "• Nearby objects glued into one → lower iou slightly.\n"
    "• IDs drop after occlusion → raise track_buffer or lower match_thresh a little.\n"
    "• ID sticks after object is gone → lower track_buffer or raise match_thresh.\n"
    "• Double counts on a line → raise line_min_gap_frames or line_min_sep_px.\n"
    "• Zone IN/OUT flicker → raise zone_min_gap_frames.\n"
    "• Line at the bottom edge → Anchor=bottom and increase Ghost margin."
)

ADV_HELP_BY_KEY = {
    # Detection / tracking / hysteresis
    "imgsz": ("imgsz",
              "Size of the image sent to the AI.\n"
              "Bigger = sharper detections but slower.\n"
              "Tip: 320–640 for CPU; 960–1280 only for small/far objects."),
    "conf": ("conf",
             "Confidence filter. Higher = stricter (fewer false alarms, more misses).\n"
             "Start 0.50–0.60. Misses? lower a bit. Too many fakes? raise."),
    "iou": ("iou",
            "Box overlap used to remove duplicates. Higher merges more; lower keeps more boxes.\n"
            "Two boxes on one object → raise a little. Close objects merge → lower a little."),
    "frame_skip": ("frame_skip",
                   "How many frames to skip between analyses. 0 = every frame (best quality, slowest).\n"
                   "+1 or +2 speeds up but can miss fast motion."),
    "track_buffer": ("track_buffer",
                     "How long (in frames) a lost object ID is kept alive.\n"
                     "Higher survives short occlusions; too high may leave ghost IDs."),
    "match_thresh": ("match_thresh",
                     "How strict matching is when assigning boxes to existing IDs.\n"
                     "Higher = stricter (fewer wrong matches, more ID resets)."),
    "min_hits": ("min_hits",
                 "Frames needed before a new track is confirmed.\n"
                 "Short flickers? raise. Delayed ID appearance? lower."),
    "line_min_gap_frames": ("line_min_gap_frames",
                            "Minimum frames between two line counts for the same object (anti-bounce)."),
    "line_min_sep_px": ("line_min_sep_px",
                        "Minimum movement across the line (pixels) to count again.\n"
                        "Sliding along the line double-counts? raise this."),
    "zone_min_gap_frames": ("zone_min_gap_frames",
                            "Minimum frames between zone IN/OUT events for the same object.\n"
                            "IN/OUT toggling on the border? raise this."),

    # Overlay / Trace / Anchor / Ghost
    "preview_enabled": ("Enable LIVE preview",
                        "Shows processed video while running. Turn off to save CPU."),
    "trace_enabled": ("Trace",
                      "Draw a tail behind each object."),
    "trace_len": ("Trace len",
                  "How many recent points to keep for the tail."),
    "anchor_mode": ("Anchor",
                    "Point used for counting & trails.\n"
                    "bottom → feet/wheels on the ground (best for floor lines)\n"
                    "center → center of box."),
    "ghost_margin": ("Ghost margin (px)",
                     "Only for 'bottom' anchor: lift the point above the box edge.\n"
                     "Useful if the bottom edge touches the frame border or bounces on a line."),

    # Colors / thickness
    "trace_color": ("Trace color",
                    "Use 'auto', a #RRGGBB color (e.g. #00FF88), or B,G,R (e.g. 255,200,0)."),
    "trace_thickness": ("Trace thickness",
                        "Line width in pixels (0–16). 0 hides traces."),
    "overlay_frame_color": ("Frame color",
                            "Color for lines & zone outlines. Same format as Trace color."),
    "overlay_frame_thickness": ("Frame thickness",
                                "Line width in pixels (0–16). 0 hides frame lines."),

    # Sound alert (uses selected classes from the main window)
    "alert_enabled": ("Enable alert",
                      "Turn zone/line sound on or off. Uses currently selected classes."),
    "alert_sound": ("Alert sound file",
                    "Pick a WAV/MP3 file to play for alerts. Works on macOS, Linux, and Windows.\n"
                    "Tip: WAV is the most reliable everywhere."),
    "alert_loop": ("Loop while active",
                   "If ON: keeps playing sound while a target matches the condition.\n"
                   "If OFF: plays single pings, rate-limited by 'freeze (s)'."),
    "alert_freeze_s": ("freeze (s)",
                       "Minimum time in seconds between two sound starts (for pings or loop restarts)."),
    "alert_zone_inside": ("Mode",
                          "Choose when to play: inside the zone or outside the zone."),
}


class App(AppUIMixin, tk.Tk):
    CLASS_COL_MIN = 2
    CLASS_COL_MAX = 12
    CLASS_CELL_PX = 160

    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} — line/zone counting (YOLO + ByteTrack)")
        self.geometry("980x720")

        # --- GUI variables (with better defaults) ---
        self.input_dir = tk.StringVar(value=str(DEFAULT_IN_DIR) if DEFAULT_IN_DIR.exists() else "")
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
            # NEW sound settings:
            "alert_freeze_s": 2,   # seconds between sound starts
            "alert_zone_inside": 1,  # 1 = inside zone, 0 = outside zone
            "alert_sound": "",       # path to wav/mp3
            "alert_loop": True,      # loop while active
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
            # stop any previous test loops (safe no-op if none)
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
        tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality,
                 command=lambda *_: self._update_preset_label()).pack(side="left", fill="x", expand=True, padx=8)
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

        # Toggle row (same layout as CvC_images)
        tog = tk.Frame(lf); tog.pack(fill="x", pady=(4, 0))
        tk.Label(tog, text="Toggle:").pack(side="left")
        tk.Button(tog, text="All",   width=6, command=self._toggle_all_classes).pack(side="left", padx=(6, 0))
        tk.Button(tog, text="None",  width=6, command=self._toggle_none_classes).pack(side="left", padx=(6, 0))
        tk.Button(tog, text="Invert",width=6, command=self._toggle_invert_classes).pack(side="left", padx=(6, 0))

        # Scroll area for checkboxes (unchanged)
        self.classes_scroll = ScrollableFrame(lf, height=220)
        self.classes_scroll.pack(fill="both", expand=True)
        # react to real canvas resize (keeps the nice grid)
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

    def _update_preset_label(self):
        p = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
        self.preset_label.config(text=(f"imgsz={p['imgsz']}  conf={p['conf']}  iou={p['iou']}  "
                                       f"skip={p['frame_skip']}  buf={p['track_buffer']}  "
                                       f"match={p['match_thresh']}  hits={p['min_hits']}"))


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

    # ========== ADVANCED OPTIONS (with dynamic help) ==========
    def open_advanced(self):
        """Advanced options window — with a right-side context help panel."""
        from tkinter import colorchooser

        win = tk.Toplevel(self)
        win.transient(self)
        win.grab_set()
        win.lift(); win.focus_force()
        win.title(f"Advanced options — {APP_NAME}")
        win.geometry("1000x780")

        # two-pane layout
        root = tk.Frame(win); root.pack(fill="both", expand=True)
        left = tk.Frame(root); left.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        right = tk.Frame(root, width=360); right.pack(side="left", fill="y", padx=(8, 8), pady=8)

        # right-side help widgets
        help_title = tk.Label(right, text="Help", font=("TkDefaultFont", 10, "bold"), anchor="w")
        help_title.pack(fill="x", pady=(0,4))
        hl = tk.Frame(right); hl.pack(fill="both", expand=True)
        sb = tk.Scrollbar(hl, orient="vertical")
        help_text = tk.Text(hl, wrap="word", yscrollcommand=sb.set)
        sb.config(command=help_text.yview)
        sb.pack(side="right", fill="y")
        help_text.pack(side="left", fill="both", expand=True)
        def set_help(title: str, body: str):
            try:
                help_title.config(text=title or "Help")
                help_text.config(state="normal")
                help_text.delete("1.0", "end")
                help_text.insert("1.0", body or "")
                help_text.config(state="disabled")
            except Exception:
                pass
        set_help("Tips", ADV_HELP_INTRO)

        def attach_help(widget, key: str):
            title, body = ADV_HELP_BY_KEY.get(key, (key, ""))
            def _show(_e=None, t=title, b=body):
                set_help(t, b)
            try:
                widget.bind("<FocusIn>", _show)
                widget.bind("<Enter>", _show)
            except Exception:
                pass

        # Base from presets or current override
        p = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
        base = self.adv_params if getattr(self, "advanced_override", False) else {
            **p,
            "line_min_gap": LINE_MIN_GAP_FRAMES_DEFAULT,
            "line_min_sep": LINE_MIN_SEP_PX_DEFAULT,
            "zone_min_gap": ZONE_MIN_GAP_FRAMES_DEFAULT,
            "preview_enabled": bool(self.preview_enabled.get()),
            "trace_enabled": bool(self.trace_enabled.get()),
            "trace_len": int(self.trace_len.get()),
            "anchor_mode": self.anchor_mode.get(),
            "ghost_margin": int(self.ghost_margin.get()),
            "alert_enabled": bool(self.alert_enabled.get()),
            "alert_freeze_s": int(self.alert_freeze_s.get()),
            "alert_zone_inside": 1,
            "trace_color": str(self.adv_params.get("trace_color", "auto")) if getattr(self, "advanced_override", False) else "auto",
            "trace_thickness": int(self.adv_params.get("trace_thickness", 2)) if getattr(self, "advanced_override", False) else 2,
            "overlay_frame_color": str(self.adv_params.get("overlay_frame_color", "auto")) if getattr(self, "advanced_override", False) else "auto",
            "overlay_frame_thickness": int(self.adv_params.get("overlay_frame_thickness", 2)) if getattr(self, "advanced_override", False) else 2,
            # NEW:
            "alert_sound": self.alert_sound.get(),
            "alert_loop": bool(self.alert_loop.get()),
        }

        def row(parent, label, var, w=18, key: str|None=None):
            f = tk.Frame(parent); f.pack(fill="x", pady=3)
            tk.Label(f, text=label, width=26, anchor="w").pack(side="left")
            ent = tk.Entry(f, textvariable=var, width=w)
            ent.pack(side="left")
            if key: attach_help(ent, key)
            return f, ent

        # --- Detection/Tracking/Hysteresis ---
        frm_basic = tk.LabelFrame(left, text="Detection / Tracking / Hysteresis"); frm_basic.pack(fill="x", padx=0, pady=6)
        v_imgsz = tk.StringVar(value=str(base.get("imgsz", "")));     row(frm_basic, "imgsz", v_imgsz, key="imgsz")
        v_conf  = tk.StringVar(value=str(base.get("conf", "")));      row(frm_basic, "conf", v_conf, key="conf")
        v_iou   = tk.StringVar(value=str(base.get("iou", "")));       row(frm_basic, "iou", v_iou, key="iou")
        v_skip  = tk.StringVar(value=str(base.get("frame_skip", "")));row(frm_basic, "frame_skip", v_skip, key="frame_skip")
        v_buf   = tk.StringVar(value=str(base.get("track_buffer", ""))); row(frm_basic, "track_buffer", v_buf, key="track_buffer")
        v_match = tk.StringVar(value=str(base.get("match_thresh", ""))); row(frm_basic, "match_thresh", v_match, key="match_thresh")
        v_hits  = tk.StringVar(value=str(base.get("min_hits", "")));  row(frm_basic, "min_hits", v_hits, key="min_hits")
        v_lgap  = tk.StringVar(value=str(base.get("line_min_gap", ""))); row(frm_basic, "line_min_gap_frames", v_lgap, key="line_min_gap_frames")
        v_lsep  = tk.StringVar(value=str(base.get("line_min_sep", ""))); row(frm_basic, "line_min_sep_px", v_lsep, key="line_min_sep_px")
        v_zhgap = tk.StringVar(value=str(base.get("zone_min_gap", ""))); row(frm_basic, "zone_min_gap_frames", v_zhgap, key="zone_min_gap_frames")

        # --- Overlay / Trace / Anchor / Ghost ---
        frm_vis = tk.LabelFrame(left, text="Overlay / Trace / Anchor / Ghost"); frm_vis.pack(fill="x", padx=0, pady=6)
        v_prev  = tk.BooleanVar(value=bool(base.get("preview_enabled", True)))
        v_trace = tk.BooleanVar(value=bool(base.get("trace_enabled", True)))
        v_tlen  = tk.IntVar(value=int(base.get("trace_len", 24)))
        v_anch  = tk.StringVar(value=str(base.get("anchor_mode", "center")))
        v_ghost = tk.IntVar(value=int(base.get("ghost_margin", 12)))
        cb_prev = tk.Checkbutton(frm_vis, text="Enable LIVE preview", variable=v_prev); cb_prev.pack(side="left", padx=6, pady=4); attach_help(cb_prev, "preview_enabled")
        cb_trace= tk.Checkbutton(frm_vis, text="Trace", variable=v_trace); cb_trace.pack(side="left", padx=(12,4)); attach_help(cb_trace, "trace_enabled")
        tk.Label(frm_vis, text="len:").pack(side="left")
        sp_len = tk.Spinbox(frm_vis, from_=0, to=300, width=5, textvariable=v_tlen); sp_len.pack(side="left", padx=(2, 12)); attach_help(sp_len, "trace_len")
        tk.Label(frm_vis, text="Anchor:").pack(side="left")
        cb_anchor = ttk.Combobox(frm_vis, values=["bottom","center"], width=8, state="readonly", textvariable=v_anch)
        cb_anchor.pack(side="left", padx=(3, 12)); attach_help(cb_anchor, "anchor_mode")
        tk.Label(frm_vis, text="Ghost margin (px):").pack(side="left")
        sp_ghost = tk.Spinbox(frm_vis, from_=0, to=64, width=5, textvariable=v_ghost); sp_ghost.pack(side="left", padx=(3, 6)); attach_help(sp_ghost, "ghost_margin")

        # --- Colors/Thickness + pickers ---
        frm_colors = tk.LabelFrame(left, text="Colors/Thickness (overlay)"); frm_colors.pack(fill="x", padx=0, pady=6)
        v_tcolor = tk.StringVar(value=str(base.get("trace_color", "auto")))
        v_tth    = tk.IntVar(value=int(base.get("trace_thickness", 2)))
        v_fcolor = tk.StringVar(value=str(base.get("overlay_frame_color", "auto")))
        v_fth    = tk.IntVar(value=int(base.get("overlay_frame_thickness", 2)))

        def _pick_to_var(var):
            try:
                init = var.get().strip()
                rgb, hexv = colorchooser.askcolor(initialcolor=init if init.startswith("#") else None, title="Pick a color", parent=win)
                if hexv:
                    var.set(hexv.upper())
                win.lift(); win.focus_force()
            except Exception:
                pass

        r1 = tk.Frame(frm_colors); r1.pack(fill="x", pady=3)
        tk.Label(r1, text="Trace color (auto/#RRGGBB/B,G,R)", width=26, anchor="w").pack(side="left")
        ent_tcolor = tk.Entry(r1, textvariable=v_tcolor, width=18); ent_tcolor.pack(side="left"); attach_help(ent_tcolor, "trace_color")
        tk.Button(r1, text="Pick…", command=lambda: _pick_to_var(v_tcolor)).pack(side="left", padx=6)
        f = tk.Frame(frm_colors); f.pack(fill="x", pady=2)
        tk.Label(f, text="Trace thickness (px)", width=26, anchor="w").pack(side="left")
        sp_tth = tk.Spinbox(f, from_=0, to=16, width=6, textvariable=v_tth); sp_tth.pack(side="left"); attach_help(sp_tth, "trace_thickness")

        r2 = tk.Frame(frm_colors); r2.pack(fill="x", pady=3)
        tk.Label(r2, text="Frame color (auto/#RRGGBB/B,G,R)", width=26, anchor="w").pack(side="left")
        ent_fcolor = tk.Entry(r2, textvariable=v_fcolor, width=18); ent_fcolor.pack(side="left"); attach_help(ent_fcolor, "overlay_frame_color")
        tk.Button(r2, text="Pick…", command=lambda: _pick_to_var(v_fcolor)).pack(side="left", padx=6)
        f2 = tk.Frame(frm_colors); f2.pack(fill="x", pady=2)
        tk.Label(f2, text="Frame thickness (px)", width=26, anchor="w").pack(side="left")
        sp_fth = tk.Spinbox(f2, from_=0, to=16, width=6, textvariable=v_fth); sp_fth.pack(side="left"); attach_help(sp_fth, "overlay_frame_thickness")

        # --- Sound alert (zones/lines) ---
        frame_alert = tk.LabelFrame(left, text="Sound alert"); frame_alert.pack(fill="x", padx=0, pady=6)

        v_a_en    = tk.BooleanVar(value=bool(base.get("alert_enabled", False)))
        v_a_freeS = tk.IntVar(value=int(base.get("alert_freeze_s", 2)))
        v_a_where = tk.IntVar(value=int(base.get("alert_zone_inside", 1)))  # 1=inside, 0=outside
        v_a_sound = tk.StringVar(value=str(base.get("alert_sound", "")))
        v_a_loop  = tk.BooleanVar(value=bool(base.get("alert_loop", True)))

        r1 = tk.Frame(frame_alert); r1.pack(fill="x", padx=6, pady=2)
        cb_a_en = tk.Checkbutton(r1, text="Enable alert (uses selected classes)", variable=v_a_en); cb_a_en.pack(side="left", padx=(0,8)); attach_help(cb_a_en, "alert_enabled")

        r2 = tk.Frame(frame_alert); r2.pack(fill="x", padx=6, pady=2)
        tk.Label(r2, text="Sound file:", width=12, anchor="w").pack(side="left")
        ent_a_path = tk.Entry(r2, textvariable=v_a_sound, width=36); ent_a_path.pack(side="left", padx=(3, 6)); attach_help(ent_a_path, "alert_sound")
        def _pick_sound():
            initdir = str(DEFAULT_SOUNDS_DIR if DEFAULT_SOUNDS_DIR.exists() else PROJECT_ROOT)
            fpath = filedialog.askopenfilename(initialdir=initdir, title="Choose alert sound (wav/mp3)",
                                               filetypes=[("Audio","*.wav *.mp3 *.ogg *.flac *.m4a"), ("All","*.*")])
            if fpath: v_a_sound.set(fpath)
        tk.Button(r2, text="Browse…", command=_pick_sound).pack(side="left")
        # ▶ Test sound button — plays the currently selected file once
        tk.Button(r2, text="Test sound ▶", command=lambda: self._play_test_sound(v_a_sound.get())).pack(side="left", padx=(8,0))

        r3 = tk.Frame(frame_alert); r3.pack(fill="x", padx=6, pady=2)
        cb_loop = tk.Checkbutton(r3, text="Loop while active", variable=v_a_loop); cb_loop.pack(side="left", padx=(0,12)); attach_help(cb_loop, "alert_loop")
        tk.Label(r3, text="freeze (s):").pack(side="left")
        sp_a_free = tk.Spinbox(r3, from_=0, to=60, width=5, textvariable=v_a_freeS); sp_a_free.pack(side="left", padx=(3, 6)); attach_help(sp_a_free, "alert_freeze_s")

        r4 = tk.Frame(frame_alert); r4.pack(fill="x", padx=6, pady=(6,4))
        tk.Label(r4, text="Mode:", width=8, anchor="w").pack(side="left")
        rb_in  = tk.Radiobutton(r4, text="inside zone",  variable=v_a_where, value=1); rb_in.pack(side="left", padx=(3,12)); attach_help(rb_in, "alert_zone_inside")
        rb_out = tk.Radiobutton(r4, text="outside zone", variable=v_a_where, value=0); rb_out.pack(side="left", padx=(3,2)); attach_help(rb_out, "alert_zone_inside")

        # ---- read/write fields ----
        def _collect() -> dict:
            cur = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
            def _get(var, cast, key, dflt):
                s = var.get().strip()
                if s == "": return cur.get(key, dflt)
                try: return cast(s)
                except Exception: return dflt
            return {
                "imgsz": _get(v_imgsz, int,   "imgsz",        640),
                "conf":  _get(v_conf,  float, "conf",         0.5),
                "iou":   _get(v_iou,   float, "iou",          0.6),
                "frame_skip":   _get(v_skip,  int,   "frame_skip",   1),
                "track_buffer": _get(v_buf,   int,   "track_buffer", 60),
                "match_thresh": _get(v_match, float, "match_thresh", 0.78),
                "min_hits":     _get(v_hits,  int,   "min_hits",     2),
                "line_min_gap": int(v_lgap.get().strip() or LINE_MIN_GAP_FRAMES_DEFAULT),
                "line_min_sep": int(v_lsep.get().strip() or LINE_MIN_SEP_PX_DEFAULT),
                "zone_min_gap": int(v_zhgap.get().strip() or ZONE_MIN_GAP_FRAMES_DEFAULT),
                "preview_enabled": bool(v_prev.get()),
                "trace_enabled": bool(v_trace.get()),
                "trace_len": int(v_tlen.get()),
                "anchor_mode": v_anch.get(),
                "ghost_margin": int(v_ghost.get()),
                "trace_color": v_tcolor.get().strip(),
                "trace_thickness": int(v_tth.get()),
                "overlay_frame_color": v_fcolor.get().strip(),
                "overlay_frame_thickness": int(v_fth.get()),
                # Sound (uses selected classes)
                "alert_enabled": bool(v_a_en.get()),
                "alert_sound": v_a_sound.get().strip(),
                "alert_loop": bool(v_a_loop.get()),
                "alert_freeze_s": int(v_a_freeS.get()),
                "alert_zone_inside": int(v_a_where.get()),
            }

        def _apply():
            try:
                params = _collect()
                self.adv_params = params
                self.advanced_override = True
                # sync with fields used in run()
                self.preview_enabled.set(params["preview_enabled"])
                self.trace_enabled.set(params["trace_enabled"])
                self.trace_len.set(params["trace_len"])
                self.anchor_mode.set(params["anchor_mode"])
                self.ghost_margin.set(params["ghost_margin"])
                self.alert_enabled.set(params["alert_enabled"])
                self.alert_sound.set(params["alert_sound"])
                self.alert_loop.set(params["alert_loop"])
                self.alert_freeze_s.set(params["alert_freeze_s"])
                self._log("[ADV] Applied override (from fields).")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Adv", str(e))

        def _reset():
            self.advanced_override = False
            self._log("[ADV] Restored preset from quality slider.")
            win.destroy()

        btns = tk.Frame(left); btns.pack(fill="x", pady=10)
        tk.Button(btns, text="Apply", command=_apply).pack(side="left", padx=6)
        tk.Button(btns, text="Restore preset from slider", command=_reset).pack(side="left", padx=6)

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


    # ========== PREVIEW window ==========
    def _ensure_preview_window(self):
        """Create/activate the LIVE preview window.
        Closing the window (X) or pressing ESC == pressing the ABORT button.
        The window is raised and focused so ESC works immediately.
        """
        def _abort_from_preview(_evt=None):
            try:
                self.abort()
            except Exception:
                try:
                    self._destroy_preview_window()
                except Exception:
                    pass

        # If it already exists, just raise & focus it
        if getattr(self, "_preview_win", None) and self._preview_win.winfo_exists():
            try:
                w = self._preview_win
                w.deiconify()
                w.lift()
                w.focus_force()
                if getattr(self, "_preview_lbl", None):
                    self._preview_lbl.focus_set()
                # bring-to-front bump without staying always-on-top
                w.attributes("-topmost", True)
                w.after(250, lambda: w.attributes("-topmost", False))
            except Exception:
                pass
            return

        # Create new window
        win = tk.Toplevel(self)
        win.title("Preview (LIVE)")
        win.geometry("960x620")

        # Close (X) or Esc -> ABORT
        win.protocol("WM_DELETE_WINDOW", _abort_from_preview)
        win.bind("<Escape>", _abort_from_preview)
        win.bind("<Control-w>", _abort_from_preview)

        lbl = tk.Label(win, anchor="center", bg="#111")
        lbl.pack(fill="both", expand=True)

        self._preview_win = win
        self._preview_lbl = lbl
        self._preview_last_bgr = None  # last frame cache for resize redraws

        # Redraw on window resize so scaling follows the window
        win.bind("<Configure>", self._on_preview_resize)

        # Raise & focus so Esc works right away
        try:
            win.update_idletasks()
            win.deiconify()
            win.lift()
            win.focus_force()
            lbl.focus_set()
            win.attributes("-topmost", True)
            win.after(250, lambda: win.attributes("-topmost", False))
        except Exception:
            pass

    def _on_preview_resize(self, _evt=None):
        """Redraw last frame when the preview window changes size."""
        try:
            if getattr(self, "_preview_last_bgr", None) is not None:
                # Reuse the last frame; _show_preview_bgr will rescale to new size
                self._show_preview_bgr(self._preview_last_bgr)
        except Exception:
            pass



    def _show_preview_bgr(self, frame_bgr):
        if not self.preview_enabled.get():
            return

        def _do():
            try:
                self._ensure_preview_window()
                win = getattr(self, "_preview_win", None)
                lbl = getattr(self, "_preview_lbl", None)
                if not (win and lbl and win.winfo_exists()):
                    return

                # Cache last frame so we can redraw on window resize
                self._preview_last_bgr = frame_bgr

                # Target area: ~90% of current window client size (keep aspect ratio)
                win.update_idletasks()
                tw = max(100, win.winfo_width()  - 12)
                th = max(100, win.winfo_height() - 12)
                target_w = int(tw * 0.90)
                target_h = int(th * 0.90)

                H, W = frame_bgr.shape[:2]
                # Allow UPSCALING (important for 4K videos in a large window)
                scale = min(target_w / float(W), target_h / float(H))
                nw = max(1, int(W * scale))
                nh = max(1, int(H * scale))

                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                disp = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
                rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

                from PIL import Image, ImageTk
                imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._preview_imgtk = imgtk  # keep reference
                lbl.config(image=imgtk)
            except Exception:
                pass

        try:
            self.after(0, _do)
        except Exception:
            pass


    def _destroy_preview_window(self):
        try:
            if self._preview_win and self._preview_win.winfo_exists():
                self._preview_win.destroy()
        except Exception:
            pass
        self._preview_win = None
        self._preview_lbl = None
        self._preview_imgtk = None


# ---- robust entrypoint with on-screen + file logging of startup errors ----
def _safe_main():
    import traceback, tkinter as _tk
    from pathlib import Path as _P
    from paths import OUTPUTS as _LOG_DIR   # dodaj u góry funkcji lub modułu
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
        # 1) dump full traceback to console
        traceback.print_exc()
        # 2) save to output/ui_error.log
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "="*60 + "\nFATAL UI ERROR\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        # 3) also show a Tk messagebox so you see it if console closes
        try:
            _r = _tk.Tk(); _r.withdraw()
            from tkinter import messagebox as _mb
            _mb.showerror(f"{APP_NAME} – startup error", traceback.format_exc())
            _r.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    _safe_main()

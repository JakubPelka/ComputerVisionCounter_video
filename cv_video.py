# cv_video.py — compact UI (EN) + dynamic class grid (up to 12 cols) + presets (preview/trace/anchor/ghost/alert)
from __future__ import annotations
import threading, sys, json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cv_video_gui import ScrollableFrame, CounterEditor, AppUIMixin
from cv_video_run import run as core_run
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


class App(AppUIMixin, tk.Tk):
    # --- class grid configuration ---
    CLASS_COL_MIN = 2
    CLASS_COL_MAX = 12     # max columns for class grid
    CLASS_CELL_PX = 160    # approx width of one column (checkbox + label)

    def __init__(self):
        super().__init__()
        self.title("Unidrone VIDEO – line/zone counting (YOLO + ByteTrack)")
        self.geometry("980x720")

        # --- GUI state ---
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        models_default = Path(__file__).parent / MODEL_DIRNAME
        self.weights_path = tk.StringVar(value=str(find_best_weights(models_default) or models_default))

        self.quality = tk.IntVar(value=DEFAULT_QUALITY)
        self.tracker_kind = tk.StringVar(value=DEFAULT_TRACKER)
        self.overlay_mode = tk.StringVar(value="centroid")

        self.model = None; self.names = None; self.class_vars = []
        self.selected_files = []
        self._class_cols = 5

        # --- advanced / preset-able params ---
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
            # extra (stored in presets)
            "preview_enabled": True,
            "trace_enabled": True,
            "trace_len": 24,
            "anchor_mode": "center",    # "bottom"|"center"
            "ghost_margin": 12,
            "alert_enabled": False,
            "alert_classes": "cat,person",
            "alert_freq": 880,
            "alert_dur": 180,
            "alert_freeze": 1500,
        }

        # bindings (used by run)
        self.preview_enabled = tk.BooleanVar(value=self.adv_params["preview_enabled"])
        self.trace_enabled   = tk.BooleanVar(value=self.adv_params["trace_enabled"])
        self.trace_len       = tk.IntVar(value=self.adv_params["trace_len"])
        self.anchor_mode     = tk.StringVar(value=self.adv_params["anchor_mode"])
        self.ghost_margin    = tk.IntVar(value=self.adv_params["ghost_margin"])
        self.alert_enabled   = tk.BooleanVar(value=self.adv_params["alert_enabled"])
        self.alert_classes   = tk.StringVar(value=self.adv_params["alert_classes"])
        self.alert_freq      = tk.IntVar(value=self.adv_params["alert_freq"])
        self.alert_dur       = tk.IntVar(value=self.adv_params["alert_dur"])
        self.alert_freeze    = tk.IntVar(value=self.adv_params["alert_freeze"])

        # --- progress/worker + preview window ---
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = tk.StringVar(value="Ready.")
        self.abort_event = threading.Event()
        self.worker_done = threading.Event()
        self.worker_thread = None
        self._progress_indeterminate = False

        self._preview_win = None
        self._preview_lbl = None
        self._preview_imgtk = None

        # --- build UI ---
        self.build_ui_compact()
        self._autoload_best_model()

    # ========== UI (compact) ==========
    def build_ui_compact(self):
        root = tk.Frame(self); root.pack(fill="both", expand=True, padx=8, pady=6)

        # Paths
        self._row_browse(root, "Input video folder:", self.input_dir, self.browse_input, is_dir=True)
        f_files = tk.Frame(root); f_files.pack(fill="x", pady=2)
        tk.Button(f_files, text="Select video files…", command=self.browse_files).pack(side="left")
        self.files_label = tk.Label(f_files, text="— none —"); self.files_label.pack(side="left", padx=8)
        tk.Button(f_files, text="Clear selection", command=self.clear_files).pack(side="left", padx=(8,0))

        self._row_browse(root, "Output folder (optional):", self.output_dir, self.browse_output, is_dir=True)
        self._row_browse(root, "Weights (.pt/.zip):", self.weights_path, self.browse_weights, is_dir=False)

        # Source
        srcf = tk.LabelFrame(root, text="Input source"); srcf.pack(fill="x", pady=4)
        self.src_mode = tk.StringVar(value="files"); self.cam_index = tk.StringVar(value="0"); self.url_input = tk.StringVar(value="")
        def _src_toggle(*_):
            mf = self.src_mode.get()
            cam_ent.config(state=("normal" if mf=="camera" else "disabled"))
            url_ent.config(state=("normal" if mf=="url" else "disabled"))
        tk.Radiobutton(srcf, text="Files",   variable=self.src_mode, value="files",  command=_src_toggle).pack(side="left", padx=6)
        tk.Radiobutton(srcf, text="Camera",  variable=self.src_mode, value="camera", command=_src_toggle).pack(side="left", padx=6)
        tk.Label(srcf, text="Index:").pack(side="left")
        cam_ent = tk.Entry(srcf, width=4, textvariable=self.cam_index); cam_ent.pack(side="left", padx=(0,8))
        tk.Radiobutton(srcf, text="RTSP/HTTP URL", variable=self.src_mode, value="url", command=_src_toggle).pack(side="left", padx=6)
        url_ent = tk.Entry(srcf, textvariable=self.url_input); url_ent.pack(side="left", fill="x", expand=True, padx=(0,6))
        _src_toggle()

        # Quality
        qf = tk.Frame(root); qf.pack(fill="x", pady=4)
        tk.Label(qf, text="Quality (=1 faster/weaker, 5 = ULTRA)").pack(side="left")
        tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality,
                 command=lambda *_: self._update_preset_label()).pack(side="left", fill="x", expand=True, padx=8)
        self.preset_label = tk.Label(qf, text=""); self.preset_label.pack(side="left")
        self._update_preset_label()

        # Overlay + Tracker
        ot = tk.Frame(root); ot.pack(fill="x", pady=4)
        ov = tk.LabelFrame(ot, text="Overlay"); ov.pack(side="left", fill="x", expand=True, padx=(0,6))
        tk.Radiobutton(ov, text="Centroids",      variable=self.overlay_mode, value="centroid").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boxes",          variable=self.overlay_mode, value="boxes").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boxes + conf",   variable=self.overlay_mode, value="boxes_conf").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Polygons",       variable=self.overlay_mode, value="polygon").pack(side="left", padx=6)

        tr = tk.LabelFrame(ot, text="Tracker"); tr.pack(side="left", padx=(6,0))
        tk.Radiobutton(tr, text="ByteTrack", variable=self.tracker_kind, value="bytetrack").pack(side="left", padx=6)
        tk.Radiobutton(tr, text="BoT-SORT",  variable=self.tracker_kind, value="botsort").pack(side="left", padx=6)

        # Classes — scrollable + dynamic columns
        lf = tk.LabelFrame(root, text="Class selection (after loading weights)"); lf.pack(fill="both", expand=True, pady=4)
        self.classes_scroll = ScrollableFrame(lf, height=220)
        self.classes_scroll.pack(fill="both", expand=True)
        self.classes_scroll.canvas.bind("<Configure>", self._on_classes_canvas_config)

        # Controls + Progress
        bf = tk.Frame(root); bf.pack(fill="x", pady=6)
        self.btn_start = tk.Button(bf, text="START", command=self.start); self.btn_start.pack(side="left")
        tk.Button(bf, text="Advanced options…", command=self.open_advanced).pack(side="left", padx=8)
        self.btn_abort = tk.Button(bf, text="ABORT", command=self.abort, state="disabled"); self.btn_abort.pack(side="left", padx=(8,0))

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

    # ========== model loading ==========
    def browse_input(self):
        d = filedialog.askdirectory(title="Select input folder with videos")
        if d: self.input_dir.set(d)

    def browse_files(self):
        files = filedialog.askopenfilenames(title="Select video files",
                                            filetypes=[("Video","*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.mpg *.mpeg *.ts")])
        if files:
            self.selected_files = list(files); self.files_label.config(text=f"{len(self.selected_files)} file(s) selected")
        else:
            self.selected_files = []; self.files_label.config(text="— none —")

    def clear_files(self):
        self.selected_files = []; self.files_label.config(text="— none —")

    def browse_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d: self.output_dir.set(d)

    def browse_weights(self):
        initdir = str(Path(self.weights_path.get()).parent) if self.weights_path.get() else str(Path(__file__).parent / MODEL_DIRNAME)
        f = filedialog.askopenfilename(initialdir=initdir, title="Select weights",
                                       filetypes=[("Weights",".pt .zip"), ("All","*.*")])
        if f:
            self.weights_path.set(f); self.load_model_and_classes()

    def _autoload_best_model(self):
        try:
            wp = self.weights_path.get().strip()
            if not wp or Path(wp).is_dir():
                best = find_best_weights(Path(wp) if wp else (Path(__file__).parent / MODEL_DIRNAME))
                if best: self.weights_path.set(str(best))
            if self.weights_path.get(): self.load_model_and_classes()
        except Exception:
            pass

    def load_model_and_classes(self):
        try:
            out_dir = ensure_dir((Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else Path.cwd()) / "results")
            temp_root = ensure_dir(out_dir / "temp"); extract_dir = ensure_dir(temp_root / "extracted_models")

            wp = Path(self.weights_path.get().strip())
            if wp.is_dir():
                best = find_best_weights(wp)
                if not best: raise FileNotFoundError(f"No .pt/.zip found in {wp}")
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
        """Draw class checkbox grid; preserve selections on relayout."""
        container = self.classes_scroll.inner
        previously_selected = set(idx for (_nm, var, idx) in getattr(self, "class_vars", []) if var.get())

        for w in container.winfo_children():
            w.destroy()
        self.class_vars.clear()

        id2name = list(names.values()) if isinstance(names, dict) else list(names)

        cols = self._class_cols or self._calc_class_cols()
        cols = max(self.CLASS_COL_MIN, min(self.CLASS_COL_MAX, cols))

        for i, nm in enumerate(id2name):
            var = tk.BooleanVar(value=(i in previously_selected))
            cb = tk.Checkbutton(container, text=nm, variable=var)
            r, c = divmod(i, cols)
            cb.grid(row=r, column=c, sticky="w", padx=6, pady=3)
            self.class_vars.append((nm, var, i))

    def _calc_class_cols(self, width: int | None = None) -> int:
        """Number of columns based on canvas width."""
        try:
            if width is None:
                width = max(320, self.classes_scroll.canvas.winfo_width())
        except Exception:
            width = 800
        cols = max(self.CLASS_COL_MIN,
                   min(self.CLASS_COL_MAX, width // self.CLASS_CELL_PX))
        return cols

    def _on_classes_canvas_config(self, event):
        """Relayout when the class canvas width changes."""
        try:
            new_cols = self._calc_class_cols(event.width)
        except Exception:
            new_cols = self._calc_class_cols()
        if new_cols != self._class_cols:
            self._class_cols = new_cols
            if self.names is not None:
                self._populate_classes(self.names)

    def selected_class_indices(self):
        return [idx for (nm, v, idx) in self.class_vars if v.get()]

    # ========== Advanced options (with presets) ==========
    def open_advanced(self):
        win = tk.Toplevel(self); win.title("Advanced options"); win.geometry("680x720")

        # base: slider preset + defaults (if override off), otherwise current adv_params
        p = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
        base = self.adv_params if self.advanced_override else {
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
            "alert_classes": self.alert_classes.get(),
            "alert_freq": int(self.alert_freq.get()),
            "alert_dur": int(self.alert_dur.get()),
            "alert_freeze": int(self.alert_freeze.get()),
        }

        def add_row(parent, lbl, var, w=18):
            f = tk.Frame(parent); f.pack(fill="x", pady=4)
            tk.Label(f, text=lbl, width=26, anchor="w").pack(side="left")
            e = tk.Entry(f, textvariable=var, width=w); e.pack(side="left")
            return e

        # Detection / Tracking / Hysteresis
        frame_basic = tk.LabelFrame(win, text="Detection / Tracking / Hysteresis"); frame_basic.pack(fill="x", padx=8, pady=6)
        v_imgsz = tk.StringVar(value=str(base.get("imgsz", "")))
        v_conf  = tk.StringVar(value=str(base.get("conf", "")))
        v_iou   = tk.StringVar(value=str(base.get("iou", "")))
        v_skip  = tk.StringVar(value=str(base.get("frame_skip", "")))
        v_buf   = tk.StringVar(value=str(base.get("track_buffer", "")))
        v_match = tk.StringVar(value=str(base.get("match_thresh", "")))
        v_hits  = tk.StringVar(value=str(base.get("min_hits", "")))
        v_lgap  = tk.StringVar(value=str(base.get("line_min_gap", "")))
        v_lsep  = tk.StringVar(value=str(base.get("line_min_sep", "")))
        v_zhgap = tk.StringVar(value=str(base.get("zone_min_gap", "")))
        add_row(frame_basic, "imgsz", v_imgsz)
        add_row(frame_basic, "conf", v_conf)
        add_row(frame_basic, "iou", v_iou)
        add_row(frame_basic, "frame_skip", v_skip)
        add_row(frame_basic, "track_buffer", v_buf)
        add_row(frame_basic, "match_thresh", v_match)
        add_row(frame_basic, "min_hits", v_hits)
        add_row(frame_basic, "line_min_gap_frames", v_lgap)
        add_row(frame_basic, "line_min_sep_px", v_lsep)
        add_row(frame_basic, "zone_min_gap_frames", v_zhgap)

        # Visualization / Trace / Anchor / Ghost
        frame_vis = tk.LabelFrame(win, text="Visualization / Trace / Anchor / Ghost"); frame_vis.pack(fill="x", padx=8, pady=6)
        v_prev  = tk.BooleanVar(value=bool(base.get("preview_enabled", True)))
        v_trace = tk.BooleanVar(value=bool(base.get("trace_enabled", True)))
        v_tlen  = tk.IntVar(value=int(base.get("trace_len", 24)))
        v_anch  = tk.StringVar(value=str(base.get("anchor_mode", "center")))
        v_ghost = tk.IntVar(value=int(base.get("ghost_margin", 12)))
        tk.Checkbutton(frame_vis, text="Enable LIVE preview", variable=v_prev).pack(side="left", padx=6, pady=4)
        tk.Checkbutton(frame_vis, text="Trace", variable=v_trace).pack(side="left", padx=(12,4))
        tk.Label(frame_vis, text="len:").pack(side="left")
        tk.Spinbox(frame_vis, from_=0, to=300, width=5, textvariable=v_tlen).pack(side="left", padx=(2, 12))
        tk.Label(frame_vis, text="Anchor:").pack(side="left")
        ttk.Combobox(frame_vis, values=["bottom","center"], width=8, state="readonly",
                     textvariable=v_anch).pack(side="left", padx=(3, 12))
        tk.Label(frame_vis, text="Ghost margin (px):").pack(side="left")
        tk.Spinbox(frame_vis, from_=0, to=64, width=5, textvariable=v_ghost).pack(side="left", padx=(3, 6))

        # Audio alert (zones)
        frame_alert = tk.LabelFrame(win, text="Audio alert (zones)"); frame_alert.pack(fill="x", padx=8, pady=6)
        v_a_en   = tk.BooleanVar(value=bool(base.get("alert_enabled", False)))
        v_a_cls  = tk.StringVar(value=str(base.get("alert_classes", "cat,person")))
        v_a_freq = tk.IntVar(value=int(base.get("alert_freq", 880)))
        v_a_dur  = tk.IntVar(value=int(base.get("alert_dur", 180)))
        v_a_free = tk.IntVar(value=int(base.get("alert_freeze", 1500)))
        tk.Checkbutton(frame_alert, text="Enable alert", variable=v_a_en).pack(side="left", padx=6)
        tk.Label(frame_alert, text="Classes (CSV):").pack(side="left")
        tk.Entry(frame_alert, textvariable=v_a_cls, width=22).pack(side="left", padx=(3, 10))
        tk.Label(frame_alert, text="Hz:").pack(side="left")
        tk.Spinbox(frame_alert, from_=200, to=4000, width=6, textvariable=v_a_freq).pack(side="left", padx=(3, 10))
        tk.Label(frame_alert, text="ms:").pack(side="left")
        tk.Spinbox(frame_alert, from_=30, to=2000, width=6, textvariable=v_a_dur).pack(side="left", padx=(3, 10))
        tk.Label(frame_alert, text="cooldown (ms):").pack(side="left")
        tk.Spinbox(frame_alert, from_=0, to=10000, width=7, textvariable=v_a_free).pack(side="left", padx=(3, 6))

        # aggregate fields
        def _collect_from_fields() -> dict:
            cur = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
            def get_or(var, cast, key, default):
                s = var.get().strip() if isinstance(var, tk.StringVar) else str(var.get())
                if s != "":
                    try: return cast(s)
                    except Exception: raise ValueError(f"Field '{key}' has invalid value: {s}")
                return default if key not in cur else cur[key]
            data = {
                "imgsz": get_or(v_imgsz, int,   "imgsz",        960),
                "conf":  get_or(v_conf,  float, "conf",         0.60),
                "iou":   get_or(v_iou,   float, "iou",          0.50),
                "frame_skip":   get_or(v_skip,  int,   "frame_skip",   1),
                "track_buffer": get_or(v_buf,   int,   "track_buffer", 60),
                "match_thresh": get_or(v_match, float, "match_thresh", 0.78),
                "min_hits":     get_or(v_hits,  int,   "min_hits",     2),
                "line_min_gap": int(v_lgap.get().strip() or LINE_MIN_GAP_FRAMES_DEFAULT),
                "line_min_sep": int(v_lsep.get().strip() or LINE_MIN_SEP_PX_DEFAULT),
                "zone_min_gap": int(v_zhgap.get().strip() or ZONE_MIN_GAP_FRAMES_DEFAULT),
                # extra (visualization / alert)
                "preview_enabled": bool(v_prev.get()),
                "trace_enabled": bool(v_trace.get()),
                "trace_len": int(v_tlen.get()),
                "anchor_mode": v_anch.get(),
                "ghost_margin": int(v_ghost.get()),
                "alert_enabled": bool(v_a_en.get()),
                "alert_classes": v_a_cls.get(),
                "alert_freq": int(v_a_freq.get()),
                "alert_dur": int(v_a_dur.get()),
                "alert_freeze": int(v_a_free.get()),
            }
            return data

        PRESETS_DIR = Path(__file__).parent / "presets"
        PRESETS_DIR.mkdir(exist_ok=True)

        def do_save_preset():
            try:
                data = _collect_from_fields()
            except Exception as e:
                messagebox.showerror("Preset", str(e)); return
            defname = f"preset_q{self.quality.get()}.json"
            path = filedialog.asksaveasfilename(
                title="Save preset (JSON)",
                defaultextension=".json",
                initialdir=str(PRESETS_DIR),
                initialfile=defname,
                filetypes=[("JSON","*.json")]
            )
            if not path: return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self._log(f"[ADV] Preset saved: {path}")
            except Exception as e:
                messagebox.showerror("Save preset", str(e))

        def do_load_preset():
            path = filedialog.askopenfilename(
                title="Load preset (JSON)",
                initialdir=str(PRESETS_DIR),
                filetypes=[("JSON","*.json"), ("All","*.*")]
            )
            if not path: return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, var in [
                    ("imgsz", v_imgsz), ("conf", v_conf), ("iou", v_iou),
                    ("frame_skip", v_skip), ("track_buffer", v_buf),
                    ("match_thresh", v_match), ("min_hits", v_hits),
                    ("line_min_gap", v_lgap), ("line_min_sep", v_lsep), ("zone_min_gap", v_zhgap),
                ]:
                    if key in data: var.set(str(data[key]))
                if "preview_enabled" in data: v_prev.set(bool(data["preview_enabled"]))
                if "trace_enabled" in data:  v_trace.set(bool(data["trace_enabled"]))
                if "trace_len" in data:      v_tlen.set(int(data["trace_len"]))
                if "anchor_mode" in data:    v_anch.set(str(data["anchor_mode"]))
                if "ghost_margin" in data:   v_ghost.set(int(data["ghost_margin"]))
                if "alert_enabled" in data:  v_a_en.set(bool(data["alert_enabled"]))
                if "alert_classes" in data:  v_a_cls.set(str(data["alert_classes"]))
                if "alert_freq" in data:     v_a_freq.set(int(data["alert_freq"]))
                if "alert_dur" in data:      v_a_dur.set(int(data["alert_dur"]))
                if "alert_freeze" in data:   v_a_free.set(int(data["alert_freeze"]))
                self._log(f"[ADV] Preset loaded: {path}")
            except Exception as e:
                messagebox.showerror("Load preset", str(e))

        def _apply():
            try:
                params = _collect_from_fields()
                self.adv_params = params
                self.advanced_override = True
                # propagate "extra" fields used by run()
                self.preview_enabled.set(params["preview_enabled"])
                self.trace_enabled.set(params["trace_enabled"])
                self.trace_len.set(params["trace_len"])
                self.anchor_mode.set(params["anchor_mode"])
                self.ghost_margin.set(params["ghost_margin"])
                self.alert_enabled.set(params["alert_enabled"])
                self.alert_classes.set(params["alert_classes"])
                self.alert_freq.set(params["alert_freq"])
                self.alert_dur.set(params["alert_dur"])
                self.alert_freeze.set(params["alert_freeze"])
                self._log("[ADV] Override applied (from fields/preset).")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Advanced", str(e))

        def _reset():
            self.advanced_override = False
            self._log("[ADV] Reverted to slider preset.")
            win.destroy()

        # Buttons
        btns = tk.Frame(win); btns.pack(fill="x", pady=10)
        tk.Button(btns, text="Apply", command=_apply).pack(side="left", padx=6)
        tk.Button(btns, text="Revert to slider preset", command=_reset).pack(side="left", padx=6)
        tk.Button(btns, text="Save preset…", command=do_save_preset).pack(side="right", padx=6)
        tk.Button(btns, text="Load preset…", command=do_load_preset).pack(side="right", padx=6)

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
                    inp = Path(self.input_dir.get().strip())
                    if not inp.exists():
                        messagebox.showerror("Input", "Select a valid folder or choose files.")
                        self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                    sources = sorted([p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED_VID_EXTS])
                    if not sources:
                        self._log("No video files in folder.")
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

            out_base = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else (base_in or Path.cwd())
            outp = ensure_dir(out_base / "results")

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

    # ========== LIVE preview window ==========
    def _ensure_preview_window(self):
        if self._preview_win and (self._preview_win.winfo_exists()):
            return
        win = tk.Toplevel(self)
        win.title("Preview (LIVE)")
        win.geometry("860x520")
        win.protocol("WM_DELETE_WINDOW", lambda: self._destroy_preview_window())
        lbl = tk.Label(win, anchor="center", bg="#111")
        lbl.pack(fill="both", expand=True)
        self._preview_win = win
        self._preview_lbl = lbl

    def _show_preview_bgr(self, frame_bgr):
        if not self.preview_enabled.get():
            return
        def _do():
            try:
                self._ensure_preview_window()
                frame = frame_bgr
                h, w = frame.shape[:2]
                maxw = 840
                if w > maxw:
                    r = maxw / w
                    frame = cv2.resize(frame, (int(maxw), int(h*r)))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self._preview_imgtk = imgtk
                self._preview_lbl.config(image=imgtk)
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


if __name__ == "__main__":
    app = App(); app.mainloop()

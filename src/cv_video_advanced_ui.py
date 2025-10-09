# cv_video_advanced_ui.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
from pathlib import Path
import json
from datetime import datetime

# ───────────────────── path helpers ─────────────────────
def _get_project_root() -> Path:
    p = Path(__file__).resolve()
    if "src" in p.parts:
        i = p.parts.index("src")
        return Path(*p.parts[:i])
    return p.parent

def _sounds_dir() -> Path:
    d = _get_project_root() / "sounds"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _presets_dir() -> Path:
    d = _get_project_root() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ───────────────────── value helpers ─────────────────────
def _int(var, default=0) -> int:
    try: return int(str(var.get()).strip())
    except Exception: return int(default)

def _float(var, default=0.0) -> float:
    try: return float(str(var.get()).strip().replace(",", "."))
    except Exception: return float(default)

def _str(var, default="") -> str:
    try: return str(var.get()).strip()
    except Exception: return str(default)

# ───────────────────── UI helpers ─────────────────────
def _row(parent, label_text: str, build_widget) -> tuple[ttk.Frame, tk.Widget]:
    """Horizontal row: left label + widget built by build_widget(frame)."""
    fr = ttk.Frame(parent)
    ttk.Label(fr, text=label_text, width=22, anchor="w").pack(side="left")
    w = build_widget(fr)
    try:
        if isinstance(w, tk.Widget):
            w.pack(side="left")
    except Exception:
        pass
    fr.pack(fill="x", padx=6, pady=(2, 2))
    return fr, w

def _pick_color_into(var: tk.StringVar):
    try:
        init = var.get().strip()
        if not init or init.lower() == "auto":
            init = "#FFFFFF"
        if not init.startswith("#") and "," in init:
            b, g, r = [int(x.strip()) for x in init.split(",")]
            init = f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        init = "#FFFFFF"
    _, hexv = colorchooser.askcolor(color=init)
    if hexv:
        var.set(hexv)

# ───────────────────── contextual help ─────────────────────
HELP = {
    "imgsz": "Network input size (pixels). Lower = faster; higher catches smaller objects. 320–640 typical.",
    "conf": "Detection confidence threshold. Detections below this score are ignored.",
    "iou": "IoU threshold for NMS (merging overlapping boxes). Higher keeps more overlaps.",
    "frame_skip": "Process every Nth frame (1 = all frames). Higher = faster, lower temporal detail.",
    "track_buffer": "How long a lost track is kept (frames). Helps maintain IDs across brief occlusions.",
    "match_thresh": "Tracker association threshold. Higher = stricter matching between frames.",
    "min_hits": "Frames required before a new track is trusted. Filters short flickers.",
    "line_min_gap": "Debounce for line counts (frames) per object to prevent double counting.",
    "line_min_sep": "Minimal pixel travel between two line events per object.",
    "zone_min_gap": "Debounce for entering/leaving zones (frames) per object.",
    "live_preview": "Show a live preview window during processing.",
    "trace_enabled": "Draw trails of recent object movement.",
    "trace_len": "Number of points kept in each trail.",
    "anchor_mode": "Which point is used for counters: center or bottom edge (feet on ground).",
    "ghost_margin": "If anchor is bottom, shift it upward by this many pixels (avoid border touches).",
    "trace_color": "Trail color. ‘auto’ = per-track. Also accepts #RRGGBB or B,G,R.",
    "trace_thickness": "Trail line thickness (pixels).",
    "overlay_frame_color": "Color of lines/zones. ‘auto’ = default orange.",
    "overlay_frame_thickness": "Line/zone thickness (pixels).",
    "alert_enabled": "Play a sound when selected classes are inside/outside the zone.",
    "alert_sound": "Audio file to play (.wav/.mp3/…).",
    "alert_loop": "Loop sound while active (on) or play ping with cooldown (off).",
    "alert_freeze_s": "Cooldown between pings in seconds (used when looping is off).",
    "alert_zone_inside": "Alert condition: inside (1) vs outside (0) the zone.",
    "hud_scale": "Size of the bottom-right HUD panel relative to auto scaling. 100% = default.",
}

def _bind_help(widget: tk.Widget, key: str, help_label: tk.Label):
    def on(_e=None): help_label.config(text=HELP.get(key, ""))
    widget.bind("<FocusIn>", on)
    widget.bind("<Enter>", on)

# ───────────────────── main builder ─────────────────────
def build_advanced_settings(parent: tk.Misc, app) -> ttk.Frame:
    """
    Build Advanced dialog as a Notebook with tabs:
      • Main  – existing options
      • Extras – future features; includes HUD size control now
    """
    if not hasattr(app, "adv_params"):
        app.adv_params = {}
    p = app.adv_params

    defaults = {
        "imgsz": p.get("imgsz", 320),
        "conf": p.get("conf", 0.5),
        "iou": p.get("iou", 0.6),
        "frame_skip": p.get("frame_skip", 2),
        "track_buffer": p.get("track_buffer", 5),
        "match_thresh": p.get("match_thresh", 0.8),
        "min_hits": p.get("min_hits", 2),
        "line_min_gap": p.get("line_min_gap", 8),
        "line_min_sep": p.get("line_min_sep", 12),
        "zone_min_gap": p.get("zone_min_gap", 6),
        "live_preview": p.get("live_preview", True),
        "trace_enabled": p.get("trace_enabled", True),
        "trace_len": p.get("trace_len", 24),
        "anchor_mode": p.get("anchor_mode", "center"),
        "ghost_margin": p.get("ghost_margin", 24),
        "trace_color": p.get("trace_color", "auto"),
        "trace_thickness": p.get("trace_thickness", 2),
        "overlay_frame_color": p.get("overlay_frame_color", "auto"),
        "overlay_frame_thickness": p.get("overlay_frame_thickness", 2),
        "alert_enabled": p.get("alert_enabled", False),
        "alert_sound": p.get("alert_sound", ""),
        "alert_loop": p.get("alert_loop", True),
        "alert_freeze_s": p.get("alert_freeze_s", 2),
        "alert_zone_inside": p.get("alert_zone_inside", 1),
        # NEW:
        "hud_scale": float(p.get("hud_scale", 1.0)),  # 1.0 = 100%
    }

    # tk variables (re-use if present)
    def _ensure(name, cls, value):
        if hasattr(app, name) and isinstance(getattr(app, name), cls):
            return getattr(app, name)
        v = cls(value=value)
        setattr(app, name, v)
        return v

    # Main vars
    v_imgsz        = _ensure("v_imgsz", tk.StringVar, str(defaults["imgsz"]))
    v_conf         = _ensure("v_conf", tk.StringVar, str(defaults["conf"]))
    v_iou          = _ensure("v_iou", tk.StringVar, str(defaults["iou"]))
    v_frame_skip   = _ensure("v_frame_skip", tk.StringVar, str(defaults["frame_skip"]))
    v_track_buffer = _ensure("v_track_buffer", tk.StringVar, str(defaults["track_buffer"]))
    v_match_thresh = _ensure("v_match_thresh", tk.StringVar, str(defaults["match_thresh"]))
    v_min_hits     = _ensure("v_min_hits", tk.StringVar, str(defaults["min_hits"]))
    v_line_gap     = _ensure("v_line_gap", tk.StringVar, str(defaults["line_min_gap"]))
    v_line_sep     = _ensure("v_line_sep", tk.StringVar, str(defaults["line_min_sep"]))
    v_zone_gap     = _ensure("v_zone_gap", tk.StringVar, str(defaults["zone_min_gap"]))

    v_live_preview = _ensure("preview_enabled", tk.BooleanVar, bool(defaults["live_preview"]))
    v_trace_en     = _ensure("trace_enabled", tk.BooleanVar, bool(defaults["trace_enabled"]))
    v_trace_len    = _ensure("trace_len", tk.StringVar, str(defaults["trace_len"]))
    v_anchor_mode  = _ensure("anchor_mode", tk.StringVar, str(defaults["anchor_mode"]))
    v_ghost_margin = _ensure("ghost_margin", tk.StringVar, str(defaults["ghost_margin"]))

    v_trace_color  = _ensure("trace_color", tk.StringVar, str(defaults["trace_color"]))
    v_trace_thick  = _ensure("trace_thickness", tk.StringVar, str(defaults["trace_thickness"]))
    v_frame_color  = _ensure("frame_color", tk.StringVar, str(defaults["overlay_frame_color"]))
    v_frame_thick  = _ensure("frame_thickness", tk.StringVar, str(defaults["overlay_frame_thickness"]))

    v_alert_en     = _ensure("alert_enabled", tk.BooleanVar, bool(defaults["alert_enabled"]))
    v_alert_sound  = _ensure("alert_sound", tk.StringVar, str(defaults["alert_sound"]))
    v_alert_loop   = _ensure("alert_loop", tk.BooleanVar, bool(defaults["alert_loop"]))
    v_alert_freeze = _ensure("alert_freeze_s", tk.IntVar, int(defaults["alert_freeze_s"]))
    v_alert_inside = _ensure("alert_inside", tk.IntVar, int(defaults["alert_zone_inside"]))

    # Extras vars
    v_hud_scale    = _ensure("v_hud_scale", tk.StringVar, str(int(round(float(defaults["hud_scale"]) * 100))))

    # Notebook
    nb = ttk.Notebook(parent)

    # =========== Main tab ===========
    main = ttk.Frame(nb); nb.add(main, text="Main")

    left = ttk.Frame(main);  left.pack(side="left", fill="both", expand=True)
    right = ttk.LabelFrame(main, text="Help"); right.pack(side="left", fill="both", expand=True, padx=(8, 0))
    help_lbl = tk.Label(right, text="Focus a field to see help.", justify="left", anchor="nw", wraplength=420)
    help_lbl.pack(fill="both", expand=True, padx=8, pady=8)

    # Detection / Tracking / Hysteresis
    lf = ttk.LabelFrame(left, text="Detection / Tracking / Hysteresis"); lf.pack(fill="x", padx=6, pady=(6, 4))
    _, e = _row(lf, "imgsz",        lambda fr: ttk.Entry(fr, textvariable=v_imgsz, width=10));        _bind_help(e, "imgsz", help_lbl)
    _, e = _row(lf, "conf",         lambda fr: ttk.Entry(fr, textvariable=v_conf, width=10));         _bind_help(e, "conf", help_lbl)
    _, e = _row(lf, "iou",          lambda fr: ttk.Entry(fr, textvariable=v_iou, width=10));          _bind_help(e, "iou", help_lbl)
    _, e = _row(lf, "frame_skip",   lambda fr: ttk.Entry(fr, textvariable=v_frame_skip, width=10));   _bind_help(e, "frame_skip", help_lbl)
    _, e = _row(lf, "track_buffer", lambda fr: ttk.Entry(fr, textvariable=v_track_buffer, width=10)); _bind_help(e, "track_buffer", help_lbl)
    _, e = _row(lf, "match_thresh", lambda fr: ttk.Entry(fr, textvariable=v_match_thresh, width=10)); _bind_help(e, "match_thresh", help_lbl)
    _, e = _row(lf, "min_hits",     lambda fr: ttk.Entry(fr, textvariable=v_min_hits, width=10));     _bind_help(e, "min_hits", help_lbl)
    _, e = _row(lf, "line_min_gap_frames", lambda fr: ttk.Entry(fr, textvariable=v_line_gap, width=10)); _bind_help(e, "line_min_gap", help_lbl)
    _, e = _row(lf, "line_min_sep_px",     lambda fr: ttk.Entry(fr, textvariable=v_line_sep, width=10)); _bind_help(e, "line_min_sep", help_lbl)
    _, e = _row(lf, "zone_min_gap_frames", lambda fr: ttk.Entry(fr, textvariable=v_zone_gap, width=10)); _bind_help(e, "zone_min_gap", help_lbl)

    # Overlay / Trace / Anchor / Ghost
    lf = ttk.LabelFrame(left, text="Overlay / Trace / Anchor / Ghost"); lf.pack(fill="x", padx=6, pady=(4, 4))
    _, cb = _row(lf, "", lambda fr: ttk.Checkbutton(fr, text="Enable LIVE preview", variable=v_live_preview)); _bind_help(cb, "live_preview", help_lbl)

    def _trace_row(fr):
        box = ttk.Frame(fr)
        c = ttk.Checkbutton(box, text="Trace", variable=v_trace_en); c.pack(side="left")
        _bind_help(c, "trace_enabled", help_lbl)
        ttk.Label(box, text="len:").pack(side="left", padx=(8, 2))
        sp = ttk.Spinbox(box, from_=1, to=240, textvariable=v_trace_len, width=5); sp.pack(side="left")
        _bind_help(sp, "trace_len", help_lbl)
        return box
    _row(lf, "", _trace_row)

    def _anchor_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Anchor:").pack(side="left")
        om = ttk.OptionMenu(box, v_anchor_mode, v_anchor_mode.get(), "center", "bottom")
        om.pack(side="left", padx=(4, 12))
        _bind_help(om, "anchor_mode", help_lbl)
        ttk.Label(box, text="Ghost margin (px):").pack(side="left")
        sp = ttk.Spinbox(box, from_=0, to=256, textvariable=v_ghost_margin, width=6)
        sp.pack(side="left"); _bind_help(sp, "ghost_margin", help_lbl)
        return box
    _row(lf, "", _anchor_row)

    # Colors/Thickness (overlay)
    lf = ttk.LabelFrame(left, text="Colors/Thickness (overlay)"); lf.pack(fill="x", padx=6, pady=(4, 4))

    def _trace_color_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Trace color (auto/#RRGGBB/B,G,R)").pack(side="left")
        e = ttk.Entry(box, textvariable=v_trace_color, width=14); e.pack(side="left", padx=(6, 6))
        ttk.Button(box, text="Pick…", command=lambda: _pick_color_into(v_trace_color)).pack(side="left")
        _bind_help(e, "trace_color", help_lbl)
        return box
    _row(lf, "", _trace_color_row)

    def _trace_th_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Trace thickness (px)").pack(side="left")
        sp = ttk.Spinbox(box, from_=1, to=12, textvariable=v_trace_thick, width=6); sp.pack(side="left", padx=(6, 6))
        _bind_help(sp, "trace_thickness", help_lbl)
        return box
    _row(lf, "", _trace_th_row)

    def _frame_color_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Frame color (auto/#RRGGBB/B,G,R)").pack(side="left")
        e = ttk.Entry(box, textvariable=v_frame_color, width=14); e.pack(side="left", padx=(6, 6))
        ttk.Button(box, text="Pick…", command=lambda: _pick_color_into(v_frame_color)).pack(side="left")
        _bind_help(e, "overlay_frame_color", help_lbl)
        return box
    _row(lf, "", _frame_color_row)

    def _frame_th_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Frame thickness (px)").pack(side="left")
        sp = ttk.Spinbox(box, from_=1, to=12, textvariable=v_frame_thick, width=6); sp.pack(side="left", padx=(6, 6))
        _bind_help(sp, "overlay_frame_thickness", help_lbl)
        return box
    _row(lf, "", _frame_th_row)

    # Sound alert (zones)
    lf = ttk.LabelFrame(left, text="Sound alert (zones)"); lf.pack(fill="x", padx=6, pady=(4, 6))

    _, cb = _row(lf, "", lambda fr: ttk.Checkbutton(fr, text="Enable alert (uses selected classes)", variable=v_alert_en))
    _bind_help(cb, "alert_enabled", help_lbl)

    def _sound_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Sound file:").pack(side="left")
        e = ttk.Entry(box, textvariable=v_alert_sound, width=42); e.pack(side="left", padx=(6, 6), fill="x", expand=True)
        def _browse():
            path = filedialog.askopenfilename(
                title="Pick alert sound",
                initialdir=str(_sounds_dir()),
                filetypes=[("Audio", "*.wav;*.mp3;*.ogg;*.flac;*.aac;*.m4a"), ("WAV", "*.wav"), ("All files", "*.*")],
            )
            if path: v_alert_sound.set(path)
        ttk.Button(box, text="Browse…", command=_browse).pack(side="left", padx=(0, 6))
        def _test():
            fn = getattr(app, "_play_test_sound", None)
            if callable(fn): fn(_str(v_alert_sound))
            else: messagebox.showinfo("Test sound", "No test function available in this build.")
        ttk.Button(box, text="▶ Test", command=_test).pack(side="left")
        _bind_help(e, "alert_sound", help_lbl)
        return box
    _row(lf, "", _sound_row)

    def _loop_row(fr):
        box = ttk.Frame(fr)
        cb = ttk.Checkbutton(box, text="Loop while active", variable=v_alert_loop); cb.pack(side="left")
        _bind_help(cb, "alert_loop", help_lbl)
        ttk.Label(box, text="freeze (s):").pack(side="left", padx=(10, 2))
        sp = ttk.Spinbox(box, from_=0, to=30, textvariable=v_alert_freeze, width=6); sp.pack(side="left")
        _bind_help(sp, "alert_freeze_s", help_lbl)
        return box
    _row(lf, "", _loop_row)

    def _mode_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Mode:").pack(side="left")
        rb1 = ttk.Radiobutton(box, text="inside zone", variable=v_alert_inside, value=1); rb1.pack(side="left", padx=(6, 6))
        rb2 = ttk.Radiobutton(box, text="outside zone", variable=v_alert_inside, value=0); rb2.pack(side="left", padx=(6, 6))
        _bind_help(rb1, "alert_zone_inside", help_lbl); _bind_help(rb2, "alert_zone_inside", help_lbl)
        return box
    _row(lf, "", _mode_row)

    # Buttons (Apply / Restore / Save / Load)
    def _collect() -> dict:
        return {
            "imgsz": _int(v_imgsz, defaults["imgsz"]),
            "conf": _float(v_conf, defaults["conf"]),
            "iou": _float(v_iou, defaults["iou"]),
            "frame_skip": _int(v_frame_skip, defaults["frame_skip"]),
            "track_buffer": _int(v_track_buffer, defaults["track_buffer"]),
            "match_thresh": _float(v_match_thresh, defaults["match_thresh"]),
            "min_hits": _int(v_min_hits, defaults["min_hits"]),
            "line_min_gap": _int(v_line_gap, defaults["line_min_gap"]),
            "line_min_sep": _int(v_line_sep, defaults["line_min_sep"]),
            "zone_min_gap": _int(v_zone_gap, defaults["zone_min_gap"]),
            "live_preview": bool(v_live_preview.get()),
            "trace_enabled": bool(v_trace_en.get()),
            "trace_len": _int(v_trace_len, defaults["trace_len"]),
            "anchor_mode": _str(v_anchor_mode, defaults["anchor_mode"]),
            "ghost_margin": _int(v_ghost_margin, defaults["ghost_margin"]),
            "trace_color": _str(v_trace_color, defaults["trace_color"]),
            "trace_thickness": _int(v_trace_thick, defaults["trace_thickness"]),
            "overlay_frame_color": _str(v_frame_color, defaults["overlay_frame_color"]),
            "overlay_frame_thickness": _int(v_frame_thick, defaults["overlay_frame_thickness"]),
            "alert_enabled": bool(v_alert_en.get()),
            "alert_sound": _str(v_alert_sound, defaults["alert_sound"]),
            "alert_loop": bool(v_alert_loop.get()),
            "alert_freeze_s": _int(v_alert_freeze, defaults["alert_freeze_s"]),
            "alert_zone_inside": _int(v_alert_inside, defaults["alert_zone_inside"]),
            "hud_scale": max(0.5, min(2.0, _int(v_hud_scale, 100) / 100.0)),
            "_meta": {"version": 2, "saved_at": datetime.now().isoformat(timespec="seconds")},
        }

    bar = ttk.Frame(left); bar.pack(fill="x", padx=6, pady=(0, 8))
    def _apply():
        app.adv_params.update(_collect())
        messagebox.showinfo("Advanced options", "Saved. Values are now active for the next run.")
    ttk.Button(bar, text="Apply", command=_apply).pack(side="left")

    def _restore():
        fn = getattr(app, "restore_adv_from_slider", None)
        if callable(fn): fn()
        else: messagebox.showinfo("Preset", "No preset restore function found in this build.")
    ttk.Button(bar, text="Restore preset from slider", command=_restore).pack(side="left", padx=(8, 0))

    def _save_preset():
        data = _collect()
        initialdir = str(_presets_dir())
        fname = f"adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(
            title="Save preset",
            initialdir=initialdir,
            defaultextension=".json",
            initialfile=fname,
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Preset", f"Saved preset:\n{path}")
        except Exception as e:
            messagebox.showerror("Preset", f"Could not save:\n{e}")
    ttk.Button(bar, text="Save preset…", command=_save_preset).pack(side="left", padx=(8, 0))

    def _load_preset():
        initialdir = str(_presets_dir())
        path = filedialog.askopenfilename(
            title="Load preset",
            initialdir=initialdir,
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            messagebox.showerror("Preset", f"Could not read file:\n{e}")
            return

        def _getk(*names, default=None):
            for n in names:
                if n in d: return d[n]
            return default

        v_imgsz.set(str(_getk("imgsz", default=defaults["imgsz"])))
        v_conf.set(str(_getk("conf", default=defaults["conf"])))
        v_iou.set(str(_getk("iou", default=defaults["iou"])))
        v_frame_skip.set(str(_getk("frame_skip", default=defaults["frame_skip"])))
        v_track_buffer.set(str(_getk("track_buffer", default=defaults["track_buffer"])))
        v_match_thresh.set(str(_getk("match_thresh", default=defaults["match_thresh"])))
        v_min_hits.set(str(_getk("min_hits", default=defaults["min_hits"])))
        v_line_gap.set(str(_getk("line_min_gap", default=defaults["line_min_gap"])))
        v_line_sep.set(str(_getk("line_min_sep", "line_min_sep_px", default=defaults["line_min_sep"])))
        v_zone_gap.set(str(_getk("zone_min_gap", default=defaults["zone_min_gap"])))

        v_live_preview.set(bool(_getk("live_preview", default=defaults["live_preview"])))
        v_trace_en.set(bool(_getk("trace_enabled", default=defaults["trace_enabled"])))
        v_trace_len.set(str(_getk("trace_len", default=defaults["trace_len"])))
        v_anchor_mode.set(str(_getk("anchor_mode", default=defaults["anchor_mode"])))
        v_ghost_margin.set(str(_getk("ghost_margin", default=defaults["ghost_margin"])))

        v_trace_color.set(str(_getk("trace_color", default=defaults["trace_color"])))
        v_trace_thick.set(str(_getk("trace_thickness", default=defaults["trace_thickness"])))
        v_frame_color.set(str(_getk("overlay_frame_color", "frame_color", default=defaults["overlay_frame_color"])))
        v_frame_thick.set(str(_getk("overlay_frame_thickness", "frame_thickness", default=defaults["overlay_frame_thickness"])))

        v_alert_en.set(bool(_getk("alert_enabled", default=defaults["alert_enabled"])))
        v_alert_sound.set(str(_getk("alert_sound", default=defaults["alert_sound"])))
        v_alert_loop.set(bool(_getk("alert_loop", default=defaults["alert_loop"])))
        v_alert_freeze.set(int(_getk("alert_freeze_s", default=defaults["alert_freeze_s"])))
        v_alert_inside.set(int(_getk("alert_zone_inside", default=defaults["alert_zone_inside"])))

        v_hud_scale.set(str(int(round(float(_getk("hud_scale", default=defaults["hud_scale"])) * 100))))

        # auto-apply so it's active
        app.adv_params.update(_collect())
        messagebox.showinfo("Preset", f"Loaded and applied:\n{path}")
    ttk.Button(bar, text="Load preset…", command=_load_preset).pack(side="left", padx=(8, 0))

    # =========== Extras tab ===========
    extras = ttk.Frame(nb); nb.add(extras, text="Extras")

    ex_left = ttk.LabelFrame(extras, text="HUD / Stats / Heatmaps"); ex_left.pack(fill="x", padx=6, pady=6)

    def _hud_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="HUD size (%)").pack(side="left")
        sp = ttk.Spinbox(box, from_=50, to=200, increment=5, textvariable=v_hud_scale, width=6)
        sp.pack(side="left", padx=(6, 6))
        return box
    _row(ex_left, "", _hud_row)

    # placeholders for future options
    ph = ttk.Frame(ex_left); ph.pack(fill="x", padx=6, pady=(6, 2))
    ttk.Checkbutton(ph, text="[soon] Movement heatmap", state="disabled").pack(anchor="w")
    ttk.Checkbutton(ph, text="[soon] Dwell-time heatmap", state="disabled").pack(anchor="w")
    ttk.Checkbutton(ph, text="[soon] Save tracks CSV", state="disabled").pack(anchor="w")

    return nb

# cv_video_advanced_ui.py — Advanced settings UI (Extras ▸ Heatmap with On/Off, Gain, Memory×)
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
from datetime import datetime

try:
    from cv_video_sound import SoundPlayer
except Exception:
    SoundPlayer = None

HELP_WIDTH = 420

# ───────────────────────────────── helpers ──────────────────────────────────
def _get_project_root() -> Path:
    p = Path(__file__).resolve()
    if "src" in p.parts:
        i = p.parts.index("src")
        return Path(*p.parts[:i])
    return p.parent

def _dir(pathname) -> Path:
    d = _get_project_root() / pathname
    d.mkdir(parents=True, exist_ok=True)
    return d

def _int(var, default=0) -> int:
    try: return int(str(var.get()).strip())
    except Exception: return int(default)

def _float(var, default=0.0) -> float:
    try: return float(str(var.get()).strip().replace(",", "."))
    except Exception: return float(default)

def _str(var, default="") -> str:
    try: return str(var.get()).strip()
    except Exception: return str(default)

def _row(parent, label_text: str, build_widget) -> tuple[ttk.Frame, tk.Widget]:
    fr = ttk.Frame(parent)
    ttk.Label(fr, text=label_text, width=22, anchor="w").pack(side="left")
    w = build_widget(fr)
    try:
        if isinstance(w, tk.Widget): w.pack(side="left")
    except Exception:
        pass
    fr.pack(fill="x", padx=6, pady=(2, 2))
    return fr, w

def _make_scrollable(parent: tk.Misc) -> ttk.Frame:
    container = ttk.Frame(parent)
    canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
    vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    interior = ttk.Frame(canvas)
    win_id = canvas.create_window((0, 0), window=interior, anchor="nw")

    def _on_config(_=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(win_id, width=canvas.winfo_width())

    interior.bind("<Configure>", _on_config)
    canvas.bind("<Configure>", _on_config)

    def _wheel(e): canvas.yview_scroll(int(-1*(e.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _wheel)

    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    container.pack(fill="both", expand=True)
    return interior

def _make_help_panel(parent: tk.Misc, title="Help", width=HELP_WIDTH):
    fr = ttk.LabelFrame(parent, text=title)
    fr.pack_propagate(False)
    fr.config(width=width)
    lbl = tk.Label(fr, text="Focus a field to see help.", justify="left", anchor="nw")
    lbl.pack(fill="both", expand=True, padx=8, pady=8)
    lbl.config(wraplength=max(50, width - 16))
    return fr, lbl

HELP = {
    "imgsz": "Image size for the AI. Larger = slower but better detail.",
    "conf": "Confidence threshold for detections.",
    "iou": "IoU used to merge duplicates.",
    "frame_skip": "Process every Nth frame (0/1 = all frames).",
    "track_buffer": "How long (frames) an ID is kept after disappearing.",
    "match_thresh": "Tracker matching strictness.",
    "min_hits": "Frames before a track is confirmed.",
    "line_min_gap": "Anti-bounce gap between line counts.",
    "line_min_sep": "Min movement across a line (px).",
    "zone_min_gap": "Anti-flicker gap for zone in/out.",

    "live_preview": "Show the processed video while running.",
    "overlay_mode": "centroid = dots; box = rectangles; box+conf = rectangles + score.",
    "trace_enabled": "Draw a trail for each track.",
    "trace_len": "Trail length (points).",
    "anchor_mode": "Counting point: center or bottom of the box.",
    "ghost_margin": "Offset for bottom anchor to avoid line bouncing.",

    "trace_color": "auto, #RRGGBB, or B,G,R.",
    "trace_thickness": "Trail thickness (px).",
    "overlay_frame_color": "Line/zone color (auto/#RRGGBB/B,G,R).",
    "overlay_frame_thickness": "Line/zone thickness (px).",

    "alert_enabled": "Play a sound when the condition is met.",
    "alert_sound": "Audio file to play.",
    "alert_loop": "Loop while active or ping with cooldown.",
    "alert_freeze_s": "Cooldown between pings (when looping is off).",
    "alert_zone_inside": "Alert for inside (1) or outside (0) of zone.",

    "hud_scale": "Bottom-right HUD scale in percent.",
    "snapshot_on_events": "Save snapshot on every line/zone event.",

    # Heatmap
    "heat_enabled": "Create a heatmap by accumulating detections.",
    "heat_overlay_on_start": "Show heatmap overlay immediately on run start. Toggle with the 'm' key during the run.",
    "heat_use_aoi": "Restrict accumulation to your drawn zones (AOI).",
    "heat_alpha": "Overlay opacity on the video (visual intensity).",
    "heat_sigma": "Blob size per detection (px).",
    "heat_window_enabled": "Use a rolling time window instead of infinite accumulation.",
    "heat_window_minutes": "Length of the rolling window, in minutes.",
    "heat_decay": "Alternative to window: per-frame decay in [0..1]. Higher = faster fade.",
    "heat_save_interval_s": "Save heatmap every N seconds (0 = off).",
    "heat_gamma": "Contrast of the overlay (0.5–2.0). Higher = punchier hotspots.",
    "heat_zero_thresh": "Values ≤ threshold are 100% transparent (true no-data).",
    "heat_gain": "Per-detection accumulation multiplier. >1 = build faster; <1 = slower.",
    "heat_memory_mult": "Multiplies the rolling window length. >1 = longer memory (slower fade).",
}

def _bind_help(widget: tk.Widget, key: str, help_label: tk.Label):
    def on(_e=None): help_label.config(text=HELP.get(key, ""))
    widget.bind("<FocusIn>", on); widget.bind("<Enter>", on)

# ───────────────────────────────── main builder ──────────────────────────────
def build_advanced_settings(parent: tk.Misc, app) -> ttk.Frame:
    if not hasattr(app, "adv_params"): app.adv_params = {}
    p = app.adv_params

    try:
        style = ttk.Style(parent)
        style.configure("Adv.TNotebook.Tab", padding=(16, 8))
        style.configure("Adv.TNotebook", tabmargins=(8, 4, 8, 0))
    except Exception:
        style = None

    # defaults snapshot (preserve existing values)
    defaults = {
        "imgsz": p.get("imgsz", 320), "conf": p.get("conf", 0.5), "iou": p.get("iou", 0.6),
        "frame_skip": p.get("frame_skip", 2), "track_buffer": p.get("track_buffer", 5),
        "match_thresh": p.get("match_thresh", 0.8), "min_hits": p.get("min_hits", 2),
        "line_min_gap": p.get("line_min_gap", 8), "line_min_sep": p.get("line_min_sep", 12),
        "zone_min_gap": p.get("zone_min_gap", 6),

        "live_preview": p.get("live_preview", True),
        "trace_enabled": p.get("trace_enabled", True), "trace_len": p.get("trace_len", 24),
        "overlay_mode": p.get("overlay_mode", "centroid"),
        "anchor_mode": p.get("anchor_mode", "center"), "ghost_margin": p.get("ghost_margin", 24),
        "trace_color": p.get("trace_color", "auto"), "trace_thickness": p.get("trace_thickness", 2),
        "overlay_frame_color": p.get("overlay_frame_color", "auto"),
        "overlay_frame_thickness": p.get("overlay_frame_thickness", 2),

        "alert_enabled": p.get("alert_enabled", False), "alert_sound": p.get("alert_sound", ""),
        "alert_loop": p.get("alert_loop", True), "alert_freeze_s": p.get("alert_freeze_s", 2),
        "alert_zone_inside": p.get("alert_zone_inside", 1),

        "hud_scale": float(p.get("hud_scale", 1.0)),
        "snapshot_on_events": bool(p.get("snapshot_on_events", False)),

        # Heatmap
        "heat_enabled": p.get("heat_enabled", False),
        "heat_overlay_on_start": p.get("heat_overlay_on_start", False),
        "heat_use_aoi": p.get("heat_use_aoi", False),
        "heat_alpha": p.get("heat_alpha", 0.5),
        "heat_sigma": p.get("heat_sigma", 8),
        "heat_window_enabled": p.get("heat_window_enabled", False),
        "heat_window_minutes": p.get("heat_window_minutes", 5.0),
        "heat_decay": p.get("heat_decay", 0.02),
        "heat_save_interval_s": p.get("heat_save_interval_s", 0),
        "heat_gamma": p.get("heat_gamma", 1.0),
        "heat_zero_thresh": p.get("heat_zero_thresh", 1e-6),
        # NEW
        "heat_gain": p.get("heat_gain", 1.0),
        "heat_memory_mult": p.get("heat_memory_mult", 1.0),
    }

    # tk variables (reuse if already on app)
    def _ensure(name, cls, value):
        if hasattr(app, name) and isinstance(getattr(app, name), cls):
            return getattr(app, name)
        v = cls(value=value); setattr(app, name, v); return v

    # Detection / tracking / hysteresis
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

    # Overlay / trace / anchor
    v_live_preview = _ensure("preview_enabled", tk.BooleanVar, bool(defaults["live_preview"]))
    app.overlay_mode = _ensure("overlay_mode", tk.StringVar, str(defaults["overlay_mode"]))
    app.anchor_mode  = _ensure("anchor_mode",  tk.StringVar, str(defaults["anchor_mode"]))
    app.ghost_margin = _ensure("ghost_margin", tk.StringVar, str(defaults["ghost_margin"]))
    app.trace_enabled= _ensure("trace_enabled", tk.BooleanVar, bool(defaults["trace_enabled"]))
    app.trace_len    = _ensure("trace_len", tk.StringVar, str(defaults["trace_len"]))

    v_trace_color   = _ensure("trace_color", tk.StringVar, str(defaults["trace_color"]))
    v_trace_thick   = _ensure("trace_thickness", tk.StringVar, str(defaults["trace_thickness"]))
    v_frame_color   = _ensure("frame_color", tk.StringVar, str(defaults["overlay_frame_color"]))
    v_frame_thick   = _ensure("frame_thickness", tk.StringVar, str(defaults["overlay_frame_thickness"]))

    # Alerts
    app.alert_enabled  = _ensure("alert_enabled", tk.BooleanVar, bool(defaults["alert_enabled"]))
    app.alert_sound    = _ensure("alert_sound", tk.StringVar, str(defaults["alert_sound"]))
    app.alert_loop     = _ensure("alert_loop", tk.BooleanVar, bool(defaults["alert_loop"]))
    app.alert_freeze_s = _ensure("alert_freeze_s", tk.IntVar, int(defaults["alert_freeze_s"]))
    v_alert_inside     = _ensure("alert_inside", tk.IntVar, int(defaults["alert_zone_inside"]))

    # Extras
    v_hud_scale       = _ensure("v_hud_scale", tk.StringVar, str(int(round(float(defaults["hud_scale"]) * 100))))
    v_snapshot_events = _ensure("snapshot_on_events", tk.BooleanVar, bool(defaults["snapshot_on_events"]))

    # Heatmap vars
    v_heat_enabled          = tk.IntVar(value=1 if defaults["heat_enabled"] else 0)  # radio 0/1
    v_heat_overlay          = tk.BooleanVar(value=bool(defaults["heat_overlay_on_start"]))
    v_heat_use_aoi          = tk.BooleanVar(value=bool(defaults["heat_use_aoi"]))
    v_heat_alpha            = tk.DoubleVar(value=float(defaults["heat_alpha"]))
    v_heat_gamma            = tk.DoubleVar(value=float(defaults["heat_gamma"]))
    v_heat_zero_thresh      = tk.DoubleVar(value=float(defaults["heat_zero_thresh"]))
    v_heat_sigma            = tk.IntVar(value=int(defaults["heat_sigma"]))
    v_heat_window_enabled   = tk.BooleanVar(value=bool(defaults["heat_window_enabled"]))
    v_heat_window_minutes   = tk.DoubleVar(value=float(defaults["heat_window_minutes"]))
    v_heat_decay            = tk.DoubleVar(value=float(defaults["heat_decay"]))
    v_heat_save_interval_s  = tk.IntVar(value=int(defaults["heat_save_interval_s"]))
    # NEW
    v_heat_gain             = tk.DoubleVar(value=float(defaults["heat_gain"]))
    v_heat_memory_mult      = tk.DoubleVar(value=float(defaults["heat_memory_mult"]))

    # ─────────────────────────── layout ───────────────────────────
    root = ttk.Frame(parent); root.pack(fill="both", expand=True)
    nb = ttk.Notebook(root, style=("Adv.TNotebook" if style else "TNotebook")); nb.pack(side="top", fill="both", expand=True)

    # MAIN tab
    main_tab = ttk.Frame(nb); nb.add(main_tab, text="Main")
    main = _make_scrollable(main_tab)
    left = ttk.Frame(main); left.pack(side="left", fill="both", expand=True)
    right, help_lbl = _make_help_panel(main, title="Help", width=HELP_WIDTH); right.pack(side="left", fill="y", padx=(8,0))

    lf = ttk.LabelFrame(left, text="Detection / Tracking / Hysteresis"); lf.pack(fill="x", padx=6, pady=(6,4))
    for key, var in (("imgsz", v_imgsz), ("conf", v_conf), ("iou", v_iou),
                     ("frame_skip", v_frame_skip), ("track_buffer", v_track_buffer),
                     ("match_thresh", v_match_thresh), ("min_hits", v_min_hits),
                     ("line_min_gap_frames", v_line_gap), ("line_min_sep_px", v_line_sep),
                     ("zone_min_gap_frames", v_zone_gap)):
        _, e = _row(lf, key, lambda fr, v=var: ttk.Entry(fr, textvariable=v, width=10)); _bind_help(e, key.split("_frames")[0].split("_px")[0], help_lbl)

    lf = ttk.LabelFrame(left, text="Overlay / Trace / Anchor / Ghost"); lf.pack(fill="x", padx=6, pady=(4,4))
    _, cb = _row(lf, "", lambda fr: ttk.Checkbutton(fr, text="Enable LIVE preview", variable=v_live_preview)); _bind_help(cb, "live_preview", help_lbl)
    def _overlay_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Overlay:").pack(side="left")
        cmb = ttk.Combobox(box, textvariable=app.overlay_mode, state="readonly", width=12, values=["centroid", "box", "box+conf"])
        cmb.pack(side="left", padx=(6,0)); _bind_help(cmb, "overlay_mode", help_lbl)
        return box
    _row(lf, "", _overlay_row)
    def _trace_row(fr):
        box = ttk.Frame(fr)
        c = ttk.Checkbutton(box, text="Trace", variable=app.trace_enabled); c.pack(side="left"); _bind_help(c, "trace_enabled", help_lbl)
        ttk.Label(box, text="len:").pack(side="left", padx=(8,2))
        sp = ttk.Spinbox(box, from_=1, to=240, textvariable=app.trace_len, width=5); sp.pack(side="left"); _bind_help(sp, "trace_len", help_lbl)
        return box
    _row(lf, "", _trace_row)
    def _anchor_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Anchor:").pack(side="left")
        om = ttk.OptionMenu(box, app.anchor_mode, app.anchor_mode.get(), "center", "bottom"); om.pack(side="left", padx=(4,12))
        _bind_help(om, "anchor_mode", help_lbl)
        ttk.Label(box, text="Ghost margin (px):").pack(side="left")
        sp = ttk.Spinbox(box, from_=0, to=256, textvariable=app.ghost_margin, width=6); sp.pack(side="left"); _bind_help(sp, "ghost_margin", help_lbl)
        return box
    _row(lf, "", _anchor_row)

    lf = ttk.LabelFrame(left, text="Colors/Thickness"); lf.pack(fill="x", padx=6, pady=(4,6))
    for key, var, width in (("Trace color (auto/#RRGGBB/B,G,R)", v_trace_color, 14),
                            ("Trace thickness (px)", v_trace_thick, 6),
                            ("Frame color (auto/#RRGGBB/B,G,R)", v_frame_color, 14),
                            ("Frame thickness (px)", v_frame_thick, 6)):
        def _make(fr, v=var, w=width): return ttk.Entry(fr, textvariable=v, width=w)
        label = " ".join(key.split()[:-1]) if key.endswith("(px)") else key
        _, e = _row(lf, label, _make)
        _bind_help(e, ("trace_color" if "Trace color" in key else
                       "trace_thickness" if "thickness" in key and "Trace" in key else
                       "overlay_frame_color" if "Frame color" in key else "overlay_frame_thickness"), help_lbl)

    # Alerts
    lf = ttk.LabelFrame(left, text="Sound alert (zones)"); lf.pack(fill="x", padx=6, pady=(0,6))
    _, cb = _row(lf, "", lambda fr: ttk.Checkbutton(fr, text="Enable alert", variable=app.alert_enabled)); _bind_help(cb, "alert_enabled", help_lbl)
    def _sound_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Sound file:").pack(side="left")
        e = ttk.Entry(box, textvariable=app.alert_sound, width=42); e.pack(side="left", padx=(6,6), fill="x", expand=True)
        ttk.Button(box, text="Browse…", command=lambda: app.alert_sound.set(filedialog.askopenfilename(initialdir=str(_dir('sounds'))))).pack(side="left")
        _bind_help(e, "alert_sound", help_lbl)
        return box
    _row(lf, "", _sound_row)
    def _loop_row(fr):
        box = ttk.Frame(fr)
        cb = ttk.Checkbutton(box, text="Loop while active", variable=app.alert_loop); cb.pack(side="left"); _bind_help(cb, "alert_loop", help_lbl)
        ttk.Label(box, text="freeze (s):").pack(side="left", padx=(10,2))
        sp = ttk.Spinbox(box, from_=0, to=30, textvariable=app.alert_freeze_s, width=6); sp.pack(side="left"); _bind_help(sp, "alert_freeze_s", help_lbl)
        return box
    _row(lf, "", _loop_row)
    def _mode_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Mode:").pack(side="left")
        rb1 = ttk.Radiobutton(box, text="inside zone", value=1, variable=v_alert_inside); rb1.pack(side="left", padx=(6,6))
        rb2 = ttk.Radiobutton(box, text="outside zone", value=0, variable=v_alert_inside); rb2.pack(side="left", padx=(6,6))
        _bind_help(rb1, "alert_zone_inside", help_lbl); _bind_help(rb2, "alert_zone_inside", help_lbl)
        return box
    _row(lf, "", _mode_row)

    # EXTRAS tab
    extras_tab = ttk.Frame(nb); nb.add(extras_tab, text="Extras")
    extras = _make_scrollable(extras_tab)
    ex_left = ttk.Frame(extras); ex_left.pack(side="left", fill="both", expand=True)
    ex_right, ex_help = _make_help_panel(extras, title="Help", width=HELP_WIDTH); ex_right.pack(side="left", fill="y", padx=(8,0))

    ex_hud = ttk.LabelFrame(ex_left, text="HUD"); ex_hud.pack(fill="x", padx=6, pady=(6,4))
    _, e = _row(ex_hud, "HUD size (%)", lambda fr: ttk.Spinbox(fr, from_=50, to=200, increment=5, textvariable=v_hud_scale, width=6)); _bind_help(e, "hud_scale", ex_help)

    ex_snap = ttk.LabelFrame(ex_left, text="Snapshots"); ex_snap.pack(fill="x", padx=6, pady=(4,4))
    _, cb = _row(ex_snap, "", lambda fr: ttk.Checkbutton(fr, text="Save snapshot on events", variable=v_snapshot_events)); _bind_help(cb, "snapshot_on_events", ex_help)

    # ───────────── Heatmap ─────────────
    ex_heat = ttk.LabelFrame(ex_left, text="Heatmap"); ex_heat.pack(fill="x", padx=6, pady=(4,6))

    def _onoff(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Create heatmap:").pack(side="left")
        rb0 = ttk.Radiobutton(box, text="Off", value=0, variable=v_heat_enabled); rb0.pack(side="left", padx=(8,2))
        rb1 = ttk.Radiobutton(box, text="On",  value=1, variable=v_heat_enabled); rb1.pack(side="left", padx=(2,0))
        _bind_help(rb0, "heat_enabled", ex_help); _bind_help(rb1, "heat_enabled", ex_help)
        return box
    _row(ex_heat, "", _onoff)

    _, cb = _row(ex_heat, "", lambda fr: ttk.Checkbutton(fr, text="Show overlay on start (press 'm' to toggle)", variable=v_heat_overlay))
    _bind_help(cb, "heat_overlay_on_start", ex_help)

    _, cb = _row(ex_heat, "", lambda fr: ttk.Checkbutton(fr, text="Restrict to zones (AOI mask)", variable=v_heat_use_aoi))
    _bind_help(cb, "heat_use_aoi", ex_help)

    def _alpha_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Intensity (alpha 0–1):").pack(side="left")
        sc = ttk.Scale(box, from_=0.0, to=1.0, variable=v_heat_alpha, orient="horizontal", length=180); sc.pack(side="left", padx=(6,0))
        _bind_help(sc, "heat_alpha", ex_help)
        return box
    _row(ex_heat, "", _alpha_row)

    def _gamma_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Contrast (gamma 0.5–2.0):").pack(side="left")
        sc = ttk.Scale(box, from_=0.5, to=2.0, variable=v_heat_gamma, orient="horizontal", length=180); sc.pack(side="left", padx=(6,0))
        _bind_help(sc, "heat_gamma", ex_help)
        return box
    _row(ex_heat, "", _gamma_row)

    def _thr_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="No-data threshold (0–0.02):").pack(side="left")
        sc = ttk.Scale(box, from_=0.0, to=0.02, variable=v_heat_zero_thresh, orient="horizontal", length=180); sc.pack(side="left", padx=(6,0))
        _bind_help(sc, "heat_zero_thresh", ex_help)
        return box
    _row(ex_heat, "", _thr_row)

    def _gain_row(fr):  # NEW
        box = ttk.Frame(fr)
        ttk.Label(box, text="Accumulation gain (0.1–5.0):").pack(side="left")
        sc = ttk.Scale(box, from_=0.1, to=5.0, variable=v_heat_gain, orient="horizontal", length=180); sc.pack(side="left", padx=(6,0))
        _bind_help(sc, "heat_gain", ex_help)
        return box
    _row(ex_heat, "", _gain_row)

    def _memory_row(fr):  # NEW
        box = ttk.Frame(fr)
        ttk.Label(box, text="Memory (fade) × (0.25–4):").pack(side="left")
        sc = ttk.Scale(box, from_=0.25, to=4.0, variable=v_heat_memory_mult, orient="horizontal", length=180); sc.pack(side="left", padx=(6,0))
        _bind_help(sc, "heat_memory_mult", ex_help)
        return box
    _row(ex_heat, "", _memory_row)

    def _sigma_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Sigma (px):").pack(side="left")
        sp = ttk.Spinbox(box, from_=1, to=64, textvariable=v_heat_sigma, width=6); sp.pack(side="left", padx=(6,0))
        _bind_help(sp, "heat_sigma", ex_help)
        return box
    _row(ex_heat, "", _sigma_row)

    ttk.Separator(ex_heat, orient="horizontal").pack(fill="x", padx=4, pady=6)

    def _window_row(fr):
        box = ttk.Frame(fr)
        cb = ttk.Checkbutton(box, text="Use rolling window (minutes)", variable=v_heat_window_enabled); cb.pack(side="left")
        _bind_help(cb, "heat_window_enabled", ex_help)
        ttk.Label(box, text="Length:").pack(side="left", padx=(12,4))
        ent = ttk.Entry(box, textvariable=v_heat_window_minutes, width=8); ent.pack(side="left")
        _bind_help(ent, "heat_window_minutes", ex_help)
        return box
    _row(ex_heat, "", _window_row)

    def _decay_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="OR per-frame decay (0–1):").pack(side="left")
        ent = ttk.Entry(box, textvariable=v_heat_decay, width=8); ent.pack(side="left", padx=(6,0))
        _bind_help(ent, "heat_decay", ex_help)
        return box
    _row(ex_heat, "", _decay_row)

    def _save_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Save interval (s, 0=off):").pack(side="left")
        ent = ttk.Entry(box, textvariable=v_heat_save_interval_s, width=8); ent.pack(side="left", padx=(6,0))
        _bind_help(ent, "heat_save_interval_s", ex_help)
        return box
    _row(ex_heat, "", _save_row)

    # ───────────────────── bottom bar ─────────────────────
    def _collect() -> dict:
        return {
            # main
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

            "overlay_mode": str(app.overlay_mode.get()),
            "anchor_mode": _str(app.anchor_mode, defaults["anchor_mode"]),
            "ghost_margin": _int(app.ghost_margin, defaults["ghost_margin"]),
            "trace_enabled": bool(app.trace_enabled.get()),
            "trace_len": _int(app.trace_len, defaults["trace_len"]),

            "trace_color": _str(v_trace_color, defaults["trace_color"]),
            "trace_thickness": _int(v_trace_thick, defaults["trace_thickness"]),
            "overlay_frame_color": _str(v_frame_color, defaults["overlay_frame_color"]),
            "overlay_frame_thickness": _int(v_frame_thick, defaults["overlay_frame_thickness"]),

            "alert_enabled": bool(app.alert_enabled.get()),
            "alert_sound": _str(app.alert_sound, defaults["alert_sound"]),
            "alert_loop": bool(app.alert_loop.get()),
            "alert_freeze_s": _int(app.alert_freeze_s, defaults["alert_freeze_s"]),
            "alert_zone_inside": int(v_alert_inside.get()),

            "hud_scale": max(0.5, min(2.0, _int(v_hud_scale, 100)/100.0)),
            "snapshot_on_events": bool(v_snapshot_events.get()),

            # heatmap
            "heat_enabled": bool(int(v_heat_enabled.get()) == 1),
            "heat_overlay_on_start": bool(v_heat_overlay.get()),
            "heat_use_aoi": bool(v_heat_use_aoi.get()),
            "heat_alpha": float(v_heat_alpha.get()),
            "heat_gamma": float(v_heat_gamma.get()),
            "heat_zero_thresh": float(v_heat_zero_thresh.get()),
            "heat_sigma": int(v_heat_sigma.get()),
            "heat_window_enabled": bool(v_heat_window_enabled.get()),
            "heat_window_minutes": float(v_heat_window_minutes.get()),
            "heat_decay": float(v_heat_decay.get()),
            "heat_save_interval_s": int(v_heat_save_interval_s.get()),
            # NEW
            "heat_gain": float(v_heat_gain.get()),
            "heat_memory_mult": float(v_heat_memory_mult.get()),
            "_meta": {"version": 9, "saved_at": datetime.now().isoformat(timespec="seconds")},
        }

    bar = ttk.Frame(root); bar.pack(side="bottom", fill="x", padx=8, pady=8)
    ttk.Button(bar, text="Apply", command=lambda: app.adv_params.update(_collect())).pack(side="left")

    def _save_preset():
        data = _collect()
        path = filedialog.asksaveasfilename(
            title="Save preset",
            initialdir=str(_dir("presets")), defaultextension=".json",
            initialfile=f"adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")]
        )
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Preset", f"Could not save:\n{e}")

    def _load_preset():
        path = filedialog.askopenfilename(
            title="Load preset",
            initialdir=str(_dir("presets")),
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")]
        )
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: d = json.load(f)
        except Exception as e:
            messagebox.showerror("Preset", f"Could not read file:\n{e}"); return

        def G(name, default):
            return d.get(name, default)

        # main
        v_imgsz.set(str(G("imgsz", defaults["imgsz"])))
        v_conf.set(str(G("conf", defaults["conf"])))
        v_iou.set(str(G("iou", defaults["iou"])))
        v_frame_skip.set(str(G("frame_skip", defaults["frame_skip"])))
        v_track_buffer.set(str(G("track_buffer", defaults["track_buffer"])))
        v_match_thresh.set(str(G("match_thresh", defaults["match_thresh"])))
        v_min_hits.set(str(G("min_hits", defaults["min_hits"])))
        v_line_gap.set(str(G("line_min_gap", defaults["line_min_gap"])))
        v_line_sep.set(str(G("line_min_sep", defaults["line_min_sep"])))
        v_zone_gap.set(str(G("zone_min_gap", defaults["zone_min_gap"])))

        app.overlay_mode.set(str(G("overlay_mode", defaults["overlay_mode"])))
        app.anchor_mode.set(str(G("anchor_mode", defaults["anchor_mode"])))
        app.ghost_margin.set(str(G("ghost_margin", defaults["ghost_margin"])))
        app.trace_enabled.set(bool(G("trace_enabled", defaults["trace_enabled"])))
        app.trace_len.set(str(G("trace_len", defaults["trace_len"])))
        v_trace_color.set(str(G("trace_color", defaults["trace_color"])))
        v_trace_thick.set(str(G("trace_thickness", defaults["trace_thickness"])))
        v_frame_color.set(str(G("overlay_frame_color", defaults["overlay_frame_color"])))
        v_frame_thick.set(str(G("overlay_frame_thickness", defaults["overlay_frame_thickness"])))

        app.alert_enabled.set(bool(G("alert_enabled", defaults["alert_enabled"])))
        app.alert_sound.set(str(G("alert_sound", defaults["alert_sound"])))
        app.alert_loop.set(bool(G("alert_loop", defaults["alert_loop"])))
        app.alert_freeze_s.set(int(G("alert_freeze_s", defaults["alert_freeze_s"])))
        v_alert_inside.set(int(G("alert_zone_inside", defaults["alert_zone_inside"])))

        v_hud_scale.set(str(int(round(float(G("hud_scale", defaults["hud_scale"])) * 100))))
        v_snapshot_events.set(bool(G("snapshot_on_events", defaults["snapshot_on_events"])))

        # heat
        v_heat_enabled.set(1 if G("heat_enabled", defaults["heat_enabled"]) else 0)
        v_heat_overlay.set(bool(G("heat_overlay_on_start", defaults["heat_overlay_on_start"])))
        v_heat_use_aoi.set(bool(G("heat_use_aoi", defaults["heat_use_aoi"])))
        v_heat_alpha.set(float(G("heat_alpha", defaults["heat_alpha"])))
        v_heat_gamma.set(float(G("heat_gamma", defaults["heat_gamma"])))
        v_heat_zero_thresh.set(float(G("heat_zero_thresh", defaults["heat_zero_thresh"])))
        v_heat_sigma.set(int(G("heat_sigma", defaults["heat_sigma"])))
        v_heat_window_enabled.set(bool(G("heat_window_enabled", defaults["heat_window_enabled"])))
        v_heat_window_minutes.set(float(G("heat_window_minutes", defaults["heat_window_minutes"])))
        v_heat_decay.set(float(G("heat_decay", defaults["heat_decay"])))
        v_heat_save_interval_s.set(int(G("heat_save_interval_s", defaults["heat_save_interval_s"])))
        # NEW
        v_heat_gain.set(float(G("heat_gain", defaults["heat_gain"])))
        v_heat_memory_mult.set(float(G("heat_memory_mult", defaults["heat_memory_mult"])))

        app.adv_params.update(_collect())

    ttk.Button(bar, text="Save preset…", command=_save_preset).pack(side="left", padx=(8,0))
    ttk.Button(bar, text="Load preset…", command=_load_preset).pack(side="left", padx=(8,0))
    ttk.Button(bar, text="Close", command=parent.winfo_toplevel().destroy).pack(side="right")

    return root

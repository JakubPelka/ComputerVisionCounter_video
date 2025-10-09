# cv_video_advanced_ui.py — Advanced settings UI (fixed help width, grouped extras, tab styling)
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
from datetime import datetime

# use project sound player for consistent behavior
try:
    from cv_video_sound import SoundPlayer
except Exception:
    SoundPlayer = None  # guarded use

# ───────────────────── constants ─────────────────────
HELP_WIDTH = 420   # fixed help panel width (prevents left/right "jumping")

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
    fr = ttk.Frame(parent)
    ttk.Label(fr, text=label_text, width=22, anchor="w").pack(side="left")
    w = build_widget(fr)
    try:
        if isinstance(w, tk.Widget): w.pack(side="left")
    except Exception:
        pass
    fr.pack(fill="x", padx=6, pady=(2, 2))
    return fr, w

def _browse_sound(var):
    path = filedialog.askopenfilename(
        title="Choose alert sound",
        initialdir=str(_sounds_dir()),
        filetypes=[("Audio", "*.wav;*.mp3;*.ogg;*.flac;*.aac;*.m4a"), ("All files", "*.*")]
    )
    if path: var.set(path)

def _make_scrollable(parent: tk.Misc) -> ttk.Frame:
    """Return an interior frame that is vertically scrollable."""
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

    # mouse wheel
    def _on_wheel(e):
        canvas.yview_scroll(int(-1*(e.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_wheel)

    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    container.pack(fill="both", expand=True)
    return interior

def _make_help_panel(parent: tk.Misc, title="Help", width=HELP_WIDTH):
    """Fixed-width help panel with wrapped label (no size jitter)."""
    fr = ttk.LabelFrame(parent, text=title)
    fr.pack_propagate(False)  # keep fixed width
    fr.config(width=width)
    lbl = tk.Label(fr, text="Focus a field to see help.", justify="left", anchor="nw")
    lbl.pack(fill="both", expand=True, padx=8, pady=8)
    # fixed wrap to the help frame width
    lbl.config(wraplength=max(50, width - 16))
    return fr, lbl

# ───────────────────── contextual help ─────────────────────
HELP = {
    # Detection / tracking
    "imgsz": "Image size for the AI. Bigger = better detail but slower. Try 320–640 on CPU; 960–1280 only for tiny/far objects.",
    "conf": "Confidence threshold. Higher = stricter (fewer false alarms, more misses). Start around 0.50–0.60.",
    "iou": "Overlap used to merge duplicate boxes. Higher merges more; lower keeps more boxes.",
    "frame_skip": "Process every Nth frame. 0 = every frame (best quality, slowest). +1 or +2 speeds up but can miss fast motion.",
    "track_buffer": "How long (frames) an ID is kept after it disappears. Higher survives short occlusions; too high may leave ghosts.",
    "match_thresh": "How strict the tracker is when matching boxes to existing IDs. Higher = stricter (fewer mismatches, more resets).",
    "min_hits": "Frames required before a new track is confirmed. Flicker? raise. Delayed IDs? lower.",
    "line_min_gap": "Min frames between two line counts for the same object (anti-bounce).",
    "line_min_sep": "Min movement across the line (px) before counting again. Sliding along the line? raise this.",
    "zone_min_gap": "Min frames between zone IN/OUT for the same object. Border toggling? raise this.",

    # Overlay / trace / anchor
    "live_preview": "Show the processed video while running. Turn off to save CPU.",
    "overlay_mode": "centroid = dots; box = rectangles; box+conf = rectangles with score.",
    "trace_enabled": "Draw a tail behind each tracked object.",
    "trace_len": "How many recent points to keep per tail.",
    "anchor_mode": "Point used for counting & trails: bottom (feet/wheels) or center.",
    "ghost_margin": "Only for bottom anchor: lift the point above the box edge to avoid border/line bounces.",

    # Colors / thickness
    "trace_color": "Use 'auto', a #RRGGBB value (e.g. #00FF88), or B,G,R (e.g. 255,200,0).",
    "trace_thickness": "Tail thickness in pixels.",
    "overlay_frame_color": "Color for lines & zone outlines. Same formats as trace color.",
    "overlay_frame_thickness": "Line/zone thickness (pixels).",

    # Alert
    "alert_enabled": "Play a sound when the condition below is met for selected classes.",
    "alert_sound": "Audio file to play. Use Play to test.",
    "alert_loop": "Loop while condition holds (on) or play single pings with cooldown (off).",
    "alert_freeze_s": "Cooldown between pings (only when looping is off).",
    "alert_zone_inside": "Alert when an object is inside the zone (1) or outside (0).",

    # Extras
    "hud_scale": "Size of the bottom-right HUD panel. 100% = default.",
    "snapshot_on_events": "Save an annotated snapshot on every line/zone event.",
}

def _bind_help(widget: tk.Widget, key: str, help_label: tk.Label):
    def on(_e=None): help_label.config(text=HELP.get(key, ""))
    widget.bind("<FocusIn>", on)
    widget.bind("<Enter>", on)

# ───────────────────── main builder ─────────────────────
def build_advanced_settings(parent: tk.Misc, app) -> ttk.Frame:
    """
    Layout:
      • Notebook with Main + Extras (both have fixed-width Help panel)
      • Bottom action bar (Apply / Save / Load / Close) always visible
    """
    if not hasattr(app, "adv_params"):
        app.adv_params = {}
    p = app.adv_params

    # ttk style: make tabs a bit taller/more visible
    try:
        style = ttk.Style(parent)
        style.configure("Adv.TNotebook.Tab", padding=(16, 8))  # bigger tabs
        style.configure("Adv.TNotebook", tabmargins=(8, 4, 8, 0))
    except Exception:
        style = None

    # defaults snapshot
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
    }

    # tk variables (re-use if present)
    def _ensure(name, cls, value):
        if hasattr(app, name) and isinstance(getattr(app, name), cls):
            return getattr(app, name)
        v = cls(value=value); setattr(app, name, v); return v

    # Detection / Tracking / Hysteresis vars
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

    # Overlay / Trace / Anchor / Ghost
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

    # ────────────────── outer container: notebook + bottom bar ──────────────────
    root = ttk.Frame(parent); root.pack(fill="both", expand=True)
    nb_style = "Adv.TNotebook" if style else "TNotebook"
    nb = ttk.Notebook(root, style=nb_style); nb.pack(side="top", fill="both", expand=True)

    # === MAIN TAB (scrollable) ===
    main_tab = ttk.Frame(nb); nb.add(main_tab, text="Main")
    main = _make_scrollable(main_tab)

    left = ttk.Frame(main);  left.pack(side="left", fill="both", expand=True)
    right, help_lbl = _make_help_panel(main, title="Help", width=HELP_WIDTH)
    right.pack(side="left", fill="y", padx=(8, 0))  # fixed-width help panel

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

    def _overlay_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Overlay mode:").pack(side="left")
        cmb = ttk.Combobox(box, textvariable=app.overlay_mode, width=12, state="readonly",
                           values=["centroid", "box", "box+conf"])
        cmb.pack(side="left", padx=(6, 0))
        _bind_help(cmb, "overlay_mode", help_lbl)
        return box
    _row(lf, "", _overlay_row)

    def _trace_row(fr):
        box = ttk.Frame(fr)
        c = ttk.Checkbutton(box, text="Trace", variable=app.trace_enabled); c.pack(side="left")
        _bind_help(c, "trace_enabled", help_lbl)
        ttk.Label(box, text="len:").pack(side="left", padx=(8, 2))
        sp = ttk.Spinbox(box, from_=1, to=240, textvariable=app.trace_len, width=5); sp.pack(side="left")
        _bind_help(sp, "trace_len", help_lbl)
        return box
    _row(lf, "", _trace_row)

    def _anchor_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Anchor:").pack(side="left")
        om = ttk.OptionMenu(box, app.anchor_mode, app.anchor_mode.get(), "center", "bottom")
        om.pack(side="left", padx=(4, 12))
        _bind_help(om, "anchor_mode", help_lbl)
        ttk.Label(box, text="Ghost margin (px):").pack(side="left")
        sp = ttk.Spinbox(box, from_=0, to=256, textvariable=app.ghost_margin, width=6)
        sp.pack(side="left"); _bind_help(sp, "ghost_margin", help_lbl)
        return box
    _row(lf, "", _anchor_row)

    # Colors/Thickness (overlay)
    lf = ttk.LabelFrame(left, text="Colors/Thickness (overlay)"); lf.pack(fill="x", padx=6, pady=(4, 4))
    def _trace_color_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Trace color (auto/#RRGGBB/B,G,R)").pack(side="left")
        e = ttk.Entry(box, textvariable=v_trace_color, width=14); e.pack(side="left", padx=(6, 6))
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
    _, cb = _row(lf, "", lambda fr: ttk.Checkbutton(fr, text="Enable alert (uses selected classes)", variable=app.alert_enabled))
    _bind_help(cb, "alert_enabled", help_lbl)

    sound_player_ref = {"player": None}
    def _play_sound():
        if not SoundPlayer:
            messagebox.showwarning("Sound", "Sound player not available in this build."); return
        path = (app.alert_sound.get() or "").strip()
        if not path:
            messagebox.showwarning("Sound", "Please choose a sound file first."); return
        try:
            sound_player_ref["player"] = SoundPlayer(path)
            sound_player_ref["player"].play_once()
        except Exception as e:
            messagebox.showerror("Sound", f"Could not play:\n{e}")
    def _stop_sound():
        try:
            if sound_player_ref["player"]: sound_player_ref["player"].stop()
        except Exception: pass

    def _sound_row(fr):
        box = ttk.Frame(fr)
        ttk.Label(box, text="Sound file:").pack(side="left")
        e = ttk.Entry(box, textvariable=app.alert_sound, width=42); e.pack(side="left", padx=(6, 6), fill="x", expand=True)
        ttk.Button(box, text="Browse…", command=lambda: _browse_sound(app.alert_sound)).pack(side="left", padx=(0, 6))
        ttk.Button(box, text="Play", command=_play_sound).pack(side="left")
        ttk.Button(box, text="Stop", command=_stop_sound).pack(side="left", padx=(4, 0))
        _bind_help(e, "alert_sound", help_lbl)
        return box
    _row(lf, "", _sound_row)

    def _loop_row(fr):
        box = ttk.Frame(fr)
        cb = ttk.Checkbutton(box, text="Loop while active", variable=app.alert_loop); cb.pack(side="left")
        _bind_help(cb, "alert_loop", help_lbl)
        ttk.Label(box, text="freeze (s):").pack(side="left", padx=(10, 2))
        sp = ttk.Spinbox(box, from_=0, to=30, textvariable=app.alert_freeze_s, width=6); sp.pack(side="left")
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

    # === EXTRAS TAB (scrollable + help) ===
    extras_tab = ttk.Frame(nb); nb.add(extras_tab, text="Extras")
    extras = _make_scrollable(extras_tab)

    ex_left = ttk.Frame(extras); ex_left.pack(side="left", fill="both", expand=True)
    ex_right, ex_help_lbl = _make_help_panel(extras, title="Help", width=HELP_WIDTH)
    ex_right.pack(side="left", fill="y", padx=(8, 0))

    # HUD group
    ex_hud = ttk.LabelFrame(ex_left, text="HUD"); ex_hud.pack(fill="x", padx=6, pady=(6, 4))
    def _hud_spin(fr):
        return ttk.Spinbox(fr, from_=50, to=200, increment=5, textvariable=v_hud_scale, width=6)
    _, e = _row(ex_hud, "HUD size (%)", _hud_spin); _bind_help(e, "hud_scale", ex_help_lbl)

    # Snapshots group
    ex_snap = ttk.LabelFrame(ex_left, text="Snapshots"); ex_snap.pack(fill="x", padx=6, pady=(4, 6))
    def _snap_row(fr):
        return ttk.Checkbutton(fr, text="Save snapshot on events", variable=v_snapshot_events)
    _, cb = _row(ex_snap, "", _snap_row); _bind_help(cb, "snapshot_on_events", ex_help_lbl)

    # ────────────────── bottom action bar (outside tabs) ──────────────────
    def _collect() -> dict:
        return {
            # Detection/Tracking/Hysteresis
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
            # Overlay/Trace/Anchor/Ghost
            "overlay_mode": str(app.overlay_mode.get()),
            "anchor_mode": _str(app.anchor_mode, defaults["anchor_mode"]),
            "ghost_margin": _int(app.ghost_margin, defaults["ghost_margin"]),
            "trace_enabled": bool(app.trace_enabled.get()),
            "trace_len": _int(app.trace_len, defaults["trace_len"]),
            # Colors/Thickness
            "trace_color": _str(v_trace_color, defaults["trace_color"]),
            "trace_thickness": _int(v_trace_thick, defaults["trace_thickness"]),
            "overlay_frame_color": _str(v_frame_color, defaults["overlay_frame_color"]),
            "overlay_frame_thickness": _int(v_frame_thick, defaults["overlay_frame_thickness"]),
            # Alerts
            "alert_enabled": bool(app.alert_enabled.get()),
            "alert_sound": _str(app.alert_sound, defaults["alert_sound"]),
            "alert_loop": bool(app.alert_loop.get()),
            "alert_freeze_s": _int(app.alert_freeze_s, defaults["alert_freeze_s"]),
            "alert_zone_inside": _int(v_alert_inside, defaults["alert_zone_inside"]),
            # Extras
            "hud_scale": max(0.5, min(2.0, _int(v_hud_scale, 100) / 100.0)),
            "snapshot_on_events": bool(v_snapshot_events.get()),
            "_meta": {"version": 6, "saved_at": datetime.now().isoformat(timespec="seconds")},
        }

    bar = ttk.Frame(root); bar.pack(side="bottom", fill="x", padx=8, pady=8)

    def _apply():
        app.adv_params.update(_collect())  # silent apply
    ttk.Button(bar, text="Apply", command=_apply).pack(side="left")

    def _save_preset():
        data = _collect()
        initialdir = str(_presets_dir())
        fname = f"adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(
            title="Save preset",
            initialdir=initialdir, defaultextension=".json", initialfile=fname,
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
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
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: d = json.load(f)
        except Exception as e:
            messagebox.showerror("Preset", f"Could not read file:\n{e}"); return

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

        app.overlay_mode.set(str(_getk("overlay_mode", default=defaults["overlay_mode"])))
        app.anchor_mode.set(str(_getk("anchor_mode", default=defaults["anchor_mode"])))
        app.ghost_margin.set(str(_getk("ghost_margin", default=defaults["ghost_margin"])))

        app.trace_enabled.set(bool(_getk("trace_enabled", default=defaults["trace_enabled"])))
        app.trace_len.set(str(_getk("trace_len", default=defaults["trace_len"])))
        v_trace_color.set(str(_getk("trace_color", default=defaults["trace_color"])))
        v_trace_thick.set(str(_getk("trace_thickness", default=defaults["trace_thickness"])))
        v_frame_color.set(str(_getk("overlay_frame_color", "frame_color", default=defaults["overlay_frame_color"])))
        v_frame_thick.set(str(_getk("overlay_frame_thickness", "frame_thickness", default=defaults["overlay_frame_thickness"])))

        app.alert_enabled.set(bool(_getk("alert_enabled", default=defaults["alert_enabled"])))
        app.alert_sound.set(str(_getk("alert_sound", default=defaults["alert_sound"])))
        app.alert_loop.set(bool(_getk("alert_loop", default=defaults["alert_loop"])))
        app.alert_freeze_s.set(int(_getk("alert_freeze_s", default=defaults["alert_freeze_s"])))
        v_alert_inside.set(int(_getk("alert_zone_inside", default=defaults["alert_zone_inside"])))

        v_hud_scale.set(str(int(round(float(_getk("hud_scale", default=defaults["hud_scale"])) * 100))))
        v_snapshot_events.set(bool(_getk("snapshot_on_events", default=defaults["snapshot_on_events"])))

        app.adv_params.update(_collect())  # silent apply

    ttk.Button(bar, text="Load preset…", command=_load_preset).pack(side="left", padx=(8, 0))

    def _close():
        try:
            if SoundPlayer and "player" in sound_player_ref and sound_player_ref["player"]:
                sound_player_ref["player"].stop()
        except Exception: pass
        parent.winfo_toplevel().destroy()
    ttk.Button(bar, text="Close", command=_close).pack(side="right")

    return root

# cv_video_gui.py
from __future__ import annotations
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

# we use this to discover a first frame automatically if the caller didn't pass one
from cv_video_core import SUPPORTED_VID_EXTS  # (.mp4, .mov, ...)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, height=280):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, height=height)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas)
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.win, width=e.width))
        self.inner.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta/120), "units")

class CounterEditor(tk.Toplevel):
    """
    Standard mode (files): shows a frozen first frame to draw on.
    LIVE mode (camera/URL): pass live_cap=VideoCapture — then:
      • preview is live,
      • when you choose “Add line/zone”, preview AUTO-PAUSES for easier drawing,
      • “Clear (resume LIVE)” clears sketches and resumes LIVE.
    """
    def __init__(self, master, frame_bgr: np.ndarray|None,
                 default_cfg_path: Path|None=None,
                 live_cap: cv2.VideoCapture|None=None):
        super().__init__(master)
        self.title("Counter configuration (LIVE/STATIC)")
        # make Cancel/ESC/X behave like ABORT
        self.protocol("WM_DELETE_WINDOW", self.abort)
        self.bind("<Escape>", self.abort)
        self.transient(master)
        try:
            self.grab_set()  # modal
        except Exception:
            pass

        self.lines: list[dict] = []
        self.zones: list[dict] = []
        self.mode = tk.StringVar(value="idle")
        self.cur_points: list[list[float]] = []
        self.default_cfg_path = default_cfg_path

        # cancel/return state
        self._aborted = False
        self.result: dict|None = None   # {"lines":[...], "zones":[...]}

        # LIVE state
        self.live_cap = live_cap
        self.live = bool(live_cap is not None)
        self._after_id = None
        self.base_bgr = None  # last frame for overlay (frozen while paused)
        self.disp_w = self.disp_h = 0
        self.scale = 1.0

        # If stream provided but no frame, try to read one
        if self.live_cap is not None and frame_bgr is None:
            ok, fr = self.live_cap.read()
            frame_bgr = fr if ok else None

        # If still no frame and not LIVE, try to open first available video from the main app
        if frame_bgr is None and self.live_cap is None:
            try:
                cand: Path | None = None
                # 1) selected files in main window
                if hasattr(master, "selected_files") and master.selected_files:
                    cand = Path(master.selected_files[0])
                # 2) first video in input_dir
                elif hasattr(master, "input_dir"):
                    try:
                        inp = Path(master.input_dir.get())
                    except Exception:
                        inp = None
                    if inp and inp.exists():
                        for p in sorted(inp.iterdir()):
                            if p.suffix.lower() in SUPPORTED_VID_EXTS:
                                cand = p
                                break
                if cand and cand.exists():
                    cap = cv2.VideoCapture(str(cand))
                    ok, fr = cap.read()
                    cap.release()
                    if ok:
                        frame_bgr = fr
            except Exception:
                pass

        if frame_bgr is None:
            frame_bgr = np.zeros((720,1280,3), dtype=np.uint8)

        self._init_canvas_with_frame(frame_bgr)

        # Interaction
        self.canvas.bind("<Button-1>", self.on_click)
        self.bind("<BackSpace>", self.on_backspace)
        self.bind("<Return>", self.on_enter)

        controls = tk.Frame(self); controls.pack(fill="x", pady=6)
        tk.Button(controls, text="Add line", command=lambda: self.set_mode("line")).pack(side="left", padx=4)
        tk.Button(controls, text="Add polyline", command=lambda: self.set_mode("polyline")).pack(side="left", padx=4)  # NEW    
        tk.Button(controls, text="Add zone", command=lambda: self.set_mode("zone")).pack(side="left", padx=4)
        tk.Button(controls, text="Remove last", command=self.remove_last).pack(side="left", padx=10)
        tk.Button(controls, text="Clear (resume LIVE)", command=self.clear_and_resume).pack(side="left", padx=4)
        tk.Button(controls, text="Load…", command=self.load_dialog).pack(side="left", padx=10)
        tk.Button(controls, text="Save…", command=self.save_dialog).pack(side="left", padx=4)
        tk.Button(controls, text="Cancel", command=self.abort).pack(side="right", padx=6)
        tk.Button(controls, text="OK (Save & close)", command=self.finish).pack(side="right", padx=4)

        # Quick rules for A->B direction (ASCII only so it renders everywhere)
        rules = (
            "Line counter rules:\n"
            "• Vertical line, A at bottom -> B at top:  Left->Right counts as A->B (Right->Left = B->A)\n"
            "• Vertical line, A at top    -> B at bottom: Right->Left counts as A->B (Left->Right = B->A)\n"
            "• Horizontal line, A at left -> B at right:  Up->Down counts as A->B (Down->Up = B->A)"
        )
        note = tk.Label(self, text=rules, justify="left", anchor="e", fg="#666")
        note.pack(side="bottom", fill="x", padx=8, pady=(4, 8))

        self.hint = tk.Label(self, text="Mode: LIVE" if self.live else "Mode: static frame")
        self.hint.pack(fill="x")

        # Autoload saved configuration if present
        if self.default_cfg_path and self.default_cfg_path.exists():
            try:
                self.load_from_json(self.default_cfg_path)
                self.hint.config(text=f"{'LIVE – ' if self.live else ''}Loaded: {self.default_cfg_path.name}")
            except Exception:
                pass

        # Start LIVE loop (if applicable)
        if self.live:
            self._tick_live()

        # ⟵ quality of life: open in “Add line” so clicks immediately place A then B
        self.set_mode("line")

    # ---------- Canvas init/resize ----------
    def _init_canvas_with_frame(self, frame_bgr: np.ndarray):
        self.orig_h, self.orig_w = frame_bgr.shape[:2]
        max_w, max_h = 1280, 800
        self.scale = min(1.0, max_w/self.orig_w, max_h/self.orig_h)
        self.disp_w, self.disp_h = int(self.orig_w*self.scale), int(self.orig_h*self.scale)

        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.disp_w, self.disp_h)), cv2.COLOR_BGR2RGB)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h, bg="#222")
        self.canvas.pack(fill="both", expand=True)
        self.canvas_img = self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)
        self.base_bgr = frame_bgr.copy()

    def _update_canvas_image(self, frame_bgr: np.ndarray):
        if frame_bgr is None: return
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.disp_w, self.disp_h)), cv2.COLOR_BGR2RGB)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.itemconfigure(self.canvas_img, image=self.tkimg)

    # ---------- LIVE loop ----------
    def _tick_live(self):
        if not self.live or self.live_cap is None:
            return
        ok, fr = self.live_cap.read()
        if ok and fr is not None:
            self.base_bgr = fr
            self._update_canvas_image(fr)
            self._redraw_overlay_only()
        self._after_id = self.after(33, self._tick_live)  # ~30 FPS

    # ---------- Helpers ----------
    def set_mode(self, m):
        self.mode.set(m)
        self.cur_points = []
        if m in ("line", "polyline", "zone") and self.live:
            self._pause_live()
            if m == "line":
                self.hint.config(text="PAUSE — draw straight line: click A, click B. (Backspace undo)")
            elif m == "polyline":
                self.hint.config(text="PAUSE — draw polyline: click points (A…B), Enter to finish. (Backspace undo)")
            else:
                self.hint.config(text="PAUSE — draw zone: click points (3+), Enter to finish. (Backspace undo)")
        elif m == "idle":
            self.hint.config(text="Mode: idle" if not self.live else "LIVE")
        self._redraw_overlay_only()

    def _pause_live(self):
        if self._after_id is not None:
            try: self.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None
        self.live = False
        self.hint.config(text="PAUSE – drawing")

    def _resume_live(self):
        if self.live_cap is None:
            self.hint.config(text="Mode: static frame")
            return
        self.live = True
        self.hint.config(text="LIVE – select drawing tool to pause.")
        self._tick_live()

    def clear_and_resume(self):
        """Clears sketches and RESUMES LIVE."""
        self.lines.clear(); self.zones.clear(); self.cur_points.clear(); self.mode.set("idle")
        self._redraw_overlay_only()
        self._resume_live()

    def disp_to_img(self, x, y): return [x/self.scale, y/self.scale]
    def img_to_disp(self, x, y): return [x*self.scale, y*self.scale]

    def on_click(self, e):
        if self.mode.get() == "idle":
            return
        if self.live:
            self._pause_live()
        xi, yi = self.disp_to_img(e.x, e.y)

        if self.mode.get() == "line":
            # classic 2-click line
            self.cur_points.append([xi, yi])
            if len(self.cur_points) == 2:
                self.ask_name_and_add_line(self.cur_points[0], self.cur_points[1])
                self.cur_points = []; self.mode.set("idle")
                self.hint.config(text="Line added. (Resume LIVE with “Clear (resume LIVE)”)")

        elif self.mode.get() == "polyline":
            # any number of points, finish with Enter
            if len(self.cur_points) < 64:  # sane guard
                self.cur_points.append([xi, yi])

        elif self.mode.get() == "zone":
            if len(self.cur_points) < 64:
                self.cur_points.append([xi, yi])

        self._redraw_overlay_only()

    def on_backspace(self, _):
        if self.cur_points:
            self.cur_points.pop(); self._redraw_overlay_only()

    def on_enter(self, _):
        if self.mode.get() == "polyline" and len(self.cur_points) >= 2:
            self.ask_name_and_add_polyline(self.cur_points)
            self.cur_points = []; self.mode.set("idle")
            self.hint.config(text="Polyline added. (Resume LIVE with “Clear (resume LIVE)”)")
            self._redraw_overlay_only()
        elif self.mode.get() == "zone" and len(self.cur_points) >= 3:
            self.ask_name_and_add_zone(self.cur_points)
            self.cur_points = []; self.mode.set("idle")
            self.hint.config(text="Zone added. (Resume LIVE with “Clear (resume LIVE)”)")
            self._redraw_overlay_only()

    def ask_name_and_add_line(self, a, b):
        name = self._ask_name("Line name — first click=A, second=B (direction A->B):")
        if not name: return
        self.lines.append({"name": name, "a": [float(a[0]), float(a[1])], "b":[float(b[0]), float(b[1])]})
        self._redraw_overlay_only()

    def ask_name_and_add_zone(self, pts):
        name = self._ask_name("Zone name:")
        if not name: return
        self.zones.append({"name": name, "pts": [[float(x), float(y)] for x,y in pts]})
        self._redraw_overlay_only()

    def _ask_name(self, prompt):
        win = tk.Toplevel(self); win.title("Name")
        win.protocol("WM_DELETE_WINDOW", self.abort)
        win.bind("<Escape>", self.abort)
        tk.Label(win, text=prompt, justify="left", anchor="w").pack(padx=6, pady=6, fill="x")
        var = tk.StringVar()
        e = tk.Entry(win, textvariable=var); e.pack(padx=6, pady=6, fill="x"); e.focus_set()
        out = {"val": None}
        def ok():
            out["val"] = var.get().strip(); win.destroy()
        tk.Button(win, text="OK", command=ok).pack(padx=6, pady=(0,6))
        self.wait_window(win)
        return out["val"]

    def remove_last(self):
        if self.zones: self.zones.pop()
        elif self.lines: self.lines.pop()
        self._redraw_overlay_only()

    def _redraw_overlay_only(self):
        self.canvas.delete("overlay")

        # current sketch (yellow)
        if self.cur_points:
            pts = [self.img_to_disp(p[0], p[1]) for p in self.cur_points]
            for i, (x, y) in enumerate(pts):
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="yellow", outline="", tags="overlay")
                if i > 0:
                    x0, y0 = pts[i-1]
                    self.canvas.create_line(x0, y0, x, y, fill="yellow", width=2, tags="overlay")
            if self.mode.get() == "line" and len(pts) == 2:
                # show temporary (A->B)
                ax, ay = pts[0]; bx, by = pts[1]
                self.canvas.create_line(ax, ay, bx, by, fill="#00FFFF", width=3,
                                        arrow="last", arrowshape=(12,14,6), tags="overlay")
                self.canvas.create_text((ax+bx)/2, (ay+by)/2 - 12, text="(A->B)", fill="#00FFFF", tags="overlay")

        # saved lines (straight & polyline)
        for ln in self.lines:
            col = "#00FFFF"
            if "pts" in ln and len(ln["pts"]) >= 2:
                pts = [self.img_to_disp(p[0], p[1]) for p in ln["pts"]]
                for i in range(1, len(pts)):
                    x0,y0 = pts[i-1]; x1,y1 = pts[i]
                    self.canvas.create_line(x0,y0,x1,y1, fill=col, width=3,
                                            arrow=("last" if i == len(pts)-1 else None),
                                            arrowshape=(12,14,6) if i == len(pts)-1 else None,
                                            tags="overlay")
                ax, ay = pts[0]; bx, by = pts[-1]
            else:
                a = self.img_to_disp(*ln["a"]); b = self.img_to_disp(*ln["b"])
                ax, ay = a; bx, by = b
                self.canvas.create_line(ax, ay, bx, by, fill=col, width=3,
                                        arrow="last", arrowshape=(12,14,6), tags="overlay")
            # endpoints A/B markers + label
            self.canvas.create_oval(ax-3, ay-3, ax+3, ay+3, fill=col, outline="", tags="overlay")
            self.canvas.create_oval(bx-3, by-3, bx+3, by+3, fill=col, outline="", tags="overlay")
            self.canvas.create_text(ax+8, ay-8, text="A", fill=col, tags="overlay")
            self.canvas.create_text(bx+8, by-8, text="B", fill=col, tags="overlay")
            self.canvas.create_text((ax+bx)/2, (ay+by)/2 - 12, text=f"{ln['name']}  (A->B)", fill=col, tags="overlay")

        # saved zones
        for zn in self.zones:
            pts = [self.img_to_disp(x,y) for x,y in zn["pts"]]
            for i in range(len(pts)):
                x1,y1 = pts[i]; x2,y2 = pts[(i+1)%len(pts)]
                self.canvas.create_line(x1,y1,x2,y2, fill="#FFAA00", width=2, tags="overlay")
            cx = sum([p[0] for p in zn["pts"]])/len(zn["pts"])
            cy = sum([p[1] for p in zn["pts"]])/len(zn["pts"])
            dcx, dcy = self.img_to_disp(cx, cy)
            self.canvas.create_text(dcx, dcy, text=zn["name"], fill="#FFAA00", tags="overlay")

    def ask_name_and_add_polyline(self, pts):
        name = self._ask_name("Polyline name — first point = A, last = B (direction A->B):")
        if not name: return
        self.lines.append({"name": name, "pts": [[float(x), float(y)] for x, y in pts]})
        self._redraw_overlay_only()

    # ----- ABORT / FINISH / MODAL -----
    def abort(self, _evt=None):
        """Cancel without saving; mark as aborted and close."""
        try:
            self._aborted = True
            self.result = None
        except Exception:
            pass
        # stop LIVE loop if any
        if self._after_id is not None:
            try: self.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None
        try:
            self.destroy()
        except Exception:
            pass

    def finish(self):
        """Save (to default path if set), set result, close."""
        if self.default_cfg_path:
            try:
                ensure_dir(self.default_cfg_path.parent)
                with open(self.default_cfg_path, "w", encoding="utf-8") as f:
                    json.dump({"lines": self.lines, "zones": self.zones}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        # stop LIVE loop if any
        if self._after_id is not None:
            try: self.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None
        # return value
        self.result = {"lines": self.lines[:], "zones": self.zones[:]}
        self._aborted = False
        self.destroy()

    def run_modal(self):
        """Block until closed; return (result_dict_or_None, aborted_bool)."""
        try:
            self.focus_force()
        except Exception:
            pass
        self.wait_window(self)
        return self.result, bool(self._aborted)

    # ----- file io -----
    def load_dialog(self):
        f = filedialog.askopenfilename(title="Load configuration",
                                       filetypes=[("JSON","*.json"),("All","*.*")])
        if not f: return
        try:
            self.load_from_json(Path(f))
            self.hint.config(text=f"{'LIVE – ' if self.live else ''}Loaded: {Path(f).name}")
        except Exception as e:
            messagebox.showerror("Load", str(e))

    def save_dialog(self):
        f = filedialog.asksaveasfilename(title="Save configuration",
                                         defaultextension=".json",
                                         filetypes=[("JSON","*.json")])
        if not f: return
        try:
            with open(f, "w", encoding="utf-8") as fp:
                json.dump({"lines": self.lines, "zones": self.zones}, fp, ensure_ascii=False, indent=2)
            self.hint.config(text=f"Saved: {Path(f).name}")
        except Exception as e:
            messagebox.showerror("Save", str(e))

    def load_from_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.lines = data.get("lines", [])
        self.zones = data.get("zones", [])
        self.cur_points = []
        self.mode.set("idle")
        self._redraw_overlay_only()

class AppUIMixin:
    # unchanged UI helpers (only here so imports keep working in your app)
    def build_ui(self):
        frm = tk.Frame(self); frm.pack(fill="both", expand=True, padx=8, pady=6)

        # Input
        self._row_browse(frm, "Video folder (input):", self.input_dir, self.browse_input, is_dir=True)
        f_files = tk.Frame(frm); f_files.pack(fill="x", pady=2)
        tk.Button(f_files, text="Select video files…", command=self.browse_files).pack(side="left")
        self.files_label = tk.Label(f_files, text="— none —"); self.files_label.pack(side="left", padx=8)
        tk.Button(f_files, text="Clear selection", command=self.clear_files).pack(side="left", padx=(8,0))

        # Output + weights
        self._row_browse(frm, "Output folder (optional):", self.output_dir, self.browse_output, is_dir=True)
        self._row_browse(frm, "Weights (.pt/.zip):", self.weights_path, self.browse_weights, is_dir=False)

        # Source
        srcf = tk.LabelFrame(frm, text="Input source"); srcf.pack(fill="x", pady=4)
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
        qf = tk.Frame(frm); qf.pack(fill="x", pady=4)
        tk.Label(qf, text="Quality (1 = faster/lower, 5 = ULTRA)").pack(side="left")
        tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality,
                 command=lambda *_: self._update_preset_label()).pack(side="left", fill="x", expand=True, padx=8)
        self.preset_label = tk.Label(qf, text=""); self.preset_label.pack(side="left")
        self._update_preset_label()

        # Overlay
        ov = tk.LabelFrame(frm, text="Visualization (overlay)"); ov.pack(fill="x", pady=4)
        tk.Radiobutton(ov, text="Centroids",   variable=self.overlay_mode, value="centroid").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boxes",       variable=self.overlay_mode, value="boxes").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boxes + conf",variable=self.overlay_mode, value="boxes_conf").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Polygons",    variable=self.overlay_mode, value="polygon").pack(side="left", padx=6)

        # Tracker
        tr = tk.LabelFrame(frm, text="Tracker:"); tr.pack(fill="x", pady=4)
        tk.Radiobutton(tr, text="ByteTrack", variable=self.tracker_kind, value="bytetrack").pack(side="left", padx=6)
        tk.Radiobutton(tr, text="BoT-SORT",  variable=self.tracker_kind, value="botsort").pack(side="left", padx=6)

        # Class list + TOGGLE strip (CvC_images style)
        lf = tk.LabelFrame(frm, text="Class selection (after loading weights)")
        lf.pack(fill="both", expand=True, pady=4)

        tog = tk.Frame(lf); tog.pack(fill="x", pady=(4, 0))
        tk.Label(tog, text="Toggle:").pack(side="left")
        tk.Button(tog, text="All",   width=6, command=self._toggle_all_classes).pack(side="left", padx=(6, 0))
        tk.Button(tog, text="None",  width=6, command=self._toggle_none_classes).pack(side="left", padx=(6, 0))
        tk.Button(tog, text="Invert",width=6, command=self._toggle_invert_classes).pack(side="left", padx=(6, 0))

        self.classes_scroll = ScrollableFrame(lf)
        self.classes_scroll.pack(fill="both", expand=True)

        # Controls + progress
        bf = tk.Frame(frm); bf.pack(fill="x", pady=6)
        self.btn_start = tk.Button(bf, text="START", command=self.start); self.btn_start.pack(side="left")
        tk.Button(bf, text="Advanced options…", command=self.open_advanced).pack(side="left", padx=8)
        self.btn_abort = tk.Button(bf, text="ABORT", command=self.abort, state="disabled"); self.btn_abort.pack(side="left", padx=(8,0))

        pf = tk.Frame(frm); pf.pack(fill="x", pady=(0,4))
        self.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=self.progress_var, mode="determinate")
        self._progress_indeterminate = False
        self.progressbar.pack(fill="x", side="left", expand=True)
        tk.Label(pf, textvariable=self.progress_label, width=36, anchor="w").pack(side="left", padx=6)

        # Log
        logf = tk.Frame(frm); logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, height=10, state="normal"); self.log.pack(fill="both", expand=True)

    # (helper used by your App)
    def _row_browse(self, parent, label, var, cmd, is_dir=True):
        f = tk.Frame(parent); f.pack(fill="x", pady=3)
        tk.Label(f, text=label, width=26, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(f, text="Browse…", command=cmd).pack(side="left")

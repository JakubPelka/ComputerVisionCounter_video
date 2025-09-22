# cv_video_gui.py
from __future__ import annotations
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

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
    Tryb standard (pliki): pokazuje zamrożoną pierwszą klatkę i rysujemy na niej.
    Tryb LIVE (kamera/URL): przekaż live_cap=VideoCapture — wtedy:
      • podgląd jest na żywo,
      • gdy wybierzesz „Dodaj linię/strefę”, podgląd AUTOPAUZA, żeby wygodnie narysować,
      • „Wyczyść (wznów LIVE)” czyści szkice i wznawia LIVE.
    """
    def __init__(self, master, frame_bgr: np.ndarray|None,
                 default_cfg_path: Path|None=None,
                 live_cap: cv2.VideoCapture|None=None):
        super().__init__(master)
        self.title("Konfiguracja liczników (LIVE/STATYCZNIE)")

        self.lines: list[dict] = []
        self.zones: list[dict] = []
        self.mode = tk.StringVar(value="idle")
        self.cur_points: list[list[float]] = []
        self.default_cfg_path = default_cfg_path

        # LIVE state
        self.live_cap = live_cap
        self.live = bool(live_cap is not None)
        self._after_id = None
        self.base_bgr = None  # ostatnia ramka do overlay (w pauzie – zamrożona)
        self.disp_w = self.disp_h = 0
        self.scale = 1.0

        # Przy streamie spróbuj złapać pierwszą ramkę, jeżeli nie podano frame_bgr
        if self.live_cap is not None and frame_bgr is None:
            ok, fr = self.live_cap.read()
            frame_bgr = fr if ok else None

        if frame_bgr is None:
            frame_bgr = np.zeros((720,1280,3), dtype=np.uint8)

        self._init_canvas_with_frame(frame_bgr)

        # Interakcja
        self.canvas.bind("<Button-1>", self.on_click)
        self.bind("<BackSpace>", self.on_backspace)
        self.bind("<Return>", self.on_enter)

        controls = tk.Frame(self); controls.pack(fill="x", pady=6)
        tk.Button(controls, text="Dodaj linię", command=lambda: self.set_mode("line")).pack(side="left", padx=4)
        tk.Button(controls, text="Dodaj strefę", command=lambda: self.set_mode("zone")).pack(side="left", padx=4)
        tk.Button(controls, text="Usuń ostatni element", command=self.remove_last).pack(side="left", padx=10)
        tk.Button(controls, text="Wyczyść (wznów LIVE)", command=self.clear_and_resume).pack(side="left", padx=4)
        tk.Button(controls, text="Wczytaj…", command=self.load_dialog).pack(side="left", padx=10)
        tk.Button(controls, text="Zapisz…", command=self.save_dialog).pack(side="left", padx=4)
        tk.Button(controls, text="OK (Zapisz i zamknij)", command=self.finish).pack(side="right", padx=4)

        self.hint = tk.Label(self, text="Tryb: LIVE" if self.live else "Tryb: statyczna klatka")
        self.hint.pack(fill="x")

        # Autoload zapisanej konfiguracji jeśli jest
        if self.default_cfg_path and self.default_cfg_path.exists():
            try:
                self.load_from_json(self.default_cfg_path)
                self.hint.config(text=f"{'LIVE – ' if self.live else ''}Wczytano: {self.default_cfg_path.name}")
            except Exception:
                pass

        # Start pętli LIVE
        if self.live:
            self._tick_live()

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
        if m in ("line", "zone") and self.live:
            self._pause_live()
            self.hint.config(text="PAUZA – rysuj (Backspace cofa, Enter kończy).")
        elif m == "idle":
            self.hint.config(text="Tryb: bezczynny" if not self.live else "LIVE")
        self._redraw_overlay_only()

    def _pause_live(self):
        if self._after_id is not None:
            try: self.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None
        self.live = False
        self.hint.config(text="PAUZA – rysowanie")

    def _resume_live(self):
        if self.live_cap is None:
            self.hint.config(text="Tryb: statyczna klatka")
            return
        self.live = True
        self.hint.config(text="LIVE – ustaw kamerę i kadr; wybierz rysowanie aby zapauzować.")
        self._tick_live()

    def clear_and_resume(self):
        """Czyści szkice i WZNOWI LIVE."""
        self.lines.clear(); self.zones.clear(); self.cur_points.clear(); self.mode.set("idle")
        self._redraw_overlay_only()
        self._resume_live()

    def disp_to_img(self, x, y): return [x/self.scale, y/self.scale]
    def img_to_disp(self, x, y): return [x*self.scale, y*self.scale]

    def on_click(self, e):
        if self.mode.get() == "idle":
            return
        if self.live:
            self._pause_live()  # pewność, że rysujemy na zamrożonej klatce
        xi, yi = self.disp_to_img(e.x, e.y)
        if self.mode.get() == "line":
            self.cur_points.append([xi, yi])
            if len(self.cur_points) == 2:
                self.ask_name_and_add_line(self.cur_points[0], self.cur_points[1])
                self.cur_points = []; self.mode.set("idle")
                self.hint.config(text="Linia dodana. (Wznów LIVE przyciskiem „Wyczyść (wznów LIVE)”)")
        elif self.mode.get() == "zone":
            if len(self.cur_points) < 10:
                self.cur_points.append([xi, yi])
        self._redraw_overlay_only()

    def on_backspace(self, _):
        if self.cur_points:
            self.cur_points.pop(); self._redraw_overlay_only()

    def on_enter(self, _):
        if self.mode.get() == "zone" and len(self.cur_points) >= 3:
            self.ask_name_and_add_zone(self.cur_points)
            self.cur_points = []; self.mode.set("idle")
            self.hint.config(text="Strefa dodana. (Wznów LIVE przyciskiem „Wyczyść (wznów LIVE)”)")
            self._redraw_overlay_only()

    def ask_name_and_add_line(self, a, b):
        name = self._ask_name("Nazwa linii (kierunek A→B):")
        if not name: return
        self.lines.append({"name": name, "a": [float(a[0]), float(a[1])], "b":[float(b[0]), float(b[1])]})
        self._redraw_overlay_only()

    def ask_name_and_add_zone(self, pts):
        name = self._ask_name("Nazwa strefy:")
        if not name: return
        self.zones.append({"name": name, "pts": [[float(x), float(y)] for x,y in pts]})
        self._redraw_overlay_only()

    def _ask_name(self, prompt):
        win = tk.Toplevel(self); win.title("Nazwa")
        tk.Label(win, text=prompt).pack(padx=6, pady=6)
        var = tk.StringVar()
        e = tk.Entry(win, textvariable=var); e.pack(padx=6, pady=6); e.focus_set()
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
        # cur_points – żółte
        for i,p in enumerate(self.cur_points):
            dx,dy = self.img_to_disp(p[0], p[1])
            self.canvas.create_oval(dx-3, dy-3, dx+3, dy+3, fill="yellow", outline="", tags="overlay")
            if i>0:
                px,py = self.img_to_disp(self.cur_points[i-1][0], self.cur_points[i-1][1])
                self.canvas.create_line(px,py,dx,dy, fill="yellow", width=2, tags="overlay")
        # linie
        for ln in self.lines:
            a = self.img_to_disp(*ln["a"]); b = self.img_to_disp(*ln["b"])
            self.canvas.create_line(a[0],a[1],b[0],b[1], fill="#00FFFF", width=3, tags="overlay")
            self.canvas.create_text((a[0]+b[0])/2, (a[1]+b[1])/2 - 10, text=ln["name"], fill="#00FFFF", tags="overlay")
        # strefy
        for zn in self.zones:
            pts = [self.img_to_disp(x,y) for x,y in zn["pts"]]
            for i in range(len(pts)):
                x1,y1 = pts[i]; x2,y2 = pts[(i+1)%len(pts)]
                self.canvas.create_line(x1,y1,x2,y2, fill="#FFAA00", width=2, tags="overlay")
            cx = sum([p[0] for p in zn["pts"]])/len(zn["pts"])
            cy = sum([p[1] for p in zn["pts"]])/len(zn["pts"])
            dcx, dcy = self.img_to_disp(cx, cy)
            self.canvas.create_text(dcx, dcy, text=zn["name"], fill="#FFAA00", tags="overlay")

    def finish(self):
        if self.default_cfg_path:
            try:
                ensure_dir(self.default_cfg_path.parent)
                with open(self.default_cfg_path, "w", encoding="utf-8") as f:
                    json.dump({"lines": self.lines, "zones": self.zones}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        # stop ewentualny loop LIVE
        if self._after_id is not None:
            try: self.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None
        self.destroy()

    def load_dialog(self):
        f = filedialog.askopenfilename(title="Wczytaj konfigurację",
                                       filetypes=[("JSON","*.json"),("Wszystkie","*.*")])
        if not f: return
        try:
            self.load_from_json(Path(f))
            self.hint.config(text=f"{'LIVE – ' if self.live else ''}Wczytano: {Path(f).name}")
        except Exception as e:
            messagebox.showerror("Wczytaj", str(e))

    def save_dialog(self):
        f = filedialog.asksaveasfilename(title="Zapisz konfigurację",
                                         defaultextension=".json",
                                         filetypes=[("JSON","*.json")])
        if not f: return
        try:
            with open(f, "w", encoding="utf-8") as fp:
                json.dump({"lines": self.lines, "zones": self.zones}, fp, ensure_ascii=False, indent=2)
            self.hint.config(text=f"Zapisano: {Path(f).name}")
        except Exception as e:
            messagebox.showerror("Zapisz", str(e))

    def load_from_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.lines = data.get("lines", [])
        self.zones = data.get("zones", [])
        self.cur_points = []
        self.mode.set("idle")
        self._redraw_overlay_only()

class AppUIMixin:
    def build_ui(self):
        frm = tk.Frame(self); frm.pack(fill="both", expand=True, padx=8, pady=6)

        # Wejście
        self._row_browse(frm, "Folder z wideo (wejście):", self.input_dir, self.browse_input, is_dir=True)
        f_files = tk.Frame(frm); f_files.pack(fill="x", pady=2)
        tk.Button(f_files, text="Wybierz pliki wideo...", command=self.browse_files).pack(side="left")
        self.files_label = tk.Label(f_files, text="— brak —"); self.files_label.pack(side="left", padx=8)
        tk.Button(f_files, text="Wyczyść wybór", command=self.clear_files).pack(side="left", padx=(8,0))

        # Wyjście + wagi
        self._row_browse(frm, "Folder wynikowy (opcjonalnie):", self.output_dir, self.browse_output, is_dir=True)
        self._row_browse(frm, "Wagi (.pt/.zip):", self.weights_path, self.browse_weights, is_dir=False)

        # Źródło
        srcf = tk.LabelFrame(frm, text="Źródło wejściowe"); srcf.pack(fill="x", pady=4)
        self.src_mode = tk.StringVar(value="files"); self.cam_index = tk.StringVar(value="0"); self.url_input = tk.StringVar(value="")
        def _src_toggle(*_):
            mf = self.src_mode.get()
            cam_ent.config(state=("normal" if mf=="camera" else "disabled"))
            url_ent.config(state=("normal" if mf=="url" else "disabled"))
        tk.Radiobutton(srcf, text="Pliki",  variable=self.src_mode, value="files", command=_src_toggle).pack(side="left", padx=6)
        tk.Radiobutton(srcf, text="Kamera", variable=self.src_mode, value="camera", command=_src_toggle).pack(side="left", padx=6)
        tk.Label(srcf, text="Index:").pack(side="left")
        cam_ent = tk.Entry(srcf, width=4, textvariable=self.cam_index); cam_ent.pack(side="left", padx=(0,8))
        tk.Radiobutton(srcf, text="RTSP/HTTP URL", variable=self.src_mode, value="url", command=_src_toggle).pack(side="left", padx=6)
        url_ent = tk.Entry(srcf, textvariable=self.url_input); url_ent.pack(side="left", fill="x", expand=True, padx=(0,6))
        _src_toggle()

        # Jakość (suwak) + etykieta
        qf = tk.Frame(frm); qf.pack(fill="x", pady=4)
        tk.Label(qf, text="Jakość (=1 szybciej/słabiej, 5 = ULTRA)").pack(side="left")
        tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality,
                 command=lambda *_: self._update_preset_label()).pack(side="left", fill="x", expand=True, padx=8)
        self.preset_label = tk.Label(qf, text=""); self.preset_label.pack(side="left")
        self._update_preset_label()

        # Profile CPU – presety
        prof = tk.LabelFrame(frm, text="Profile (CPU)")
        prof.pack(fill="x", pady=(2,4))
        self.cpu_profile = getattr(self, "cpu_profile", None) or tk.StringVar(value="Default")
        cb = ttk.Combobox(prof, values=["Default","CPU Turbo","CPU Balanced","CPU Quality"],
                          textvariable=self.cpu_profile, state="readonly", width=16)
        cb.pack(side="left", padx=6, pady=2)
        tk.Button(prof, text="Zastosuj profil", command=self._apply_cpu_profile).pack(side="left", padx=6)

        # Overlay
        ov = tk.LabelFrame(frm, text="Wizualizacja (overlay)"); ov.pack(fill="x", pady=4)
        tk.Radiobutton(ov, text="Centroidy", variable=self.overlay_mode, value="centroid").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boksy", variable=self.overlay_mode, value="boxes").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boksy + conf", variable=self.overlay_mode, value="boxes_conf").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Polygony", variable=self.overlay_mode, value="polygon").pack(side="left", padx=6)

        # Tracker
        tr = tk.LabelFrame(frm, text="Tracker:"); tr.pack(fill="x", pady=4)
        tk.Radiobutton(tr, text="ByteTrack", variable=self.tracker_kind, value="bytetrack").pack(side="left", padx=6)
        tk.Radiobutton(tr, text="BoT-SORT", variable=self.tracker_kind, value="botsort").pack(side="left", padx=6)

        # Lista klas
        lf = tk.LabelFrame(frm, text="Wybór klas (po wczytaniu wag)"); lf.pack(fill="both", expand=True, pady=4)
        self.classes_scroll = ScrollableFrame(lf); self.classes_scroll.pack(fill="both", expand=True)

        # Sterowanie + postęp
        bf = tk.Frame(frm); bf.pack(fill="x", pady=6)
        self.btn_start = tk.Button(bf, text="START", command=self.start); self.btn_start.pack(side="left")
        tk.Button(bf, text="Opcje zaawansowane…", command=self.open_advanced).pack(side="left", padx=8)
        self.btn_abort = tk.Button(bf, text="ABORT", command=self.abort, state="disabled"); self.btn_abort.pack(side="left", padx=(8,0))

        pf = tk.Frame(frm); pf.pack(fill="x", pady=(0,4))
        self.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=self.progress_var, mode="determinate")
        self._progress_indeterminate = False
        self.progressbar.pack(fill="x", side="left", expand=True)
        tk.Label(pf, textvariable=self.progress_label, width=36, anchor="w").pack(side="left", padx=6)

        # Podgląd / Trace / Anchor / Ghost
        pvf = tk.LabelFrame(frm, text="Podgląd / Trace / Anchor / Ghost")
        pvf.pack(fill="x", pady=(4, 6))
        self.preview_enabled = getattr(self, "preview_enabled", None) or tk.BooleanVar(value=True)
        tk.Checkbutton(pvf, text="Włącz podgląd", variable=self.preview_enabled).pack(side="left", padx=6)
        self.trace_enabled = getattr(self, "trace_enabled", None) or tk.BooleanVar(value=True)
        self.trace_len     = getattr(self, "trace_len", None) or tk.IntVar(value=24)
        self.anchor_mode   = getattr(self, "anchor_mode", None) or tk.StringVar(value="bottom")
        self.ghost_margin  = getattr(self, "ghost_margin", None) or tk.IntVar(value=12)
        tk.Checkbutton(pvf, text="Ślad", variable=self.trace_enabled).pack(side="left", padx=(6, 4))
        tk.Label(pvf, text="len:").pack(side="left")
        tk.Spinbox(pvf, from_=0, to=300, width=5, textvariable=self.trace_len).pack(side="left", padx=(2, 12))
        tk.Label(pvf, text="Anchor:").pack(side="left")
        ttk.Combobox(pvf, values=["bottom","center"], width=8, state="readonly",
                     textvariable=self.anchor_mode).pack(side="left", padx=(3, 12))
        tk.Label(pvf, text="Ghost margin (px):").pack(side="left")
        tk.Spinbox(pvf, from_=0, to=64, width=5, textvariable=self.ghost_margin).pack(side="left", padx=(3, 6))

        # ALERTY DŹWIĘKOWE
        af = tk.LabelFrame(frm, text="Alert dźwiękowy (strefy)")
        af.pack(fill="x", pady=(0,8))
        self.alert_enabled = getattr(self, "alert_enabled", None) or tk.BooleanVar(value=False)
        self.alert_classes = getattr(self, "alert_classes", None) or tk.StringVar(value="cat,person")
        self.alert_freq    = getattr(self, "alert_freq", None)    or tk.IntVar(value=880)
        self.alert_dur     = getattr(self, "alert_dur", None)     or tk.IntVar(value=180)
        self.alert_freeze  = getattr(self, "alert_freeze", None)  or tk.IntVar(value=1500)  # ms
        tk.Checkbutton(af, text="Włącz alert", variable=self.alert_enabled).pack(side="left", padx=6)
        tk.Label(af, text="Klasy (CSV):").pack(side="left")
        tk.Entry(af, textvariable=self.alert_classes, width=24).pack(side="left", padx=(3, 12))
        tk.Label(af, text="Hz:").pack(side="left")
        tk.Spinbox(af, from_=200, to=4000, width=6, textvariable=self.alert_freq).pack(side="left", padx=(3, 12))
        tk.Label(af, text="ms:").pack(side="left")
        tk.Spinbox(af, from_=30, to=2000, width=6, textvariable=self.alert_dur).pack(side="left", padx=(3, 12))
        tk.Label(af, text="freeze (ms):").pack(side="left")
        tk.Spinbox(af, from_=0, to=10000, width=7, textvariable=self.alert_freeze).pack(side="left", padx=(3, 6))

        # Log
        logf = tk.Frame(frm); logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, height=10, state="normal"); self.log.pack(fill="both", expand=True)
        self._update_preset_label()

    # ====== Profile CPU: ustawia adv_params i włącza override ======
    def _apply_cpu_profile(self):
        prof = self.cpu_profile.get()
        # Bezpieczna baza
        base = dict(imgsz=384, conf=0.60, iou=0.50, frame_skip=2,
                    track_buffer=4, match_thresh=0.88, min_hits=2,
                    line_min_gap=8, line_min_sep=12, zone_min_gap=6)
        if prof == "CPU Turbo":
            base.update(imgsz=320, frame_skip=3, track_buffer=3, match_thresh=0.90, min_hits=1)
        elif prof == "CPU Balanced":
            base.update(imgsz=384, frame_skip=2, track_buffer=4, match_thresh=0.88, min_hits=2)
        elif prof == "CPU Quality":
            base.update(imgsz=512, frame_skip=1, track_buffer=6, match_thresh=0.85, min_hits=2)
        else:
            self.advanced_override = False
            self._log("[PROFILE] Default – użyję presetu z suwaka jakości.")
            self._update_preset_label(); return
        self.adv_params = base
        self.advanced_override = True
        self._log(f"[PROFILE] Zastosowano: {prof} → {base}")
        self._update_preset_label()

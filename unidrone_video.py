import os, json, zipfile, threading, time, math, sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd

import torch
from ultralytics import YOLO
from PIL import Image, ImageTk

try:
    import supervision as sv
except Exception:
    sv = None

# --- Scrollowalny kontener na listę klas ---
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

        # aktualizacja scrollregion, gdy zawartość się zmieni
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # dopasuj szerokość okna do szerokości canvasu
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.win, width=e.width))

        # przewijanie kółkiem (Windows)
        self.inner.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        # delta 120 na Windows → krok 1
        self.canvas.yview_scroll(int(-event.delta/120), "units")


# ===================== STAŁE / PRESETY =====================
SUPPORTED_VID_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")
MODEL_DIRNAME = "models"

# Presety jakości (dla wideo)
VIDEO_PRESETS = {
    1: {"imgsz": 640,  "conf": 0.50, "iou": 0.60, "frame_skip": 2, "track_buffer": 30, "match_thresh": 0.80, "min_hits": 2},
    2: {"imgsz": 896,  "conf": 0.55, "iou": 0.55, "frame_skip": 2, "track_buffer": 45, "match_thresh": 0.80, "min_hits": 2},
    3: {"imgsz": 960,  "conf": 0.60, "iou": 0.50, "frame_skip": 1, "track_buffer": 60, "match_thresh": 0.78, "min_hits": 2},
    4: {"imgsz": 1280, "conf": 0.65, "iou": 0.50, "frame_skip": 1, "track_buffer": 75, "match_thresh": 0.75, "min_hits": 3},
    5: {"imgsz": 1280, "conf": 0.70, "iou": 0.45, "frame_skip": 0, "track_buffer": 90, "match_thresh": 0.75, "min_hits": 3},  # ULTRA
}
DEFAULT_QUALITY = 5  # ULTRA
DEFAULT_TRACKER = "bytetrack"  # "bytetrack" lub "botsort"

# Histereza zdarzeń
LINE_MIN_GAP_FRAMES_DEFAULT = 8
LINE_MIN_SEP_PX_DEFAULT    = 12
ZONE_MIN_GAP_FRAMES_DEFAULT = 6

# ===================== UTIL =====================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def device_auto_str() -> str:
    return "0" if torch.cuda.is_available() else "cpu"

def score_weight_name(p: Path) -> int:
    name = p.name.lower(); score = 0
    try: score += int(p.stat().st_size // (1024*1024))
    except Exception: pass
    if "x.pt" in name: score += 100
    if "l.pt" in name: score += 80
    if "m.pt" in name: score += 60
    return score

def find_best_weights(models_dir: Path) -> Path | None:
    cands = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.zip"))
    if not cands: return None
    cands.sort(key=score_weight_name, reverse=True); return cands[0]

def resolve_weights_to_pt(path: Path, extract_dir: Path) -> Path:
    if path.suffix.lower() == ".pt": return path
    if path.suffix.lower() == ".zip":
        ensure_dir(extract_dir)
        with zipfile.ZipFile(path, "r") as z: z.extractall(extract_dir)
        pts = list(extract_dir.rglob("*.pt"))
        if not pts: raise RuntimeError("W archiwum .zip nie znaleziono pliku .pt")
        pts.sort(key=score_weight_name, reverse=True); return pts[0]
    raise ValueError("Wybierz .pt lub .zip")

def numbered_path(path: Path) -> Path:
    if not path.exists(): return path
    stem, suf = path.stem, path.suffix; i = 1
    while True:
        cand = path.with_name(f"{stem}_{i}{suf}")
        if not cand.exists(): return cand
        i += 1

def open_video_writer_collision(path: Path, w: int, h: int, fps: float) -> (cv2.VideoWriter, Path):
    out_path = path
    try:
        if out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer.isOpened(): return writer, out_path
    except Exception:
        pass
    out_path = numbered_path(path)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return writer, out_path

def save_json_collision(obj, path: Path) -> Path:
    try:
        if path.exists():
            try: path.unlink()
            except Exception: pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        alt = numbered_path(path)
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return alt

def save_csv_collision(df: pd.DataFrame, path: Path) -> Path:
    try:
        if path.exists():
            try: path.unlink()
            except Exception: pass
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except Exception:
        alt = numbered_path(path)
        df.to_csv(alt, index=False, encoding="utf-8")
        return alt

# ====== Geometria linii/poligonów ======
def line_side(a, b, p):
    # znak det(B-A, P-A); >0 po lewej od A->B, <0 po prawej
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

def segments_intersect(p1, p2, q1, q2):
    def orient(a,b,c):
        v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if v > 0: return 1
        if v < 0: return -1
        return 0
    def on_seg(a,b,c):
        return (min(a[0],b[0]) - 1e-6 <= c[0] <= max(a[0],b[0]) + 1e-6 and
                min(a[1],b[1]) - 1e-6 <= c[1] <= max(a[1],b[1]) + 1e-6)
    o1 = orient(p1,p2,q1); o2 = orient(p1,p2,q2)
    o3 = orient(q1,q2,p1); o4 = orient(q1,q2,p2)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_seg(p1,p2,q1): return True
    if o2 == 0 and on_seg(p1,p2,q2): return True
    if o3 == 0 and on_seg(q1,q2,p1): return True
    if o4 == 0 and on_seg(q1,q2,p2): return True
    return False

def dist_point_to_segment(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    ab2 = abx*abx + aby*aby
    if ab2 <= 1e-9: return math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, (apx*abx + apy*aby)/ab2))
    cx, cy = ax + t*abx, ay + t*aby
    return math.hypot(px-cx, py-cy)

def point_in_polygon(pt, poly):
    poly_np = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, pt, False) >= 0

# ===================== EDYTOR KONFIGU Linii/Poligonów =====================
class CounterEditor(tk.Toplevel):
    """
    Edytor na pierwszej klatce: dodawanie wielu linii (A->B) i poligonów.
    + Wczytywanie/zapisywanie konfiguracji JSON.
    Sterowanie:
      - 'Dodaj linię' -> klik A, klik B -> nazwa -> Zapisz
      - 'Dodaj strefę' -> klikaj punkty (min 3, max 10), Enter kończy -> nazwa -> Zapisz
      - Backspace cofa ostatni punkt
      - Można usuwać ostatni element / czyścić wszystko
    """
    def __init__(self, master, frame_bgr, default_cfg_path: Path|None=None):
        super().__init__(master)
        self.title("Konfiguracja liczników (pierwsza klatka)")
        self.lines = []   # {name, a(x,y), b(x,y)}
        self.zones = []   # {name, pts[[x,y],...]}
        self.mode = tk.StringVar(value="idle")
        self.cur_points = []
        self.default_cfg_path = default_cfg_path

        self.orig_h, self.orig_w = frame_bgr.shape[:2]
        max_w, max_h = 1280, 800
        scale = min(1.0, max_w/self.orig_w, max_h/self.orig_h)
        self.scale = scale
        disp_w, disp_h = int(self.orig_w*scale), int(self.orig_h*scale)
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))

        # UI
        self.canvas = tk.Canvas(self, width=disp_w, height=disp_h, bg="#222")
        self.canvas.pack(fill="both", expand=True)
        self.canvas_img = self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)
        self.canvas.bind("<Button-1>", self.on_click)
        self.bind("<BackSpace>", self.on_backspace)
        self.bind("<Return>", self.on_enter)

        controls = tk.Frame(self); controls.pack(fill="x", pady=6)
        tk.Button(controls, text="Dodaj linię", command=lambda: self.set_mode("line")).pack(side="left", padx=4)
        tk.Button(controls, text="Dodaj strefę", command=lambda: self.set_mode("zone")).pack(side="left", padx=4)
        tk.Button(controls, text="Usuń ostatni element", command=self.remove_last).pack(side="left", padx=10)
        tk.Button(controls, text="Wyczyść", command=self.clear_all).pack(side="left", padx=4)
        tk.Button(controls, text="Wczytaj…", command=self.load_dialog).pack(side="left", padx=10)
        tk.Button(controls, text="Zapisz…", command=self.save_dialog).pack(side="left", padx=4)
        tk.Button(controls, text="OK (Zapisz i zamknij)", command=self.finish).pack(side="right", padx=4)
        self.hint = tk.Label(self, text="Tryb: bezczynny"); self.hint.pack(fill="x")

        # auto-load default config jeśli istnieje
        if self.default_cfg_path and self.default_cfg_path.exists():
            try:
                self.load_from_json(self.default_cfg_path)
                self.hint.config(text=f"Wczytano: {self.default_cfg_path.name}")
            except Exception:
                pass

        self._redraw()

    def set_mode(self, m):
        self.mode.set(m)
        self.cur_points = []
        if m == "line":
            self.hint.config(text="Dodaj linię: kliknij punkt A, potem B. Backspace cofa, Enter kończy.")
        elif m == "zone":
            self.hint.config(text="Dodaj strefę: klikaj 3–10 punktów. Backspace cofa, Enter kończy.")
        else:
            self.hint.config(text="Tryb: bezczynny")
        self._redraw()

    def disp_to_img(self, x, y): return [x/self.scale, y/self.scale]
    def img_to_disp(self, x, y): return [x*self.scale, y*self.scale]

    def on_click(self, e):
        if self.mode.get() == "idle": return
        xi, yi = self.disp_to_img(e.x, e.y)
        if self.mode.get() == "line":
            self.cur_points.append([xi, yi])
            if len(self.cur_points) == 2:
                self.ask_name_and_add_line(self.cur_points[0], self.cur_points[1])
                self.cur_points = []; self.mode.set("idle"); self.hint.config(text="Linia dodana.")
        elif self.mode.get() == "zone":
            if len(self.cur_points) < 10:
                self.cur_points.append([xi, yi])
        self._redraw()

    def on_backspace(self, _):
        if self.cur_points:
            self.cur_points.pop(); self._redraw()

    def on_enter(self, _):
        if self.mode.get() == "zone" and len(self.cur_points) >= 3:
            self.ask_name_and_add_zone(self.cur_points)
            self.cur_points = []; self.mode.set("idle"); self.hint.config(text="Strefa dodana.")
            self._redraw()

    def ask_name_and_add_line(self, a, b):
        name = self._ask_name("Nazwa linii (kierunek A→B):")
        if not name: return
        self.lines.append({"name": name, "a": [float(a[0]), float(a[1])], "b":[float(b[0]), float(b[1])]})
        self._redraw()

    def ask_name_and_add_zone(self, pts):
        name = self._ask_name("Nazwa strefy:")
        if not name: return
        self.zones.append({"name": name, "pts": [[float(x), float(y)] for x,y in pts]})
        self._redraw()

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
        if self.zones:
            self.zones.pop()
        elif self.lines:
            self.lines.pop()
        self._redraw()

    def clear_all(self):
        self.lines.clear(); self.zones.clear(); self.cur_points.clear(); self.mode.set("idle")
        self._redraw()

    def _redraw(self):
        self.canvas.delete("overlay")
        # Cur points
        for i,p in enumerate(self.cur_points):
            dx,dy = self.img_to_disp(p[0], p[1])
            self.canvas.create_oval(dx-3, dy-3, dx+3, dy+3, fill="yellow", outline="", tags="overlay")
            if i>0:
                px,py = self.img_to_disp(self.cur_points[i-1][0], self.cur_points[i-1][1])
                self.canvas.create_line(px,py,dx,dy, fill="yellow", width=2, tags="overlay")
        # Lines
        for ln in self.lines:
            a = self.img_to_disp(*ln["a"]); b = self.img_to_disp(*ln["b"])
            self.canvas.create_line(a[0],a[1],b[0],b[1], fill="#00FFFF", width=3, tags="overlay")
            self.canvas.create_text((a[0]+b[0])/2, (a[1]+b[1])/2 - 10, text=ln["name"], fill="#00FFFF", tags="overlay")
        # Zones
        for zn in self.zones:
            pts = [self.img_to_disp(x,y) for x,y in zn["pts"]]
            for i in range(len(pts)):
                x1,y1 = pts[i]; x2,y2 = pts[(i+1)%len(pts)]
                self.canvas.create_line(x1,y1,x2,y2, fill="#FFAA00", width=2, tags="overlay")
            # nazwa w centroidzie
            cx = sum([p[0] for p in zn["pts"]])/len(zn["pts"])
            cy = sum([p[1] for p in zn["pts"]])/len(zn["pts"])
            dcx, dcy = self.img_to_disp(cx, cy)
            self.canvas.create_text(dcx, dcy, text=zn["name"], fill="#FFAA00", tags="overlay")

    def finish(self):
        # auto-save domyślną konfigurację jeśli mamy ścieżkę
        if self.default_cfg_path:
            try:
                ensure_dir(self.default_cfg_path.parent)
                with open(self.default_cfg_path, "w", encoding="utf-8") as f:
                    json.dump({"lines": self.lines, "zones": self.zones}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        self.destroy()

    # === load/save dialog ===
    def load_dialog(self):
        f = filedialog.askopenfilename(title="Wczytaj konfigurację",
                                       filetypes=[("JSON","*.json"),("Wszystkie","*.*")])
        if not f: return
        try:
            self.load_from_json(Path(f))
            self.hint.config(text=f"Wczytano: {Path(f).name}")
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
        self._redraw()

# ===================== APLIKACJA WIDEO =====================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unidrone VIDEO – liczenie przekroczeń linii/stref (YOLO + ByteTrack)")
        self.geometry("1100x860")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        models_default = Path(__file__).parent / MODEL_DIRNAME
        self.weights_path = tk.StringVar(value=str(find_best_weights(models_default) or models_default))

        self.quality = tk.IntVar(value=DEFAULT_QUALITY)
        self.tracker_kind = tk.StringVar(value=DEFAULT_TRACKER) # "bytetrack"|"botsort"

        self.overlay_mode = tk.StringVar(value="centroid")  # "centroid"|"boxes"|"boxes_conf"

        self.model = None; self.names = None; self.class_vars = []
        self.selected_files = []

        # Advanced
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
        }

        # Progress / abort
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = tk.StringVar(value="Gotowe.")
        self.abort_event = threading.Event()
        self.worker_done = threading.Event()
        self.worker_thread = None

        self._build_ui()
        self._autoload_best_model()

    def _build_ui(self):
        frm = tk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        self._row_browse(frm, "Folder z wideo (wejście):", self.input_dir, self.browse_input)
        files_row = tk.Frame(frm); files_row.pack(fill="x", pady=3)
        tk.Button(files_row, text="Wybierz pliki wideo…", command=self.browse_files).pack(side="left")
        self.files_label = tk.Label(files_row, text="— brak —"); self.files_label.pack(side="left", padx=8)
        tk.Button(files_row, text="Wyczyść wybór", command=self.clear_files).pack(side="left", padx=8)

        self._row_browse(frm, "Folder wynikowy (opcjonalnie):", self.output_dir, self.browse_output)
        tk.Label(frm, text="Wyniki zapiszemy do podfolderu 'results/'. Jeśli nie wskażesz, użyjemy folderu wideo.").pack(anchor="w")

        self._row_browse(frm, "Wagi (.pt/.zip):", self.weights_path, self.browse_weights, is_dir=False)

        qf = tk.LabelFrame(frm, text="Jakość (1 = szybciej/słabiej, 5 = ULTRA)")
        qf.pack(fill="x", pady=6)
        sc = tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality, showvalue=True,
                      command=lambda _=None: self._update_preset_label())
        sc.pack(side="left", fill="x", expand=True, padx=6)
        self.preset_label = tk.Label(qf, text=""); self.preset_label.pack(side="left", padx=6)
        self._update_preset_label()

        vis = tk.LabelFrame(frm, text="Wizualizacja (overlay)")
        vis.pack(fill="x", pady=6)
        tk.Radiobutton(vis, text="Centroidy", variable=self.overlay_mode, value="centroid").pack(side="left", padx=6)
        tk.Radiobutton(vis, text="Boksy", variable=self.overlay_mode, value="boxes").pack(side="left", padx=6)
        tk.Radiobutton(vis, text="Boksy + conf", variable=self.overlay_mode, value="boxes_conf").pack(side="left", padx=6)

        tk.Label(frm, text="Tracker:").pack(anchor="w")
        trf = tk.Frame(frm); trf.pack(fill="x")
        tk.Radiobutton(trf, text="ByteTrack", variable=self.tracker_kind, value="bytetrack").pack(side="left", padx=6)
        tk.Radiobutton(trf, text="BoT-SORT", variable=self.tracker_kind, value="botsort").pack(side="left", padx=6)

        self.class_frame = tk.LabelFrame(frm, text="Wybór klas (po wczytaniu wag)")
        self.class_frame.pack(fill="both", expand=True, pady=6)
        self.classes_scroll = ScrollableFrame(self.class_frame, height=280)
        self.classes_scroll.pack(fill="both", expand=True)

        act = tk.Frame(frm); act.pack(fill="x", pady=8)
        self.btn_start = tk.Button(act, text="START", command=self.start); self.btn_start.pack(side="left")
        tk.Button(act, text="Opcje zaawansowane…", command=self.open_advanced).pack(side="left", padx=8)
        self.btn_abort = tk.Button(act, text="ABORT", command=self.abort, state="disabled"); self.btn_abort.pack(side="left", padx=10)

        self.log = tk.Text(frm, height=14); self.log.pack(fill="both", expand=True, pady=(6,2))

        pf = tk.Frame(frm); pf.pack(fill="x", pady=4)
        self.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=self.progress_var)
        self.progressbar.pack(fill="x")
        tk.Label(pf, textvariable=self.progress_label, anchor="w").pack(fill="x")

    def _row_browse(self, parent, label, var, cmd, is_dir=True):
        f = tk.Frame(parent); f.pack(fill="x", pady=3)
        tk.Label(f, text=label, width=26, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(f, text="Wybierz…", command=cmd).pack(side="left")

    def _update_preset_label(self):
        p = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
        self.preset_label.config(text=f"imgsz={p['imgsz']}  conf={p['conf']}  iou={p['iou']}  skip={p['frame_skip']}  buf={p['track_buffer']}  match={p['match_thresh']}  hits={p['min_hits']}")

    # ==== pickers ====
    def browse_input(self):
        d = filedialog.askdirectory(title="Wybierz folder z wideo")
        if d: self.input_dir.set(d)

    def browse_files(self):
        files = filedialog.askopenfilenames(title="Wybierz pliki wideo",
                                            filetypes=[("Wideo","*.mp4 *.mov *.avi *.mkv *.m4v *.wmv")])
        if files:
            self.selected_files = list(files); self.files_label.config(text=f"Wybrano {len(self.selected_files)} plików")
        else:
            self.selected_files = []; self.files_label.config(text="— brak —")

    def clear_files(self):
        self.selected_files = []; self.files_label.config(text="— brak —")

    def browse_output(self):
        d = filedialog.askdirectory(title="Wybierz folder wynikowy")
        if d: self.output_dir.set(d)

    def browse_weights(self):
        initdir = str(Path(self.weights_path.get()).parent) if self.weights_path.get() else str(Path(__file__).parent / MODEL_DIRNAME)
        f = filedialog.askopenfilename(initialdir=initdir, title="Wybierz wagi",
                                       filetypes=[("Wagi",".pt .zip"), ("Wszystkie","*.*")])
        if f:
            self.weights_path.set(f); self.load_model_and_classes()

    def _autoload_best_model(self):
        try:
            wp = self.weights_path.get().strip()
            if not wp or Path(wp).is_dir():
                best = find_best_weights(Path(wp) if wp else (Path(__file__).parent / MODEL_DIRNAME))
                if best: self.weights_path.set(str(best))
            if self.weights_path.get(): self.load_model_and_classes()
        except Exception: pass

    def load_model_and_classes(self):
        try:
            out_base = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else None
            out_dir = ensure_dir((out_base or Path.cwd()) / "results")
            temp_root = ensure_dir(out_dir / "temp"); extract_dir = ensure_dir(temp_root / "extracted_models")

            wp = Path(self.weights_path.get().strip())
            if wp.is_dir():
                best = find_best_weights(wp)
                if not best: raise FileNotFoundError(f"W {wp} brak .pt/.zip")
                wp = best

            pt = resolve_weights_to_pt(wp, extract_dir)
            self._log(f"Wczytuję model: {pt}")
            self.model = YOLO(str(pt)); self.names = self.model.names
            self._populate_classes(self.names); self._log("Wagi i lista klas wczytane.")
        except Exception as e:
            messagebox.showerror("Model", f"Nie można wczytać wag:\n{e}")

    def _populate_classes(self, names):
        # Wyczyść
        container = self.classes_scroll.inner
        for w in container.winfo_children():
            w.destroy()
        self.class_vars.clear()

        # Lista nazw z modelu
        id2name = list(names.values()) if isinstance(names, dict) else list(names)

        # Siatka: 5 kolumn; scroll rozwiązuje resztę
        cols = 5
        for i, nm in enumerate(id2name):
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(container, text=nm, variable=var)
            r, c = divmod(i, cols)
            cb.grid(row=r, column=c, sticky="w", padx=6, pady=4)
            self.class_vars.append((nm, var, i))

    def selected_class_indices(self):
        return [idx for (nm, v, idx) in self.class_vars if v.get()]

    # ==== advanced ====
    def open_advanced(self):
        win = tk.Toplevel(self); win.title("Opcje zaawansowane"); win.geometry("560x540")
        p = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])

        def add_row(lbl, var):
            f = tk.Frame(win); f.pack(fill="x", pady=3)
            tk.Label(f, text=lbl, width=26, anchor="w").pack(side="left")
            e = tk.Entry(f, textvariable=var, width=18); e.pack(side="left"); return e

        base = self.adv_params if self.advanced_override else {**p,
            "line_min_gap": LINE_MIN_GAP_FRAMES_DEFAULT,
            "line_min_sep": LINE_MIN_SEP_PX_DEFAULT,
            "zone_min_gap": ZONE_MIN_GAP_FRAMES_DEFAULT
        }

        v_imgsz = tk.StringVar(value=str(base.get("imgsz","")))
        v_conf  = tk.StringVar(value=str(base.get("conf","")))
        v_iou   = tk.StringVar(value=str(base.get("iou","")))
        v_skip  = tk.StringVar(value=str(base.get("frame_skip","")))
        v_buf   = tk.StringVar(value=str(base.get("track_buffer","")))
        v_match = tk.StringVar(value=str(base.get("match_thresh","")))
        v_hits  = tk.StringVar(value=str(base.get("min_hits","")))
        v_lgap  = tk.StringVar(value=str(base.get("line_min_gap","")))
        v_lsep  = tk.StringVar(value=str(base.get("line_min_sep","")))
        v_zhgap = tk.StringVar(value=str(base.get("zone_min_gap","")))

        add_row("imgsz", v_imgsz)
        add_row("conf", v_conf)
        add_row("iou", v_iou)
        add_row("frame_skip", v_skip)
        add_row("track_buffer", v_buf)
        add_row("match_thresh", v_match)
        add_row("min_hits", v_hits)
        add_row("line_min_gap_frames", v_lgap)
        add_row("line_min_sep_px", v_lsep)
        add_row("zone_min_gap_frames", v_zhgap)

        def apply():
            try:
                cur = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
                def get_or(v, cast, key, default):
                    s = v.get().strip()
                    if s != "": return cast(s)
                    return default if key not in cur else cur[key]
                self.adv_params = {
                    "imgsz": get_or(v_imgsz, int, "imgsz", 960),
                    "conf": get_or(v_conf, float, "conf", 0.6),
                    "iou": get_or(v_iou, float, "iou", 0.5),
                    "frame_skip": get_or(v_skip, int, "frame_skip", 1),
                    "track_buffer": get_or(v_buf, int, "track_buffer", 60),
                    "match_thresh": get_or(v_match, float, "match_thresh", 0.78),
                    "min_hits": get_or(v_hits, int, "min_hits", 2),
                    "line_min_gap": int(v_lgap.get()) if v_lgap.get().strip()!="" else LINE_MIN_GAP_FRAMES_DEFAULT,
                    "line_min_sep": int(v_lsep.get()) if v_lsep.get().strip()!="" else LINE_MIN_SEP_PX_DEFAULT,
                    "zone_min_gap": int(v_zhgap.get()) if v_zhgap.get().strip()!="" else ZONE_MIN_GAP_FRAMES_DEFAULT,
                }
                self.advanced_override = True
                self._log("[ADV] Zastosowano override.")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Adv", str(e))

        def reset():
            self.advanced_override = False
            self._log("[ADV] Przywrócono preset z suwaka.")
            win.destroy()

        tk.Button(win, text="Zastosuj", command=apply).pack(side="left", padx=8, pady=8)
        tk.Button(win, text="Przywróć preset", command=reset).pack(side="left", padx=8, pady=8)

    # ==== ABORT ====
    def abort(self):
        self.abort_event.set()
        self._set_progress(None, "Przerywam…")
        def _wait_and_reset():
            try:
                if self.worker_thread is not None:
                    self.worker_done.wait(timeout=3.0)
            finally:
                self.after(0, lambda: (
                    self.progress_var.set(0.0),
                    self.progress_label.set("Przerwano. Gotowe."),
                    self.btn_start.config(state="normal"),
                    self.btn_abort.config(state="disabled")
                ))
        threading.Thread(target=_wait_and_reset, daemon=True).start()

    # ==== START ====
    def start(self):
        try:
            if self.btn_start['state'] == "disabled": return
            self.abort_event.clear()
            self.btn_start.config(state="disabled")
            self.btn_abort.config(state="normal")
            self._set_progress(0.0, "Przygotowuję…")

            videos = []
            if self.selected_files:
                videos = [Path(p) for p in self.selected_files]
                base_in = videos[0].parent
            else:
                inp = Path(self.input_dir.get().strip())
                if not inp.exists():
                    messagebox.showerror("Wejście", "Wskaż poprawny folder lub pliki.")
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                videos = sorted([p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED_VID_EXTS])
                if not videos:
                    self._log("Brak plików wideo w folderze.")
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                base_in = inp

            if self.model is None:
                self.load_model_and_classes()
                if self.model is None:
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return

            selected_idx = self.selected_class_indices()
            if not selected_idx:
                messagebox.showwarning("Klasy", "Zaznacz co najmniej jedną klasę.")
                self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return

            out_base = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else base_in
            outp = ensure_dir(out_base / "results")
            self.worker_done.clear()
            self.worker_thread = threading.Thread(target=self._run, args=(videos, outp, selected_idx), daemon=True)
            self.worker_thread.start()
        except Exception as e:
            self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")
            messagebox.showerror("Błąd", str(e))

    # ==== GŁÓWNY WORKER ====
    def _run(self, videos, outp: Path, selected_idx):
        t0 = time.time()
        try:
            vids_dir = ensure_dir(outp / "videos")
            ev_dir   = ensure_dir(outp / "events")
            summ_dir = ensure_dir(outp / "summary")
            cnt_dir  = ensure_dir(outp / "counters")
            temp_dir = ensure_dir(outp / "temp")

            # parametry
            p = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY]).copy()
            if self.advanced_override:
                p.update(self.adv_params)
            imgsz = int(p["imgsz"]); conf = float(p["conf"]); iou = float(p["iou"])
            frame_skip = int(p["frame_skip"]); stride = max(1, frame_skip + 1)
            track_buffer = int(p["track_buffer"]); match_thresh = float(p["match_thresh"]); min_hits = int(p["min_hits"])
            line_min_gap = int(p.get("line_min_gap", LINE_MIN_GAP_FRAMES_DEFAULT))
            line_min_sep = int(p.get("line_min_sep", LINE_MIN_SEP_PX_DEFAULT))
            zone_min_gap = int(p.get("zone_min_gap", ZONE_MIN_GAP_FRAMES_DEFAULT))
            tracker_kind = self.tracker_kind.get()

            device = device_auto_str()
            id2name = self.model.names if isinstance(self.model.names, dict) else {i:nm for i,nm in enumerate(self.model.names)}
            select_names = [id2name[i] for i in selected_idx]

            self._log(f"Param: imgsz={imgsz}, conf={conf}, iou={iou}, frame_skip={frame_skip} (stride={stride}), buf={track_buffer}, match={match_thresh}, hits={min_hits}, device={device}")
            self._log(f"Tracker: {tracker_kind} | Klasy: {', '.join(select_names)}")
            self._log(f"Histereza: line_gap={line_min_gap}, line_sep={line_min_sep}px, zone_gap={zone_min_gap}")

            for vi, vid_path in enumerate(videos):
                if self.abort_event.is_set(): break
                self._log(f"► Wideo {vi+1}/{len(videos)}: {vid_path.name}")

                cap = cv2.VideoCapture(str(vid_path))
                if not cap.isOpened():
                    self._log(f"[WARN] Nie można otworzyć: {vid_path}")
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
                fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps>1e-3 else 25.0
                W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

                # pobierz 1. klatkę do edytora konfiguracji liczników
                ret, first_frame = cap.read()
                cap.release()
                if not ret or first_frame is None:
                    self._log(f"[WARN] Brak pierwszej klatki: {vid_path.name}")
                    continue

                default_cfg_path = cnt_dir / f"{vid_path.stem}.json"
                editor = CounterEditor(self, first_frame, default_cfg_path=default_cfg_path)
                self.wait_window(editor)
                lines_cfg = editor.lines[:]   # [{name, a, b}]
                zones_cfg = editor.zones[:]   # [{name, pts}]

                # writer (kolizje nazw); dopasuj FPS do stride (zachowujemy normalną prędkość)
                fps_out = max(1.0, fps / float(stride))
                writer, out_video_path = open_video_writer_collision(vids_dir / f"{vid_path.stem}_annotated.mp4", W, H, fps_out)
                if not writer or not writer.isOpened():
                    self._log(f"[ERR] Nie można otworzyć VideoWriter dla: {vid_path.name}")
                    continue

                # Przygotuj tracker-konfig dla ultralytics (tymczasowy yaml)
                tracker_yaml = (temp_dir / f"{tracker_kind}.yaml")
                with open(tracker_yaml, "w", encoding="utf-8") as f:
                    if tracker_kind == "botsort":
                        f.write(
f"""tracker_type: botsort
track_high_thresh: {conf}
track_low_thresh: {max(conf-0.1, 0.05)}
new_track_thresh: {conf}
track_buffer: {track_buffer}
match_thresh: {match_thresh}
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.25
min_hits: {min_hits}
"""
                        )
                    else:
                        f.write(
f"""tracker_type: bytetrack
track_high_thresh: {conf}
track_low_thresh: {max(conf-0.1, 0.05)}
new_track_thresh: {conf}
track_buffer: {track_buffer}
match_thresh: {match_thresh}
min_box_area: 10
mot20: False
"""
                        )

                # Pętla śledzenia przez Ultralytics (stream=True), z vid_stride
                generator = self.model.track(
                    source=str(vid_path),
                    stream=True,
                    verbose=False,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                    classes=selected_idx,
                    tracker=str(tracker_yaml),
                    persist=True,
                    save=False,
                    vid_stride=stride  # <— przetwarzaj co N-tą klatkę
                )

                # Stany per track
                last_centroid = {}  # tid -> (x,y)
                line_states = [{ } for _ in lines_cfg]
                line_counts = [{"ab":0,"ba":0} for _ in lines_cfg]
                zone_states = [{ } for _ in zones_cfg]
                zone_counts = [{"in":0,"out":0} for _ in zones_cfg]
                events = []

                processed = 0
                start_time = time.time()
                est_total_processed = (total_frames // stride) if total_frames else None

                for res in generator:
                    if self.abort_event.is_set(): break
                    processed += 1
                    # Szacowany indeks klatki ≈ processed*stride
                    frame_idx = processed * stride - 1

                    frame = res.orig_img.copy() if hasattr(res, "orig_img") and res.orig_img is not None else None
                    if frame is None:
                        try: frame = res.plot()
                        except Exception: continue

                    det_boxes, det_confs, det_cids, det_ids = [], [], [], []
                    if res.boxes is not None and len(res.boxes) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        cls = res.boxes.cls.cpu().numpy().astype(int)
                        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.array([-1]*len(xyxy))
                        for b,s,c,tid in zip(xyxy, confs, cls, ids):
                            if tid < 0:  # pomijamy brak ID
                                continue
                            det_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                            det_confs.append(float(s))
                            det_cids.append(int(c))
                            det_ids.append(int(tid))

                    # centroidy
                    centroids = []
                    for b in det_boxes:
                        cx = 0.5*(b[0]+b[2]); cy = 0.5*(b[1]+b[3]); centroids.append((cx,cy))

                    # Linie + Strefy
                    for (tid, b, s, cid, (cx,cy)) in zip(det_ids, det_boxes, det_confs, det_cids, centroids):
                        # Linie
                        for li, ln in enumerate(lines_cfg):
                            a = (ln["a"][0], ln["a"][1]); b2 = (ln["b"][0], ln["b"][1])
                            st = line_states[li].get(tid, {"last_side": None, "last_frame": -9999})
                            prev_side = st["last_side"]
                            cur_side = line_side(a, b2, (cx,cy))
                            crossed = False; direction = None
                            if prev_side is not None:
                                prev_c = last_centroid.get(tid, (cx,cy))
                                if segments_intersect(prev_c, (cx,cy), a, b2):
                                    if prev_side < 0 and cur_side > 0:
                                        direction = "ab"
                                    elif prev_side > 0 and cur_side < 0:
                                        direction = "ba"
                                    if direction is not None:
                                        if frame_idx - st["last_frame"] >= line_min_gap:
                                            if dist_point_to_segment(a, b2, (cx,cy)) >= line_min_sep:
                                                crossed = True
                            st["last_side"] = cur_side
                            if crossed:
                                st["last_frame"] = frame_idx
                                line_states[li][tid] = st
                                line_counts[li][direction] += 1
                                events.append({
                                    "video": str(vid_path.name),
                                    "frame": int(frame_idx),
                                    "time_sec": float(frame_idx / max(1.0, fps)),
                                    "track_id": int(tid),
                                    "class_id": int(cid),
                                    "class_name": self.names[cid] if isinstance(self.names, dict) else self.names[cid],
                                    "event_type": f"line_{direction}",
                                    "counter_name": ln["name"],
                                    "conf": float(s)
                                })
                            else:
                                line_states[li][tid] = st

                        # Strefy
                        for zi, zn in enumerate(zones_cfg):
                            s = zone_states[zi].get(tid, {"inside": False, "last_change": -9999})
                            inside_now = point_in_polygon((cx,cy), zn["pts"])
                            if inside_now != s["inside"]:
                                if frame_idx - s["last_change"] >= zone_min_gap:
                                    s["inside"] = inside_now
                                    s["last_change"] = frame_idx
                                    zone_states[zi][tid] = s
                                    ev = "zone_in" if inside_now else "zone_out"
                                    if inside_now: zone_counts[zi]["in"] += 1
                                    else: zone_counts[zi]["out"] += 1
                                    events.append({
                                        "video": str(vid_path.name),
                                        "frame": int(frame_idx),
                                        "time_sec": float(frame_idx / max(1.0, fps)),
                                        "track_id": int(tid),
                                        "class_id": int(cid),
                                        "class_name": self.names[cid] if isinstance(self.names, dict) else self.names[cid],
                                        "event_type": ev,
                                        "counter_name": zn["name"],
                                        "conf": float(s)
                                    })
                            else:
                                zone_states[zi][tid] = s

                        last_centroid[tid] = (cx,cy)

                    # ====== OVERLAY ======
                    mode = self.overlay_mode.get()
                    if mode == "centroid":
                        for (tid, (cx,cy)) in zip(det_ids, centroids):
                            cv2.circle(frame, (int(cx), int(cy)), 4, (0,255,0), -1, lineType=cv2.LINE_AA)
                            cv2.putText(frame, f"ID {tid}", (int(cx)+6, int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                    elif mode in ("boxes", "boxes_conf"):
                        show_conf = (mode == "boxes_conf")
                        if sv is not None:
                            det = sv.Detections(xyxy=np.array(det_boxes, dtype=np.float32),
                                                confidence=np.array(det_confs, dtype=np.float32),
                                                class_id=np.array(det_cids, dtype=int),
                                                tracker_id=np.array(det_ids, dtype=int))
                            labels = []
                            for s,c,tid in zip(det_confs, det_cids, det_ids):
                                nm = self.names[c] if isinstance(self.names, dict) else self.names[c]
                                labels.append(f"{nm} ID{tid}" + (f" {s:.2f}" if show_conf else ""))
                            frame = sv.BoxAnnotator(thickness=2).annotate(frame, det, labels=labels)
                        else:
                            for (x1,y1,x2,y2), s, c, tid in zip(det_boxes, det_confs, det_cids, det_ids):
                                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                                lbl = f"ID{tid} {(self.names[c] if isinstance(self.names, dict) else self.names[c])}"
                                if show_conf: lbl += f" {s:.2f}"
                                cv2.putText(frame, lbl, (int(x1), max(14, int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

                    # rysuj linie i liczniki
                    for li, ln in enumerate(lines_cfg):
                        a = (int(ln["a"][0]), int(ln["a"][1])); b2 = (int(ln["b"][0]), int(ln["b"][1]))
                        cv2.arrowedLine(frame, a, b2, (0,255,255), 3, tipLength=0.08)
                        cv2.putText(frame, f"{ln['name']}  A->B:{line_counts[li]['ab']}  B->A:{line_counts[li]['ba']}",
                                    (min(a[0],b2[0])+6, min(a[1],b2[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                    # rysuj strefy i liczniki
                    for zi, zn in enumerate(zones_cfg):
                        poly = np.array(zn["pts"], dtype=np.int32)
                        cv2.polylines(frame, [poly], True, (0,165,255), 2)
                        cx = int(np.mean(poly[:,0])); cy = int(np.mean(poly[:,1]))
                        cv2.putText(frame, f"{zn['name']} IN:{zone_counts[zi]['in']} OUT:{zone_counts[zi]['out']}",
                                    (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)

                    writer.write(frame)

                    # progres
                    if est_total_processed:
                        frac = (processed)/max(1, est_total_processed)
                        eta = self._eta(time.time()-start_time, min(1.0, frac))
                        self._set_progress(frac*100.0, f"{vid_path.name} — przetw. {processed}/{est_total_processed} klatek — stride {stride} — ETA {eta}")
                    else:
                        self._set_progress(None, f"{vid_path.name} — przetw. {processed} — stride {stride}")

                writer.release()

                # zapisy zdarzeń i sum
                df = pd.DataFrame(events)
                ev_path = save_csv_collision(df, ev_dir / f"{vid_path.stem}_events.csv")
                summary = {
                    "video": str(vid_path.name),
                    "lines": [{"name": ln["name"], "A_to_B": line_counts[i]["ab"], "B_to_A": line_counts[i]["ba"]} for i,ln in enumerate(lines_cfg)],
                    "zones": [{"name": zn["name"], "IN": zone_counts[i]["in"], "OUT": zone_counts[i]["out"]} for i,zn in enumerate(zones_cfg)],
                    "total_events": int(len(events))
                }
                sum_path = save_json_collision(summary, summ_dir / f"{vid_path.stem}_counts.json")
                self._log(f"Zapisano: {out_video_path.name}, {ev_path.name}, {Path(sum_path).name}")

            # meta
            meta = {
                "started_at": t0, "finished_at": time.time(),
                "params": {
                    "quality": int(self.quality.get()) if not self.advanced_override else "ADV",
                    **(VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])),
                    **(self.adv_params if self.advanced_override else {}),
                    "device_auto": device, "tracker": self.tracker_kind.get(),
                    "overlay_mode": self.overlay_mode.get()
                },
                "selected_classes": [self.names[i] if isinstance(self.names, dict) else self.names[i] for i in selected_idx],
                "output_dir": str(outp)
            }
            save_json_collision(meta, outp / "run_metadata.json")

            if self.abort_event.is_set():
                self._set_progress(None, "Przerwano.")
                self._log("=== PRZERWANO przez użytkownika ===")
            else:
                self._set_progress(100.0, "Gotowe.")
                self._log(f"Zakończono. Wyniki: {outp}")

        except Exception as e:
            self._log(f"[BŁĄD] {e}")
        finally:
            self.worker_done.set()
            self.btn_start.config(state="normal")
            self.btn_abort.config(state="disabled")

    # ==== helpers ====
    def _log(self, msg): 
        try: self.log.insert("end", msg+"\n"); self.log.see("end")
        except Exception: pass

    def _set_progress(self, percent: float|None, text: str):
        def _upd():
            if percent is not None:
                self.progress_var.set(max(0.0, min(100.0, percent)))
            self.progress_label.set(text)
        try: self.after(0, _upd)
        except Exception: pass

    def _eta(self, elapsed_s: float, progress_frac: float) -> str:
        if progress_frac <= 1e-6: return "--:--"
        total = elapsed_s / progress_frac
        remain = max(0.0, total - elapsed_s)
        m = int(remain // 60); s = int(remain % 60)
        return f"{m:02d}:{s:02d}"

# ===================== MAIN =====================
if __name__ == "__main__":
    app = App(); app.mainloop()

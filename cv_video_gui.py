# cv_video_gui.py (GUI components)
# Auto-extracted from cv_video.py
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
    def __init__(self, master, frame_bgr, default_cfg_path: Path|None=None):
        super().__init__(master)
        self.title("Konfiguracja liczników (pierwsza klatka)")
        self.lines = []
        self.zones = []
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
            self.hint.config(text="Dodaj linię: kliknij A, potem B. Backspace cofa, Enter kończy.")
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
        for i,p in enumerate(self.cur_points):
            dx,dy = self.img_to_disp(p[0], p[1])
            self.canvas.create_oval(dx-3, dy-3, dx+3, dy+3, fill="yellow", outline="", tags="overlay")
            if i>0:
                px,py = self.img_to_disp(self.cur_points[i-1][0], self.cur_points[i-1][1])
                self.canvas.create_line(px,py,dx,dy, fill="yellow", width=2, tags="overlay")
        for ln in self.lines:
            a = self.img_to_disp(*ln["a"]); b = self.img_to_disp(*ln["b"])
            self.canvas.create_line(a[0],a[1],b[0],b[1], fill="#00FFFF", width=3, tags="overlay")
            self.canvas.create_text((a[0]+b[0])/2, (a[1]+b[1])/2 - 10, text=ln["name"], fill="#00FFFF", tags="overlay")
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
        self.destroy()

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






class AppUIMixin:
    def build_ui(self):
        # Główny kontener
        frm = tk.Frame(self); frm.pack(fill="both", expand=True, padx=8, pady=6)

        # --- Wejście: folder + (albo) wybór plików ---
        self._row_browse(frm, "Folder z wideo (wejście):", self.input_dir, self.browse_input, is_dir=True)
        f_files = tk.Frame(frm); f_files.pack(fill="x", pady=2)
        tk.Button(f_files, text="Wybierz pliki wideo...", command=self.browse_files).pack(side="left")
        self.files_label = tk.Label(f_files, text="— brak —"); self.files_label.pack(side="left", padx=8)
        tk.Button(f_files, text="Wyczyść wybór", command=self.clear_files).pack(side="left", padx=(8,0))

        # --- Wyjście ---
        self._row_browse(frm, "Folder wynikowy (opcjonalnie):", self.output_dir, self.browse_output, is_dir=True)

        # --- Wagi ---
        self._row_browse(frm, "Wagi (.pt/.zip):", self.weights_path, self.browse_weights, is_dir=False)

        # --- Źródło wejściowe: pliki/kamera/url ---
        srcf = tk.LabelFrame(frm, text="Źródło wejściowe"); srcf.pack(fill="x", pady=4)
        self.src_mode = tk.StringVar(value="files")
        self.cam_index = tk.StringVar(value="0")
        self.url_input = tk.StringVar(value="")
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

        # --- Jakość (preset) ---
        qf = tk.Frame(frm); qf.pack(fill="x", pady=4)
        tk.Label(qf, text="Jakość (=1 szybciej/słabiej, 5 = ULTRA)").pack(side="left")
        tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality,
                 command=lambda *_: self._update_preset_label()).pack(side="left", fill="x", expand=True, padx=8)
        self.preset_label = tk.Label(qf, text=""); self.preset_label.pack(side="left")
        self._update_preset_label()

        # --- Overlay ---
        ov = tk.LabelFrame(frm, text="Wizualizacja (overlay)"); ov.pack(fill="x", pady=4)
        tk.Radiobutton(ov, text="Centroidy", variable=self.overlay_mode, value="centroid").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boksy", variable=self.overlay_mode, value="boxes").pack(side="left", padx=6)
        tk.Radiobutton(ov, text="Boksy + conf", variable=self.overlay_mode, value="boxes_conf").pack(side="left", padx=6)

        # --- Tracker ---
        tr = tk.LabelFrame(frm, text="Tracker:"); tr.pack(fill="x", pady=4)
        tk.Radiobutton(tr, text="ByteTrack", variable=self.tracker_kind, value="bytetrack").pack(side="left", padx=6)
        tk.Radiobutton(tr, text="BoT-SORT", variable=self.tracker_kind, value="botsort").pack(side="left", padx=6)

        # --- Lista klas ---
        lf = tk.LabelFrame(frm, text="Wybór klas (po wczytaniu wag)"); lf.pack(fill="both", expand=True, pady=4)
        self.classes_scroll = ScrollableFrame(lf); self.classes_scroll.pack(fill="both", expand=True)

        # --- Przyciski + postęp ---
        bf = tk.Frame(frm); bf.pack(fill="x", pady=6)
        self.btn_start = tk.Button(bf, text="START", command=self.start); self.btn_start.pack(side="left")
        tk.Button(bf, text="Opcje zaawansowane…", command=self.open_advanced).pack(side="left", padx=8)
        self.btn_abort = tk.Button(bf, text="ABORT", command=self.abort, state="disabled"); self.btn_abort.pack(side="left", padx=(8,0))

        pf = tk.Frame(frm); pf.pack(fill="x", pady=(0,4))
        self.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=self.progress_var, mode="determinate")
        self._progress_indeterminate = False
        self.progressbar.pack(fill="x", side="left", expand=True)
        tk.Label(pf, textvariable=self.progress_label, width=36, anchor="w").pack(side="left", padx=6)

        # --- Podgląd na żywo ---
        pvf = tk.LabelFrame(frm, text="Podgląd na żywo")
        pvf.pack(fill="both", expand=False, pady=(4, 4))
        top = tk.Frame(pvf); top.pack(fill="x")
        self.preview_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="Włącz podgląd", variable=self.preview_enabled).pack(side="left", padx=6)
        self.preview_label = tk.Label(pvf, anchor="center")
        self.preview_label.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Log ---
        logf = tk.Frame(frm); logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, height=10, state="normal"); self.log.pack(fill="both", expand=True)

        self._update_preset_label()


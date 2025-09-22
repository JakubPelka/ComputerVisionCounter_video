# cv_video.py — starter + GUI glue + podgląd w osobnym oknie + Presety (Zapisz/Wczytaj)
from __future__ import annotations
import threading, sys, json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cv_video_gui import ScrollableFrame, CounterEditor, AppUIMixin
from cv_video_run import run as core_run
from cv_video_core import (
    ensure_dir, device_auto_str, open_video_writer_collision,
    save_json_collision, save_csv_collision,
    score_weight_name, find_best_weights, resolve_weights_to_pt,
    SUPPORTED_VID_EXTS, MODEL_DIRNAME,
    VIDEO_PRESETS, DEFAULT_QUALITY, DEFAULT_TRACKER,
    LINE_MIN_GAP_FRAMES_DEFAULT, LINE_MIN_SEP_PX_DEFAULT, ZONE_MIN_GAP_FRAMES_DEFAULT,
)

import cv2, pandas as pd, torch
from PIL import Image, ImageTk
from ultralytics import YOLO
try:
    from ultralytics.nn.modules import block as _ublock
    if not hasattr(_ublock, "C3k2"):
        raise RuntimeError("Ultralytics bez wsparcia YOLOv11 (brak C3k2).")
except Exception as e:
    print("Ultralytics check:", e, file=sys.stderr)


class App(AppUIMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unidrone VIDEO – liczenie przekroczeń linii/stref (YOLO + ByteTrack)")
        self.geometry("1100x860")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        models_default = Path(__file__).parent / MODEL_DIRNAME
        self.weights_path = tk.StringVar(value=str(find_best_weights(models_default) or models_default))

        self.quality = tk.IntVar(value=DEFAULT_QUALITY)
        self.tracker_kind = tk.StringVar(value=DEFAULT_TRACKER)
        self.overlay_mode = tk.StringVar(value="centroid")

        self.model = None; self.names = None; self.class_vars = []
        self.selected_files = []

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

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = tk.StringVar(value="Gotowe.")
        self.abort_event = threading.Event()
        self.worker_done = threading.Event()
        self.worker_thread = None
        self._progress_indeterminate = False  # flaga trybu "indeterminate"

        # osobne okno podglądu
        self._preview_win = None
        self._preview_lbl = None
        self._preview_imgtk = None
        
        self.preview_enabled = tk.BooleanVar(value=True)
        self.trace_enabled   = tk.BooleanVar(value=True)
        self.trace_len       = tk.IntVar(value=24)
        self.anchor_mode     = tk.StringVar(value="center")
        self.ghost_margin    = tk.IntVar(value=12)

        self.build_ui()
        self._autoload_best_model()

    # ---- GUI helpers (używane w mixinie) ----
    def _row_browse(self, parent, label, var, cmd, is_dir=True):
        f = tk.Frame(parent); f.pack(fill="x", pady=3)
        tk.Label(f, text=label, width=26, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(f, text="Wybierz…", command=cmd).pack(side="left")

    def _update_preset_label(self):
        p = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
        self.preset_label.config(text=f"imgsz={p['imgsz']}  conf={p['conf']}  iou={p['iou']}  "
                                      f"skip={p['frame_skip']}  buf={p['track_buffer']}  "
                                      f"match={p['match_thresh']}  hits={p['min_hits']}")

    def browse_input(self):
        d = filedialog.askdirectory(title="Wybierz folder z wideo")
        if d: self.input_dir.set(d)

    def browse_files(self):
        files = filedialog.askopenfilenames(title="Wybierz pliki wideo",
                                            filetypes=[("Wideo","*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.mpg *.mpeg *.ts")])
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
        except Exception:
            pass

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
        container = self.classes_scroll.inner
        for w in container.winfo_children(): w.destroy()
        self.class_vars.clear()
        id2name = list(names.values()) if isinstance(names, dict) else list(names)
        cols = 5
        for i, nm in enumerate(id2name):
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(container, text=nm, variable=var)
            r, c = divmod(i, cols)
            cb.grid(row=r, column=c, sticky="w", padx=6, pady=4)
            self.class_vars.append((nm, var, i))

    def selected_class_indices(self):
        return [idx for (nm, v, idx) in self.class_vars if v.get()]

    # ========= OPCJE ZAAWANSOWANE + PRESET SAVE/LOAD =========
    def open_advanced(self):
        win = tk.Toplevel(self); win.title("Opcje zaawansowane"); win.geometry("620x640")

        # bazowy preset (gdy override wył., to preset z suwaka + domyślne histerezy)
        p = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
        base = self.adv_params if self.advanced_override else {
            **p,
            "line_min_gap": LINE_MIN_GAP_FRAMES_DEFAULT,
            "line_min_sep": LINE_MIN_SEP_PX_DEFAULT,
            "zone_min_gap": ZONE_MIN_GAP_FRAMES_DEFAULT
        }

        def add_row(parent, lbl, var, w=18):
            f = tk.Frame(parent); f.pack(fill="x", pady=4)
            tk.Label(f, text=lbl, width=26, anchor="w").pack(side="left")
            e = tk.Entry(f, textvariable=var, width=w); e.pack(side="left")
            return e

        # pola (StringVar, łatwo zapisywać do JSON)
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

        body = tk.Frame(win); body.pack(fill="both", expand=True, padx=8, pady=8)
        add_row(body, "imgsz", v_imgsz)
        add_row(body, "conf", v_conf)
        add_row(body, "iou", v_iou)
        add_row(body, "frame_skip", v_skip)
        add_row(body, "track_buffer", v_buf)
        add_row(body, "match_thresh", v_match)
        add_row(body, "min_hits", v_hits)
        add_row(body, "line_min_gap_frames", v_lgap)
        add_row(body, "line_min_sep_px", v_lsep)
        add_row(body, "zone_min_gap_frames", v_zhgap)

        def _collect_from_fields() -> dict:
            cur = VIDEO_PRESETS.get(int(self.quality.get()), VIDEO_PRESETS[DEFAULT_QUALITY])
            def get_or(v, cast, key, default):
                s = v.get().strip()
                if s != "":
                    try: return cast(s)
                    except Exception: raise ValueError(f"Pole '{key}' ma nieprawidłową wartość: {s}")
                return default if key not in cur else cur[key]
            return {
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
            }

        PRESETS_DIR = Path(__file__).parent / "presets"
        PRESETS_DIR.mkdir(exist_ok=True)

        def do_save_preset():
            try:
                data = _collect_from_fields()
            except Exception as e:
                messagebox.showerror("Preset", str(e)); return
            defname = f"preset_q{self.quality.get()}.json"
            path = filedialog.asksaveasfilename(
                title="Zapisz preset (JSON)",
                defaultextension=".json",
                initialdir=str(PRESETS_DIR),
                initialfile=defname,
                filetypes=[("JSON","*.json")]
            )
            if not path: return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self._log(f"[ADV] Zapisano preset: {path}")
            except Exception as e:
                messagebox.showerror("Zapisz preset", str(e))

        def do_load_preset():
            path = filedialog.askopenfilename(
                title="Wczytaj preset (JSON)",
                initialdir=str(PRESETS_DIR),
                filetypes=[("JSON","*.json"), ("Wszystkie","*.*")]
            )
            if not path: return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "imgsz" in data: v_imgsz.set(str(data["imgsz"]))
                if "conf" in data: v_conf.set(str(data["conf"]))
                if "iou" in data: v_iou.set(str(data["iou"]))
                if "frame_skip" in data: v_skip.set(str(data["frame_skip"]))
                if "track_buffer" in data: v_buf.set(str(data["track_buffer"]))
                if "match_thresh" in data: v_match.set(str(data["match_thresh"]))
                if "min_hits" in data: v_hits.set(str(data["min_hits"]))
                if "line_min_gap" in data: v_lgap.set(str(data["line_min_gap"]))
                if "line_min_sep" in data: v_lsep.set(str(data["line_min_sep"]))
                if "zone_min_gap" in data: v_zhgap.set(str(data["zone_min_gap"]))
                self._log(f"[ADV] Wczytano preset: {path}")
            except Exception as e:
                messagebox.showerror("Wczytaj preset", str(e))

        def _apply():
            try:
                self.adv_params = _collect_from_fields()
                self.advanced_override = True
                self._log("[ADV] Zastosowano override (z pól/preset).")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Adv", str(e))

        def _reset():
            self.advanced_override = False
            self._log("[ADV] Przywrócono preset z suwaka jakości.")
            win.destroy()

        btns = tk.Frame(win); btns.pack(fill="x", pady=10)
        tk.Button(btns, text="Zastosuj", command=_apply).pack(side="left", padx=6)
        tk.Button(btns, text="Przywróć preset z suwaka", command=_reset).pack(side="left", padx=6)
        tk.Button(btns, text="Zapisz preset…", command=do_save_preset).pack(side="right", padx=6)
        tk.Button(btns, text="Wczytaj preset…", command=do_load_preset).pack(side="right", padx=6)

    # --- sterowanie zadaniem ---
    def abort(self):
        self.abort_event.set()
        self._set_progress(None, "Przerywam…")
        def _wait_and_reset():
            try:
                if self.worker_thread is not None:
                    self.worker_done.wait(timeout=3.0)
            finally:
                def _reset_ui():
                    # zatrzymaj indeterminate i przywróć determinate
                    try:
                        self.progressbar.stop()
                        self.progressbar.config(mode="determinate")
                        self._progress_indeterminate = False
                    except Exception:
                        pass
                    self.progress_var.set(0.0)
                    self.progress_label.set("Przerwano. Gotowe.")
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
            self._set_progress(0.0, "Przygotowuję…")

            sources = []
            base_in = None
            if hasattr(self, "src_mode") and self.src_mode.get() == "camera":
                try: cam_idx = int(self.cam_index.get().strip())
                except Exception:
                    messagebox.showerror("Kamera", "Index kamery musi być liczbą całkowitą.")
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                sources = [cam_idx]
            elif hasattr(self, "src_mode") and self.src_mode.get() == "url":
                url = self.url_input.get().strip()
                if not url:
                    messagebox.showerror("URL", "Podaj RTSP/HTTP URL strumienia.")
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                sources = [url]
            else:
                if self.selected_files:
                    sources = [Path(p) for p in self.selected_files]
                    base_in = sources[0].parent if sources and isinstance(sources[0], Path) else None
                else:
                    inp = Path(self.input_dir.get().strip())
                    if not inp.exists():
                        messagebox.showerror("Wejście", "Wskaż poprawny folder lub pliki.")
                        self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled"); return
                    sources = sorted([p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED_VID_EXTS])
                    if not sources:
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

            out_base = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else (base_in or Path.cwd())
            outp = ensure_dir(out_base / "results")

            self.worker_done.clear()
            self.worker_thread = threading.Thread(target=core_run, args=(self, sources, outp, selected_idx), daemon=True)
            self.worker_thread.start()
        except Exception as e:
            self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")
            messagebox.showerror("Błąd", str(e))

    # --- log / progress ---
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

    # ========= PREVIEW W OSOBNYM OKNIE =========
    def _ensure_preview_window(self):
        if self._preview_win and (self._preview_win.winfo_exists()):
            return
        win = tk.Toplevel(self)
        win.title("Podgląd (LIVE)")
        win.geometry("900x600")
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
                maxw = 880
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

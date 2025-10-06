# convert.py
# Prosty preproces do MP4: najpierw REMUX (bezstratnie), gdy trzeba fallback TRANSCODING (H.264/AAC)
# - GUI (tkinter) lub CLI
# - Wyszukuje ffmpeg w PATH albo w ./bin/ffmpeg(.exe)
# - Zapis do <wyjściowy>/converted/<nazwa>.mp4 (z auto-numeracją, jeśli plik istnieje)

from __future__ import annotations
import sys, os, shutil, subprocess, threading, time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# --------- USTAWIENIA DOMYŚLNE ---------
DEFAULT_OUT_SUBDIR = "converted"
VIDEO_FILTER = [("Wideo", "*.m2ts *.mts *.ts *.mp4 *.mov *.avi *.mkv *.m4v *.wmv"), ("Wszystkie pliki", "*.*")]

# --------- narzędzia ---------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def which_ffmpeg() -> str | None:
    # 1) PATH
    exe = shutil.which("ffmpeg")
    if exe: return exe
    # 2) lokalny portable
    here = Path(__file__).parent
    for cand in ["bin/ffmpeg.exe", "bin/ffmpeg"]:
        p = here / cand
        if p.exists():
            return str(p)
    return None

def numbered_path(path: Path) -> Path:
    """Jeśli plik istnieje, dopisz sufiks _N."""
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1

def run_ffmpeg(cmd: list[str], abort_event: threading.Event|None=None) -> int:
    """Uruchamia ffmpeg i pozwala przerwać (ABORT). Zwraca kod wyjścia."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        while True:
            if proc.poll() is not None:
                return proc.returncode
            if abort_event is not None and abort_event.is_set():
                try:
                    proc.terminate()
                    time.sleep(0.2)
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
                return -9  # aborted
            time.sleep(0.05)
    except Exception:
        return 1

def convert_one(ffmpeg: str, src: Path, out_root: Path, abort_event: threading.Event|None=None, log=print) -> Path|None:
    """Spróbuj REMUX -> fallback TRANSCODE. Zwróć ścieżkę wynikową lub None."""
    out_dir = ensure_dir(out_root / DEFAULT_OUT_SUBDIR)
    out_mp4 = numbered_path(out_dir / (src.stem + ".mp4"))

    # A) REMUX (bezstratnie)
    remux_cmd = [
        ffmpeg, "-y", "-hide_banner",
        "-i", str(src),
        "-map", "0:v:0", "-map", "0:a?",
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_mp4)
    ]
    log(f"[REMUX] {src.name} → {out_mp4.name}")
    rc = run_ffmpeg(remux_cmd, abort_event)
    if rc == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0:
        log("   OK (remux).")
        return out_mp4
    if rc == -9:
        log("   PRZERWANO.")
        return None
    log("   Remux nieudany → transkodowanie…")

    # B) TRANSCODE (H.264/AAC) – pewniejsze, ale wolniejsze
    out_mp4 = numbered_path(out_dir / (src.stem + ".mp4"))  # ponowna kolizja jest możliwa
    trans_cmd = [
        ffmpeg, "-y", "-hide_banner",
        "-i", str(src),
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_mp4)
    ]
    log(f"[XCODE] {src.name} → {out_mp4.name}")
    rc = run_ffmpeg(trans_cmd, abort_event)
    if rc == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0:
        log("   OK (transcode).")
        return out_mp4
    if rc == -9:
        log("   PRZERWANO.")
        return None
    log("   Błąd transkodowania.")
    return None

# --------- CLI ---------
def run_cli(args: list[str]) -> int:
    out_dir = None
    files = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--out" and i+1 < len(args):
            out_dir = Path(args[i+1]); i += 2
        else:
            files.append(Path(a)); i += 1

    if not files:
        print("Użycie: python convert.py <plik1> <plik2> ... [--out D:\\wyniki]")
        return 2

    ffmpeg = which_ffmpeg()
    if not ffmpeg:
        print("Nie znaleziono ffmpeg. Dodaj go do PATH albo umieść w ./bin/ffmpeg(.exe)")
        return 1

    if out_dir is None:
        out_dir = files[0].parent

    abort_event = threading.Event()
    ok, fail = 0, 0
    for p in files:
        if not p.exists():
            print(f"[POMIŃ] Nie ma pliku: {p}")
            continue
        res = convert_one(ffmpeg, p, out_dir, abort_event)
        if res is not None:
            print(f"[OK] {p.name} → {res}")
            ok += 1
        else:
            print(f"[ERR] {p.name}")
            fail += 1

    print(f"Zakończono. Sukces: {ok}, błędy: {fail}.")
    return 0 if fail == 0 else 3

# --------- GUI ---------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Convert to MP4 (ffmpeg) — remux/transcode")
        self.geometry("780x520")

        self.files: list[Path] = []
        self.out_dir = tk.StringVar(value="")
        self.abort_event = threading.Event()
        self.worker = None

        frm = tk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        row1 = tk.Frame(frm); row1.pack(fill="x")
        tk.Button(row1, text="Wybierz pliki…", command=self.pick_files).pack(side="left")
        tk.Button(row1, text="Wyczyść listę", command=self.clear_files).pack(side="left", padx=6)

        self.files_list = tk.Text(frm, height=8)
        self.files_list.pack(fill="both", expand=False, pady=6)

        row2 = tk.Frame(frm); row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Folder wyjściowy (opcjonalnie):", width=28, anchor="w").pack(side="left")
        tk.Entry(row2, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(row2, text="Wybierz…", command=self.pick_out_dir).pack(side="left")

        act = tk.Frame(frm); act.pack(fill="x", pady=8)
        self.btn_start = tk.Button(act, text="START", command=self.start)
        self.btn_start.pack(side="left")
        self.btn_abort = tk.Button(act, text="ABORT", command=self.abort, state="disabled")
        self.btn_abort.pack(side="left", padx=10)

        self.log = tk.Text(frm, height=12)
        self.log.pack(fill="both", expand=True, pady=(6,2))

        pf = tk.Frame(frm); pf.pack(fill="x", pady=4)
        self.pbar = ttk.Progressbar(pf, maximum=100.0)
        self.pbar.pack(fill="x")
        self.plabel = tk.Label(pf, text="Gotowe.")
        self.plabel.pack(anchor="w")

        # sprawdź ffmpeg
        if not which_ffmpeg():
            messagebox.showwarning("FFmpeg", "Nie znaleziono ffmpeg.\nDodaj do PATH lub umieść w ./bin/ffmpeg(.exe) obok skryptu.")

    def pick_files(self):
        files = filedialog.askopenfilenames(title="Wybierz pliki wideo", filetypes=VIDEO_FILTER)
        if not files:
            return
        self.files = [Path(p) for p in files]
        self.files_list.delete("1.0", "end")
        for p in self.files:
            self.files_list.insert("end", str(p) + "\n")

    def clear_files(self):
        self.files = []
        self.files_list.delete("1.0", "end")

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="Wybierz folder wyjściowy")
        if d:
            self.out_dir.set(d)

    def start(self):
        if not self.files:
            messagebox.showwarning("Wejście", "Najpierw wybierz pliki.")
            return
        if self.worker and self.worker.is_alive():
            return
        self.abort_event.clear()
        self.btn_start.config(state="disabled")
        self.btn_abort.config(state="normal")
        self.log.delete("1.0", "end")
        self.pbar["value"] = 0.0
        self.plabel.config(text="Start…")
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def abort(self):
        self.abort_event.set()
        self.plabel.config(text="Przerywam…")

    def _run(self):
        ffmpeg = which_ffmpeg()
        if not ffmpeg:
            self._log("Nie znaleziono ffmpeg. Dodaj do PATH albo umieść w ./bin/ffmpeg(.exe)")
            self._done()
            return

        out_root = Path(self.out_dir.get()) if self.out_dir.get().strip() else self.files[0].parent
        total = len(self.files)
        ok, fail = 0, 0

        for i, src in enumerate(self.files, start=1):
            if self.abort_event.is_set():
                break
            if not src.exists():
                self._log(f"[POMIŃ] Brak pliku: {src}")
                continue

            self._log(f"({i}/{total}) {src.name}")
            res = convert_one(ffmpeg, src, out_root, self.abort_event, log=self._log)
            if self.abort_event.is_set():
                break
            if res is not None:
                self._log(f"  → {res}")
                ok += 1
            else:
                self._log("  → BŁĄD")
                fail += 1

            self._progress(100.0 * i / total, f"Przetworzono {i}/{total}")

        if self.abort_event.is_set():
            self._log("=== PRZERWANO ===")
        else:
            self._log(f"Zakończono. Sukces: {ok}, błędy: {fail}.")
        self._done()

    def _log(self, msg: str):
        try:
            self.log.insert("end", msg + "\n"); self.log.see("end")
        except Exception:
            pass

    def _progress(self, val: float, text: str):
        try:
            self.pbar["value"] = max(0.0, min(100.0, val))
            self.plabel.config(text=text)
        except Exception:
            pass

    def _done(self):
        try:
            self.btn_start.config(state="normal")
            self.btn_abort.config(state="disabled")
            if self.abort_event.is_set():
                self.plabel.config(text="Przerwano.")
            else:
                self.plabel.config(text="Gotowe.")
        except Exception:
            pass

# --------- MAIN ---------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(run_cli(sys.argv[1:]))
    else:
        app = App()
        app.mainloop()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap środowiska do ./_pkgs z bezpieczną wymianą folderu.
- Instalacja do _pkgs_new → atomowa podmiana na _pkgs (koniec PermissionError na .pyd).
- Twarde wersje (CPU-only, Python 3.12): numpy 1.26.4, scipy 1.11.4, OpenCV 4.10.0.84,
  torch 2.3.1+cpu, torchvision 0.18.1+cpu, ultralytics 8.3.201, lap>=0.5.12 itd.
- Najpierw instalujemy numpy 1.26.4, potem bazę, potem torch/vision z --no-deps,
  następnie ultralytics + lap → brak wciągania NumPy 2.x.
- Tryb online (PyPI / CPU index) i offline (z katalogu ./wheels).
- Po sukcesie startuje wskazany skrypt (dev → prod fallback).
"""

import os, sys, subprocess, shutil, socket, time, runpy, argparse
from pathlib import Path

HERE       = Path(__file__).resolve().parent
PKGS_DIR   = HERE / "_pkgs"
PKGS_NEW   = HERE / "_pkgs_new"
PKGS_OLD   = HERE / f"_pkgs_old_{time.strftime('%Y%m%d_%H%M%S')}"
WHEELS_DIR = HERE / "wheels"

# Domyślny target: preferuj _dev, w przeciwnym razie prod
DEFAULT_TARGETS = ["cv_video.py"]

# Piny wersji — stabilny zestaw CPU dla Py 3.12
PIN = {
    "numpy":  "numpy==1.26.4",
    "pillow": "Pillow==10.4.0",
    "opencv": "opencv-python==4.10.0.84",
    "pyyaml": "PyYAML==6.0.1",
    "pandas": "pandas==2.2.2",
    "scipy":  "scipy==1.11.4",
    "ultra":  "ultralytics==8.3.201",
    "lap":    "lap>=0.5.12",
    # zależności torch-a, które damy ręcznie (bo --no-deps)
    "filelock":        "filelock==3.13.1",
    "typing_extensions":"typing_extensions==4.12.2",
    "sympy":           "sympy==1.13.3",
    "networkx":        "networkx==3.3",
    "jinja2":          "Jinja2==3.1.4",
    "fsspec":          "fsspec==2024.6.1",
    "mkl":             "mkl==2021.4.0",
    "intel_openmp":    "intel-openmp==2021.4.0",
    "tbb":             "tbb==2021.11.0",
}

TORCH_CPU   = ("torch==2.3.1+cpu", "torchvision==0.18.1+cpu")
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

def log(msg: str): print(msg, flush=True)

def prefer_local_pkgs_on_sys_path(path: Path):
    p = str(path)
    sys.path[:] = [p] + [x for x in sys.path if x != p]

def is_online(host="pypi.org", port=443, timeout=2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def run_pip(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pip", *args]
    print(">>>", " ".join(cmd), flush=True)
    # streamujemy output na żywo
    return subprocess.call(cmd)

def ensure_dirs():
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    if PKGS_NEW.exists():
        shutil.rmtree(PKGS_NEW, ignore_errors=True)
    PKGS_NEW.mkdir(parents=True, exist_ok=True)

def install_online() -> bool:
    log("[NET] Internet OK → instaluję do ./_pkgs_new")
    ok = True

    # (a) NUMPY 1.26.4 NAJPIERW
    if run_pip(["install","--upgrade","--no-cache-dir","--no-warn-script-location",
                "--target", str(PKGS_NEW), PIN["numpy"]]) != 0:
        return False

    # (b) BAZA zależności zgodnych z numpy 1.26.4
    base = [PIN["pillow"], PIN["opencv"], PIN["pyyaml"], PIN["pandas"], PIN["scipy"]]
    if run_pip(["install","--upgrade","--no-cache-dir","--no-warn-script-location",
                "--target", str(PKGS_NEW), *base]) != 0:
        return False

    # (c) Zależności torch-a ręcznie (bo torch/vision zainstalujemy --no-deps)
    deps = [PIN["filelock"], PIN["typing_extensions"], PIN["sympy"], PIN["networkx"],
            PIN["jinja2"], PIN["fsspec"], PIN["mkl"], PIN["intel_openmp"], PIN["tbb"]]
    if run_pip(["install","--upgrade","--no-cache-dir","--no-warn-script-location",
                "--target", str(PKGS_NEW), *deps]) != 0:
        return False

    # (d) TORCH + VISION z CPU indexem, ale BEZ zależności (żeby nie ruszyło NumPy)
    if run_pip(["install","--upgrade","--no-cache-dir","--no-warn-script-location",
                "--target", str(PKGS_NEW), "--index-url", TORCH_INDEX, "--no-deps", *TORCH_CPU]) != 0:
        return False

    # (e) Ultralytics + lap (do _pkgs_new)
    if run_pip(["install","--upgrade","--no-cache-dir","--no-warn-script-location",
                "--target", str(PKGS_NEW), PIN["ultra"], PIN["lap"]]) != 0:
        return False

    # (f) (opcjonalnie) supervision — odkomentuj, jeśli chcesz mieć od razu
    if run_pip(["install","--upgrade","--no-cache-dir","--no-warn-script-location",
                "--target", str(PKGS_NEW), "supervision>=0.20.0"]) != 0:
         ok = False

    return ok

def install_offline() -> bool:
    log("[OFFLINE] Brak internetu → instaluję z ./wheels do ./_pkgs_new")
    if not any(WHEELS_DIR.glob("*.whl")):
        log("[ERR] Katalog ./wheels jest pusty."); return False

    ok = True
    # (a) NUMPY 1.26.4 najpierw
    if run_pip(["install","--no-index","--find-links", str(WHEELS_DIR),
                "--no-warn-script-location","--target", str(PKGS_NEW), PIN["numpy"]]) != 0:
        return False

    # (b) baza
    base = [PIN["pillow"], PIN["opencv"], PIN["pyyaml"], PIN["pandas"], PIN["scipy"]]
    if run_pip(["install","--no-index","--find-links", str(WHEELS_DIR),
                "--no-warn-script-location","--target", str(PKGS_NEW), *base]) != 0:
        return False

    # (c) deps torch
    deps = [PIN["filelock"], PIN["typing_extensions"], PIN["sympy"], PIN["networkx"],
            PIN["jinja2"], PIN["fsspec"], PIN["mkl"], PIN["intel_openmp"], PIN["tbb"]]
    if run_pip(["install","--no-index","--find-links", str(WHEELS_DIR),
                "--no-warn-script-location","--target", str(PKGS_NEW), *deps]) != 0:
        return False

    # (d) torch/vision z wheels (również bez deps)
    if run_pip(["install","--no-index","--find-links", str(WHEELS_DIR),
                "--no-warn-script-location","--target", str(PKGS_NEW), "--no-deps", *TORCH_CPU]) != 0:
        return False

    # (e) ultralytics + lap
    if run_pip(["install","--no-index","--find-links", str(WHEELS_DIR),
                "--no-warn-script-location","--target", str(PKGS_NEW), PIN["ultra"], PIN["lap"]]) != 0:
        return False

    # (f) supervision (opcjonalnie)
    if run_pip(["install","--no-index","--find-links", str(WHEELS_DIR),
              "--no-warn-script-location","--target", str(PKGS_NEW), "supervision>=0.20.0"]):
        return ok

def module_path(m) -> str:
    return getattr(m, "__file__", "") or ""

def verify_imports_from(pkgs_dir: Path) -> bool:
    prefer_local_pkgs_on_sys_path(pkgs_dir)
    ok = True
    try:
        import torch, torchvision, numpy, cv2
        import ultralytics
        from ultralytics.nn.modules import block as blk
        print(f"[OK] torch {torch.__version__} @ {module_path(torch)}")
        print(f"[OK] torchvision {torchvision.__version__} @ {module_path(torchvision)}")
        print(f"[OK] numpy {numpy.__version__} @ {module_path(numpy)}")
        print(f"[OK] opencv {cv2.__version__} @ {module_path(cv2)}")
        print(f"[OK] ultralytics {ultralytics.__version__} @ {module_path(ultralytics)}")
        if not hasattr(torch, "library") or not hasattr(torch.library, "register_fake"):
            raise RuntimeError("Torch za stary: brak torch.library.register_fake (YOLOv11 tego wymaga).")
        if not hasattr(blk, "C3k2"):
            raise RuntimeError("Ultralytics bez bloku C3k2 (za stare na YOLOv11).")
        # sprawdź, że ładujemy z _pkgs_new
        for m in (torch, torchvision, numpy, cv2, ultralytics):
            p = module_path(m)
            if str(pkgs_dir) not in p:
                print(f"[WARN] {m.__name__} nie importuje się z {pkgs_dir} (sys.path preferuje _pkgs).")
    except Exception as e:
        print("[ERR] Weryfikacja importów:", e); ok = False
    return ok

def swap_pkgs_dirs() -> bool:
    # stary _pkgs → backup (jeśli istnieje), _pkgs_new → _pkgs
    try:
        if PKGS_DIR.exists():
            try:
                PKGS_DIR.rename(PKGS_OLD)
                log(f"[INFO] Stary _pkgs → {PKGS_OLD.name}")
            except Exception as e:
                log(f"[WARN] Rename _pkgs nieudany ({e}) → usuwam stary _pkgs")
                shutil.rmtree(PKGS_DIR, ignore_errors=True)
        PKGS_NEW.rename(PKGS_DIR)
        return True
    except Exception as e:
        log(f"[WARN] Rename _pkgs_new → _pkgs nieudany ({e}) → spróbuję skopiować…")
        try:
            shutil.copytree(PKGS_NEW, PKGS_DIR, dirs_exist_ok=True)
            return True
        except Exception as e2:
            log(f"[ERR] Kopia _pkgs_new → _pkgs nie powiodła się: {e2}")
            return False

def need_install(force: bool) -> bool:
    if force:
        return True
    if not (PKGS_DIR / "ultralytics").exists():
        return True
    prefer_local_pkgs_on_sys_path(PKGS_DIR)
    try:
        import ultralytics
        from ultralytics.nn.modules import block as blk
        if not hasattr(blk, "C3k2"):
            return True
        return False
    except Exception:
        return True

def choose_default_target() -> Path | None:
    for name in DEFAULT_TARGETS:
        p = (HERE / name)
        if p.exists():
            return p
    return None

def main():
    print("=== start.py ===")
    print(f"Python:     {sys.executable}")
    print(f"Project:    {HERE}")
    print(f"PKGS_DIR:   {PKGS_DIR}")
    print(f"WHEELS_DIR: {WHEELS_DIR}")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force", action="store_true", help="wymuś pełną reinstalację do ./_pkgs")
    parser.add_argument("--offline", action="store_true", help="wymuś instalację z ./wheels (bez internetu)")
    args, rest = parser.parse_known_args()

    ensure_dirs()

    if need_install(args.force):
        online = is_online()
        ok = install_offline() if (args.offline or not online) else install_online()
        if not ok:
            print("[ERR] Instalacja do _pkgs_new nie powiodła się.")
            sys.exit(1)
        if not verify_imports_from(PKGS_NEW):
            print("[ERR] Importy z _pkgs_new nieprzechodzą — patrz log powyżej.")
            sys.exit(1)
        if not swap_pkgs_dirs():
            print("[ERR] Podmiana folderów _pkgs_new → _pkgs nie powiodła się.")
            sys.exit(1)
        print("[OK] Środowisko gotowe w ./_pkgs.")
    else:
        print("[INFO] Pakiety już są w ./_pkgs — pomijam instalację.")

    # krótka lista zawartości
    try:
        items = list(PKGS_DIR.iterdir())
        print(f"[INFO] _pkgs zawiera: {len(items)} obiektów (np.: " +
              ", ".join([x.name for x in items[:8]]) + " …)")
    except Exception:
        pass

    # Autostart
    if rest:
        target = Path(rest[0]); target_args = rest[1:]
    else:
        t = choose_default_target()
        if t is None:
            print("\n[INFO] Brak domyślnego skryptu. Użycie:")
            print("  python bootstrap_env.py unidrone_video_dev.py")
            print("  python bootstrap_env.py unidrone_video.py")
            sys.exit(0)
        target = t; target_args = []
        print(f"\n[INFO] Brak argumentów → startuję domyślnie: {target.name}")

    if not target.is_absolute():
        target = (HERE / target).resolve()
    if not target.exists():
        print(f"[ERR] Nie znaleziono: {target}"); sys.exit(2)

    print(f"\n=== Launching {target.name} ===")
    prefer_local_pkgs_on_sys_path(PKGS_DIR)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")  # nie mieszaj user-site
    sys.argv = [str(target), *target_args]
    runpy.run_path(str(target), run_name="__main__")

if __name__ == "__main__":
    main()

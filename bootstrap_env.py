# bootstrap_env.py
# Portable środowisko w ./_pkgs z fallbackiem offline z ./wheels
import sys, os, subprocess, socket, runpy, argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
PKGS_DIR    = PROJECT_DIR / "_pkgs"
WHEELS_DIR  = PROJECT_DIR / "wheels"

DEFAULT_TARGET = "unidrone_video.py"   # auto-start gdy brak argumentów

# --- wersje/pakiety (YOLOv11 + CPU) ---
ULTRA_VER   = "8.3.201"
TORCH_CPU   = ("torch==2.3.1+cpu", "torchvision==0.18.1+cpu")
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PKGS = [
    f"ultralytics=={ULTRA_VER}",
    "supervision>=0.20.0",
    "opencv-python>=4.10.0.0",
    "numpy>=1.26.4",
    "pandas>=2.2.2",
    "Pillow>=10.3.0",
    "PyYAML>=6.0.1",
    "scipy>=1.11.4",
    "onnx>=1.16.0",
    "onnxruntime==1.22.1; platform_system=='Windows'",
]

def log(msg): print(msg, flush=True)

def ensure_dirs():
    PKGS_DIR.mkdir(parents=True, exist_ok=True)
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)

def prefer_local():
    # Wpychamy _pkgs na początek sys.path (nadpisuje site-packages)
    p = str(PKGS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)

def is_online(host="pypi.org", port=443, timeout=2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def pip_run(args: list[str]) -> bool:
    """Uruchamia pip i streamuje output w czasie rzeczywistym."""
    cmd = [sys.executable, "-m", "pip", *args]
    print(">>>", " ".join(cmd), flush=True)
    # Nie buforujemy – paski postępu i logi są widoczne od razu
    return subprocess.call(cmd) == 0

def install_online() -> bool:
    log("[NET] Internet OK → instaluję do ./_pkgs")
    ok = pip_run([
        "install", "--upgrade", "--no-warn-script-location",
        "--default-timeout", "180", "--progress-bar", "on",
        "--target", str(PKGS_DIR),
        "--index-url", TORCH_INDEX, *TORCH_CPU
    ])
    if not ok:
        log("[ERR] torch/torchvision (CPU) — instalacja nie powiodła się.")
        return False
    ok = pip_run([
        "install", "--upgrade", "--no-warn-script-location",
        "--default-timeout", "180", "--progress-bar", "on",
        "--target", str(PKGS_DIR), *PKGS
    ])
    if not ok:
        log("[ERR] instalacja pozostałych pakietów nie powiodła się.")
        return False
    return True

def install_offline() -> bool:
    log("[OFFLINE] Brak internetu → próbuję z ./wheels")
    if not any(WHEELS_DIR.glob("*.whl")):
        log("[ERR] Katalog ./wheels jest pusty.")
        return False
    # Najpierw (jeśli są) koła torch/vision
    pip_run([
        "install", "--no-index", "--find-links", str(WHEELS_DIR),
        "--no-warn-script-location", "--default-timeout", "180", "--progress-bar", "on",
        "--target", str(PKGS_DIR), *TORCH_CPU
    ])
    # Reszta
    return pip_run([
        "install", "--no-index", "--find-links", str(WHEELS_DIR),
        "--no-warn-script-location", "--default-timeout", "180", "--progress-bar", "on",
        "--target", str(PKGS_DIR), *PKGS
    ])

def module_path(m) -> str:
    return getattr(m, "__file__", "") or ""

def verify_imports() -> bool:
    prefer_local()
    ok = True
    try:
        import ultralytics
        from ultralytics.nn.modules import block as blk
        upath = module_path(ultralytics)
        print(f"[OK] ultralytics {ultralytics.__version__} @ {upath}")
        if not hasattr(blk, "C3k2"):
            raise RuntimeError("Ultralytics bez bloku C3k2 (za stara wersja na YOLOv11).")
        if str(PKGS_DIR) not in upath:
            print("[WARN] ultralytics importuje się spoza _pkgs — ale _pkgs jest preferowane w sys.path.")
    except Exception as e:
        print("[ERR] ultralytics:", e); ok = False

    try:
        import torch, torchvision, cv2, numpy, pandas, PIL, yaml, scipy
        print(f"[OK] torch {torch.__version__} @ {module_path(torch)}")
        print(f"[OK] torchvision {torchvision.__version__} @ {module_path(torchvision)}")
        print(f"[OK] opencv {cv2.__version__} @ {module_path(cv2)}")
        print(f"[OK] numpy {numpy.__version__} @ {module_path(numpy)}")
        print(f"[OK] pandas {pandas.__version__} @ {module_path(pandas)}")
        print(f"[OK] Pillow {PIL.__version__} @ {module_path(PIL)}")
        print(f"[OK] PyYAML {yaml.__version__} @ {module_path(yaml)}")
        print(f"[OK] scipy {scipy.__version__} @ {module_path(scipy)}")
    except Exception as e:
        print("[ERR] import pakietów:", e); ok = False

    try:
        mods = list(PKGS_DIR.iterdir())
        print(f"[INFO] _pkgs zawiera: {len(mods)} obiektów (np.: {', '.join([m.name for m in mods[:8]])} …)")
    except Exception:
        pass
    return ok

def need_install(force: bool) -> bool:
    """Instalujemy, jeżeli:
       - wymuszono --force, LUB
       - w _pkgs NIE ma folderu pakietu (np. '_pkgs/ultralytics'), LUB
       - import ultralytics nie wskazuje na _pkgs.
    """
    if force:
        return True
    if not (PKGS_DIR / "ultralytics").exists():
        return True
    prefer_local()
    try:
        import ultralytics
        return str(PKGS_DIR) not in module_path(ultralytics)
    except Exception:
        return True

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force", action="store_true", help="wymuś reinstalację do ./_pkgs")
    parser.add_argument("--offline", action="store_true", help="wymuś tryb offline (instalacja z ./wheels)")
    args, rest = parser.parse_known_args()

    print("=== bootstrap_env.py ===")
    print(f"Python:     {sys.executable}")
    print(f"Project:    {PROJECT_DIR}")
    print(f"PKGS_DIR:   {PKGS_DIR}")
    print(f"WHEELS_DIR: {WHEELS_DIR}")

    ensure_dirs()
    prefer_local()

    if need_install(args.force):
        ok = install_offline() if args.offline or not is_online() else install_online()
        if not ok:
            print("[ERR] Instalacja nie powiodła się — sprawdź internet albo wrzuć koła do ./wheels.")
            sys.exit(1)
    else:
        print("[INFO] Pakiety w _pkgs wykryte — pomijam instalację.")

    if not verify_imports():
        print("[ERR] Importy niepełne/niezgodne — patrz log powyżej.")
        sys.exit(1)

    # wybór skryptu docelowego
    if rest:
        target_path = Path(rest[0]); target_args = rest[1:]
    else:
        default = PROJECT_DIR / DEFAULT_TARGET
        if default.exists():
            target_path = default; target_args = []
            print(f"\n[INFO] Brak argumentów → startuję domyślnie: {DEFAULT_TARGET}")
        else:
            print("\nUżycie: python bootstrap_env.py [--force] [--offline] <skrypt.py> [args...]")
            sys.exit(0)

    if not target_path.is_absolute():
        target_path = (PROJECT_DIR / target_path).resolve()
    if not target_path.exists():
        print(f"[ERR] Nie znaleziono: {target_path}"); sys.exit(2)

    print(f"\n=== Launching {target_path.name} ===")
    prefer_local()
    sys.argv = [str(target_path), *target_args]
    runpy.run_path(str(target_path), run_name="__main__")

if __name__ == "__main__":
    main()

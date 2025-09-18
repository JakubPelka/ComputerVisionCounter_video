# bootstrap_env.py
# Windows-only bootstrapper: installs/updates packages to ./_pkgs
# 1) Try ONLINE (PyPI + official PyTorch CPU index)
# 2) If offline/failure -> OFFLINE from ./wheels
# 3) Verify YOLOv11 support (C3k2); if missing -> force (re)install ultralytics
# 4) Run target app (default: unidrone_app.py; override: pass filename as arg)

from __future__ import annotations
import sys, subprocess, os, socket, runpy, json
from pathlib import Path
from importlib import import_module
from importlib.metadata import version as get_version, PackageNotFoundError

BASE = Path(__file__).parent.resolve()
PKGS_DIR = BASE / "_pkgs"
WHEELS_DIR = BASE / "wheels"

# === target app ===
DEFAULT_APP = "unidrone_app.py"  # możesz uruchomić video: python bootstrap_env.py unidrone_video.py
APP_PATH = BASE / (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_APP)

# === Requirements ===
# Torch/vision z CPU indexu – sprawdzone na Py 3.11/3.12 (CPU)
REQUIREMENTS_TORCH = [
    "torch==2.3.1+cpu",
    "torchvision==0.18.1+cpu",
]
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Reszta z PyPI – UWAGA: ultralytics wersja z obsługą YOLOv11
REQUIREMENTS_REST = [
    "ultralytics>=8.3.0",      # YOLOv11 (m.in. warstwa C3k2)
    "opencv-python>=4.8.0",
    "numpy>=1.26.0,<2.0.0",
    "pandas>=2.1.0",
    "pillow>=10.2.0",
    "tqdm>=4.66.0",
    "supervision>=0.21.0",
    "PyYAML>=6.0",
    "scipy>=1.11.0",
    "onnx>=1.15.0",
    "onnxruntime>=1.17.0",
]

SMOKE_IMPORTS = [
    ("torch", None),
    ("torchvision", None),
    ("ultralytics", None),
    ("cv2", None),
    ("numpy", None),
    ("pandas", None),
    ("PIL", None),
    ("tqdm", None),
    ("yaml", None),
    ("scipy", None),
    ("onnxruntime", None),
]

def is_online(host="pypi.org", port=443, timeout=2.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def add_pkgs_to_syspath():
    if str(PKGS_DIR) not in sys.path:
        sys.path.insert(0, str(PKGS_DIR))

def run_pip(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pip"] + args
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)

def have_all_imports(verbose=True) -> bool:
    add_pkgs_to_syspath()
    ok = True
    for mod, _ in SMOKE_IMPORTS:
        try:
            import_module(mod)
            if verbose:
                try:
                    print(f"[OK] import {mod} ({get_version(mod)})")
                except PackageNotFoundError:
                    print(f"[OK] import {mod}")
        except Exception as e:
            ok = False
            if verbose:
                print(f"[MISS] {mod}: {e}")
    return ok

def ultralytics_supports_yolo11() -> bool:
    """
    YOLOv11 używa m.in. bloku C3k2 w ultralytics.nn.modules.block.
    Jeśli atrybutu brak -> pakiet jest za stary.
    """
    try:
        add_pkgs_to_syspath()
        from ultralytics.nn.modules import block as ublock
        return hasattr(ublock, "C3k2")
    except Exception:
        return False

def install_online() -> bool:
    print("\n=== ONLINE INSTALL (to ./_pkgs) ===")
    PKGS_DIR.mkdir(parents=True, exist_ok=True)

    run_pip(["install", "--upgrade", "pip"])

    rc_t = run_pip([
        "install", "--index-url", PYTORCH_CPU_INDEX, "--target", str(PKGS_DIR),
        *REQUIREMENTS_TORCH
    ])
    if rc_t != 0:
        print("[WARN] Torch CPU install failed (continuing).")

    rc_r = run_pip(["install", "--target", str(PKGS_DIR), *REQUIREMENTS_REST])
    if rc_r != 0:
        print("[WARN] Some packages failed from PyPI.")

    return have_all_imports(verbose=True)

def install_offline_from_wheels() -> bool:
    print("\n=== OFFLINE INSTALL from ./wheels (to ./_pkgs) ===")
    if not WHEELS_DIR.exists():
        print("[ERR] ./wheels directory not found.")
        return False

    PKGS_DIR.mkdir(parents=True, exist_ok=True)

    rc = run_pip([
        "install", "--no-index", "--find-links", str(WHEELS_DIR),
        "--target", str(PKGS_DIR),
        *REQUIREMENTS_TORCH, *REQUIREMENTS_REST
    ])
    if rc != 0:
        print("[WARN] Offline constraints failed, trying all *.whl directly…")
        wheel_files = sorted(str(p) for p in WHEELS_DIR.glob("*.whl"))
        if not wheel_files:
            print("[ERR] No .whl files in ./wheels.")
            return False
        rc2 = run_pip(["install", "--no-index", "--target", str(PKGS_DIR), *wheel_files])
        if rc2 != 0:
            print("[ERR] Offline install from wheel files failed.")
            return False

    return have_all_imports(verbose=True)

def force_update_ultralytics(online: bool) -> bool:
    """
    Wymusza reinstall ultralytics do wersji z YOLOv11.
    """
    print("\n=== Force (re)install ultralytics for YOLOv11 ===")
    # Odinstaluj starą wersję z ./_pkgs (jeśli jest)
    run_pip(["uninstall", "-y", "ultralytics"])

    if online:
        rc = run_pip(["install", "--target", str(PKGS_DIR), "ultralytics>=8.3.0"])
    else:
        if not WHEELS_DIR.exists():
            print("[ERR] wheels/ missing; cannot offline-install ultralytics.")
            return False
        # Spróbuj dopasować najnowsze koło z katalogu wheels
        whls = sorted([p for p in WHEELS_DIR.glob("ultralytics-*.whl")], reverse=True)
        if not whls:
            print("[ERR] No ultralytics wheel in ./wheels.")
            return False
        rc = run_pip(["install", "--no-index", "--target", str(PKGS_DIR), str(whls[0])])

    if rc != 0:
        print("[ERR] ultralytics reinstall failed.")
        return False

    add_pkgs_to_syspath()
    ok = ultralytics_supports_yolo11()
    print(f"[CHK] YOLOv11 support (C3k2): {'OK' if ok else 'NO'}")
    return ok

def main():
    print("=== bootstrap_env.py ===")
    print(f"Python: {sys.executable}")
    print(f"Project: {BASE}")
    print(f"PKGS_DIR: {PKGS_DIR}")
    print(f"WHEELS_DIR: {WHEELS_DIR}")
    print(f"Target app: {APP_PATH.name}\n")

    add_pkgs_to_syspath()

    # 0) Quick check
    have = have_all_imports(verbose=True)
    online = is_online()

    if not have:
        if online:
            print("\n[INFO] Network available -> trying online install…")
            ok = install_online()
            if not ok:
                print("\n[INFO] Online incomplete -> trying offline wheels…")
                ok = install_offline_from_wheels()
        else:
            print("\n[INFO] No network -> trying offline wheels…")
            ok = install_offline_from_wheels()

        if not ok:
            print("\n[FATAL] Could not install all required packages.")
            sys.exit(1)

    # 1) YOLOv11 sanity (C3k2 present?)
    if not ultralytics_supports_yolo11():
        print("[INFO] Current ultralytics seems too old for YOLOv11 (missing C3k2).")
        if not force_update_ultralytics(online=online):
            print("\n[FATAL] ultralytics still misses YOLOv11 support.")
            print(" - If offline, place a recent ultralytics wheel in ./wheels and rerun.")
            sys.exit(1)

    # 2) Launch app
    if not APP_PATH.exists():
        print(f"\n[ERR] {APP_PATH.name} not found next to bootstrap_env.py")
        sys.exit(1)

    print("\n=== Launching", APP_PATH.name, "===\n")
    runpy.run_path(str(APP_PATH), run_name="__main__")

if __name__ == "__main__":
    main()

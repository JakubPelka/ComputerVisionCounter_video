# env_guard.py
# Ensures imports come from local ./_pkgs and that versions are compatible with YOLOv11.
# This module does NOT install packages (that’s handled by the bootstrap). It only verifies
# and raises clear errors if something is wrong.

from __future__ import annotations
import os, sys, importlib
from pathlib import Path
from dataclasses import dataclass

PKGS = Path(__file__).resolve().parent / "_pkgs"

@dataclass
class EnvInfo:
    torch_ver: str
    torch_loc: str
    vision_ver: str
    vision_loc: str
    numpy_ver: str
    numpy_loc: str
    cv_ver: str
    cv_loc: str
    ultra_ver: str
    ultra_loc: str
    lap_ver: str | None
    lap_loc: str | None

def _prefer_local_pkgs(strict_local: bool = True) -> None:
    if PKGS.exists():
        sp = str(PKGS)
        # Put _pkgs at the beginning of sys.path
        sys.path[:] = [sp] + [p for p in sys.path if p != sp]
        if strict_local:
            # Remove global site-packages (avoid mixing with system-installed libs)
            sys.path[:] = [p for p in sys.path if ("site-packages" not in p.lower()) or (sp in p)]

def apply(strict_local: bool = True, need_lap: bool = True) -> EnvInfo:
    """
    Use at the VERY beginning of your script:
        from env_guard import apply as _env_apply
        _env_apply(strict_local=True)

    Returns an EnvInfo with versions/paths.
    """
    if not PKGS.exists():
        raise RuntimeError(
            "Missing ./_pkgs (local packages). Launch via:\n"
            "  python start.py\n"
            "…or prepare _pkgs manually."
        )

    _prefer_local_pkgs(strict_local=strict_local)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

    # Imports from _pkgs
    try:
        torch       = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        numpy       = importlib.import_module("numpy")
        cv2         = importlib.import_module("cv2")
        ultralytics = importlib.import_module("ultralytics")
    except Exception as e:
        raise RuntimeError(
            "Failed to load the core packages from ./_pkgs.\n"
            "Run via start.py or check the contents of _pkgs.\n"
            f"Details: {e}"
        )

    # Consistency: ensure both Torch and TorchVision import from ./_pkgs
    t_loc  = str(Path(torch.__file__).resolve())
    tv_loc = str(Path(torchvision.__file__).resolve())
    if strict_local and (str(PKGS) not in t_loc or str(PKGS) not in tv_loc):
        raise RuntimeError(
            "Detected mixed TORCH/TORCHVISION sources (not importing from ./_pkgs).\n"
            f"torch:       {t_loc}\n"
            f"torchvision: {tv_loc}\n\n"
            "Launch via: python start.py"
        )

    # YOLOv11 requires torch.library.register_fake
    if not hasattr(torch, "library") or not hasattr(torch.library, "register_fake"):
        raise RuntimeError(
            "Torch is too old for YOLOv11 (missing torch.library.register_fake).\n"
            "Run via start.py and let it prepare compatible versions."
        )

    # Ultralytics: must have C3k2 block (YOLOv11)
    try:
        from ultralytics.nn.modules import block as _blk
        if not hasattr(_blk, "C3k2"):
            raise RuntimeError(
                "Ultralytics is too old (missing C3k2 block). Use the bootstrap to update."
            )
    except Exception as e:
        raise RuntimeError(f"Ultralytics verification failed: {e}")

    # lap – required by trackers
    lap_ver, lap_loc = None, None
    if need_lap:
        try:
            lap = importlib.import_module("lap")
            lap_ver = getattr(lap, "__version__", "?")
            lap_loc = str(Path(lap.__file__).resolve())
            if strict_local and str(PKGS) not in lap_loc:
                raise RuntimeError(
                    "'lap' is importing from outside ./_pkgs — Ultralytics may attempt AutoUpdate.\n"
                    "Run via start.py to install 'lap' into _pkgs."
                )
        except Exception as e:
            raise RuntimeError(
                f"Module 'lap' is missing in ./_pkgs (required for tracking). Details: {e}\n"
                "Run via start.py (it will install 'lap') or manually:\n"
                "  py -3.12 -m pip install --target _pkgs lap>=0.5.12"
            )

    return EnvInfo(
        torch_ver = getattr(torch, "__version__", "?"), torch_loc = t_loc,
        vision_ver = getattr(torchvision, "__version__", "?"), vision_loc = tv_loc,
        numpy_ver = getattr(numpy, "__version__", "?"), numpy_loc = str(Path(numpy.__file__).resolve()),
        cv_ver = getattr(cv2, "__version__", "?"), cv_loc = str(Path(cv2.__file__).resolve()),
        ultra_ver = getattr(ultralytics, "__version__", "?"), ultra_loc = str(Path(ultralytics.__file__).resolve()),
        lap_ver = lap_ver, lap_loc = lap_loc
    )

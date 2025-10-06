# env_guard.py
# Pilnuje, by importy szły z lokalnego ./_pkgs i żeby wersje były zgodne z YOLOv11.
# Nie instaluje pakietów (tym zajmuje się bootstrap) — tu tylko weryfikujemy i
# pokazujemy czytelny komunikat, jeśli coś jest nie tak.

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
        # _pkgs na początek sys.path
        sys.path[:] = [sp] + [p for p in sys.path if p != sp]
        if strict_local:
            # usuń globalne site-packages (żeby nic się nie „domieszało”)
            sys.path[:] = [p for p in sys.path if ("site-packages" not in p.lower()) or (sp in p)]

def apply(strict_local: bool = True, need_lap: bool = True) -> EnvInfo:
    """
    Używaj na SAMYM początku skryptu:
        from env_guard import apply as _env_apply
        _env_apply(strict_local=True)
    Zwraca EnvInfo z wersjami/ścieżkami.
    """
    if not PKGS.exists():
        raise RuntimeError(
            "Brak folderu ./_pkgs (lokalne pakiety). Uruchom:\n"
            "  python bootstrap_env.py unidrone_video_dev.py\n"
            "lub przygotuj _pkgs ręcznie."
        )

    _prefer_local_pkgs(strict_local=strict_local)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

    # Importy z _pkgs
    try:
        torch       = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        numpy       = importlib.import_module("numpy")
        cv2         = importlib.import_module("cv2")
        ultralytics = importlib.import_module("ultralytics")
    except Exception as e:
        raise RuntimeError(
            "Nie udało się załadować podstawowych pakietów z ./_pkgs.\n"
            "Uruchom przez bootstrap_env.py lub sprawdź zawartość _pkgs.\n"
            f"Szczegóły: {e}"
        )

    # Spójność: oba z _pkgs
    t_loc  = str(Path(torch.__file__).resolve())
    tv_loc = str(Path(torchvision.__file__).resolve())
    if strict_local and (str(PKGS) not in t_loc or str(PKGS) not in tv_loc):
        raise RuntimeError(
            "Wykryto mieszankę TORCH/TORCHVISION (nie importują się z ./_pkgs).\n"
            f"torch:       {t_loc}\n"
            f"torchvision: {tv_loc}\n\n"
            "Startuj przez: python bootstrap_env.py unidrone_video_dev.py"
        )

    # YOLOv11 wymaga torch.library.register_fake
    if not hasattr(torch, "library") or not hasattr(torch.library, "register_fake"):
        raise RuntimeError(
            "Torch za stary dla YOLOv11 (brak torch.library.register_fake).\n"
            "Uruchom przez bootstrap_env.py i pozwól mu przygotować właściwe wersje."
        )

    # Ultralytics: blok C3k2 (modele YOLOv11)
    try:
        from ultralytics.nn.modules import block as _blk
        if not hasattr(_blk, "C3k2"):
            raise RuntimeError(
                "Zbyt stary Ultralytics (brak bloku C3k2). Użyj bootstrapa, żeby zaktualizować."
            )
    except Exception as e:
        raise RuntimeError(f"Weryfikacja Ultralytics nie powiodła się: {e}")

    # lap – wymagany przez trackery
    lap_ver, lap_loc = None, None
    if need_lap:
        try:
            lap = importlib.import_module("lap")
            lap_ver = getattr(lap, "__version__", "?")
            lap_loc = str(Path(lap.__file__).resolve())
            if strict_local and str(PKGS) not in lap_loc:
                raise RuntimeError(
                    "‘lap’ importuje się spoza ./_pkgs — Ultralytics może próbować AutoUpdate.\n"
                    "Uruchom przez bootstrap_env.py, aby doinstalować ‘lap’ do _pkgs."
                )
        except Exception as e:
            raise RuntimeError(
                f"Brak modułu ‘lap’ w ./_pkgs (wymagany do trackingu). Szczegóły: {e}\n"
                "Uruchom przez bootstrap_env.py (zadba o lap) albo:\n"
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

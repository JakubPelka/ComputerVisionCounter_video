from __future__ import annotations
from pathlib import Path
import sys

# ROOT repo = rodzic folderu 'src'
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Stałe katalogów (w root, tak jak teraz)
INPUTS  = REPO_ROOT / "indata"
OUTPUTS = REPO_ROOT / "output"
MODELS  = REPO_ROOT / "models"
SOUNDS  = REPO_ROOT / "sounds"

PKGS      = REPO_ROOT / "_pkgs"
PKGS_NEW  = REPO_ROOT / "_pkgs_new"
WHEELS    = REPO_ROOT / "wheels"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def add_src_on_sys_path() -> None:
    """Gdybyś odpalał coś z innego miejsca – dopisz <root>/src do sys.path."""
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

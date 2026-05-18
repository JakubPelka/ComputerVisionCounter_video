"""Hotfix for ComputerVisionCounter VIDEO runtime patch.

Fixes:
    NameError: name 'selected_class_ids_arr' is not defined

Run from repository root:
    python tools/fix_selected_class_ids_arr.py

The script is intentionally small and reversible:
- it creates a backup next to src/cv_video_run.py
- it only touches src/cv_video_run.py
- it does not change UI, counting logic, tracking logic, heatmaps, sounds or outputs
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ROOT / "src" / "cv_video_run.py"

USAGE = "mask_keep = np.isin(cids, selected_class_ids_arr)"
INLINE_SAFE = "mask_keep = np.isin(cids, np.fromiter(selected_class_ids_set, dtype=int))"
ANCHOR = "selected_class_ids_set = set(selected_idx or [])"
DEFINITION = (
    "selected_class_ids_arr = "
    "np.fromiter(selected_class_ids_set, dtype=int) "
    "if selected_class_ids_set else None"
)


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _has_real_definition(text: str) -> bool:
    """Return True only if cv_video_run.py itself defines selected_class_ids_arr."""
    return bool(re.search(r"^\s*selected_class_ids_arr\s*=", text, flags=re.M))


def main() -> int:
    if not RUN_PATH.exists():
        print(f"ERROR: {RUN_PATH} not found. Run this script from repository root.")
        return 1

    text = RUN_PATH.read_text(encoding="utf-8")
    original = text

    if USAGE not in text:
        print("No selected_class_ids_arr usage found. Nothing to fix.")
        return 0

    if _has_real_definition(text):
        print("selected_class_ids_arr is already defined in cv_video_run.py. Nothing to fix.")
        return 0

    # Preferred fix: define the precomputed numpy array once after selected_class_ids_set.
    lines = text.splitlines(keepends=True)
    inserted = False
    new_lines: list[str] = []
    for line in lines:
        new_lines.append(line)
        if (not inserted) and (ANCHOR in line):
            indent = _indent_of(line)
            line_end = "\r\n" if line.endswith("\r\n") else "\n"
            new_lines.append(f"{indent}{DEFINITION}{line_end}")
            inserted = True

    if inserted:
        text = "".join(new_lines)
    else:
        # Fallback: restore the previous inline expression.
        # This is slightly less optimized but safe and equivalent to the old behaviour.
        text = text.replace(USAGE, INLINE_SAFE)
        print("WARNING: anchor not found; restored inline selected-class filter instead.")

    if text == original:
        print("No changes applied.")
        return 0

    backup = RUN_PATH.with_suffix(
        RUN_PATH.suffix + f".bak_selected_class_ids_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    backup.write_text(original, encoding="utf-8", newline="")
    RUN_PATH.write_text(text, encoding="utf-8", newline="")

    if inserted:
        print("Fixed: selected_class_ids_arr is now defined after selected_class_ids_set.")
    else:
        print("Fixed: selected-class filtering was restored to the original inline expression.")
    print(f"Backup written: {backup}")
    print("Next: run start.bat and repeat the same short video test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ROOT / "src" / "cv_video_run.py"
METRICS_SRC = ROOT / "src" / "cv_video_zone_metrics.py"
BACKUP_DIR = ROOT / "TEMP" / "patch_backups"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _backup(path: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BACKUP_DIR / f"{path.name}.bak_zone_metrics_{stamp}"
    shutil.copy2(path, target)
    return target


def _copy_metrics_module() -> None:
    # In this patch package the module is already placed under src/ before the tool runs.
    # This check mainly gives a clearer error if the ZIP was only partly copied.
    if not METRICS_SRC.exists():
        raise FileNotFoundError("Missing src/cv_video_zone_metrics.py. Copy the full patch package first.")


def _patch_import(text: str) -> str:
    if "from cv_video_zone_metrics import ZoneMetrics" in text:
        return text
    target = "from cv_video_stats import StatsAggregator\n"
    if target not in text:
        raise RuntimeError("Could not find StatsAggregator import anchor in cv_video_run.py")
    return text.replace(target, target + "from cv_video_zone_metrics import ZoneMetrics\n", 1)


def _patch_initialization(text: str) -> str:
    if "zone_metrics = ZoneMetrics()" in text:
        return text
    pattern = r'(?P<indent>^[ \t]*)zone_counts = \[\{"in": 0, "out": 0\} for _ in zones_cfg\]\s*\n'
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find zone_counts initialization anchor in cv_video_run.py")
    indent = match.group("indent")
    insert = match.group(0) + f"{indent}zone_metrics = ZoneMetrics()\n"
    return text[:match.start()] + insert + text[match.end():]


def _patch_frame_update(text: str) -> str:
    if "zone_metrics.update_frame(" in text:
        return text

    pattern = (
        r'(?P<call>\n\s*sound_player,\s*\n\s*alert_loop,\s*\n(?P<indent>[ \t]*)\)\s*)'
        r'\n(?P=indent)ov = frame\.copy\(\)'
    )
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError("Could not find _update_counts_and_alerts call end anchor in cv_video_run.py")
    indent = match.group("indent")
    block = f"""{match.group('call')}

{indent}try:
{indent}    zone_metrics.update_frame(
{indent}        frame_idx=frame_idx,
{indent}        time_sec=event_time_sec,
{indent}        timecode=timecode_str,
{indent}        clock=clock_str,
{indent}        zones_cfg=zones_cfg,
{indent}        det_ids=det_ids,
{indent}        det_cids=cids,
{indent}        det_confs=scores,
{indent}        anchors=anchors,
{indent}        names=names,
{indent}        point_in_polygon=point_in_polygon,
{indent}    )
{indent}except Exception as e:
{indent}    try:
{indent}        app._log(f"[WARN] Zone metrics update failed: {{e}}")
{indent}    except Exception:
{indent}        pass

{indent}ov = frame.copy()"""
    return text[:match.start()] + block + text[match.end():]


def _patch_save_outputs(text: str) -> str:
    if "_zone_dwell_times.csv" in text and "_zone_class_peaks.csv" in text:
        return text

    pattern = (
        r'(?P<block>\n(?P<indent>[ \t]*)sum_csv_path = save_csv_collision\(sum_csv_df, summ_dir / f"\{base_stem\}_\{run_tag\}_summary\.csv"\)\s*\n'
        r'(?P=indent)app\._log\(f"Saved summary CSV: \{sum_csv_path\}"\)\s*)'
        r'\n(?P=indent)# final heatmap save'
    )
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError("Could not find summary CSV save anchor in cv_video_run.py")
    indent = match.group("indent")
    block = f"""{match.group('block')}

{indent}# Additional per-zone analytics: dwell time and peak concurrent objects.
{indent}try:
{indent}    dwell_rows = zone_metrics.dwell_rows(source=src_name, run_tag=run_tag)
{indent}    peak_rows = zone_metrics.peak_rows(source=src_name, run_tag=run_tag)

{indent}    dwell_df = pd.DataFrame(dwell_rows, columns=ZoneMetrics.DWELL_COLUMNS)
{indent}    peak_df = pd.DataFrame(peak_rows, columns=ZoneMetrics.PEAK_COLUMNS)

{indent}    if zones_cfg:
{indent}        dwell_path = save_csv_collision(dwell_df, summ_dir / f"{{base_stem}}_{{run_tag}}_zone_dwell_times.csv")
{indent}        peak_path = save_csv_collision(peak_df, summ_dir / f"{{base_stem}}_{{run_tag}}_zone_class_peaks.csv")
{indent}        app._log(f"Saved zone dwell metrics CSV: {{dwell_path}}")
{indent}        app._log(f"Saved zone class peak metrics CSV: {{peak_path}}")
{indent}except Exception as e:
{indent}    try:
{indent}        app._log(f"[WARN] Zone metrics save failed: {{e}}")
{indent}    except Exception:
{indent}        pass

{indent}# final heatmap save"""
    return text[:match.start()] + block + text[match.end():]


def main() -> int:
    if not RUN_PATH.exists():
        print(f"[ERROR] Missing {RUN_PATH}")
        return 1
    _copy_metrics_module()
    backup = _backup(RUN_PATH)
    text = _read_text(RUN_PATH)
    original = text
    text = _patch_import(text)
    text = _patch_initialization(text)
    text = _patch_frame_update(text)
    text = _patch_save_outputs(text)
    if text == original:
        print("[OK] No changes needed; patch already seems applied.")
        return 0
    _write_text(RUN_PATH, text)
    print("[OK] Zone metrics patch applied.")
    print(f"[OK] Backup saved to: {backup}")
    print("[NEXT] Run start.bat and test with at least one polygon zone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print("[INFO] Your original file is backed up in TEMP/patch_backups if a backup was already created.")
        raise

# tools/apply_live_events_patch.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_FILE = REPO_ROOT / "src" / "cv_video_run.py"
BACKUP_DIR = REPO_ROOT / "TEMP" / "patch_backups"


def backup(path: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{path.name}.bak_live_events_{stamp}"
    shutil.copy2(path, dst)
    return dst


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        if new in text:
            print(f"[OK] {label}: already patched")
            return text
        raise RuntimeError(f"Could not find patch anchor for: {label}")
    return text.replace(old, new, 1)


def main() -> int:
    if not RUN_FILE.exists():
        print(f"[ERROR] Missing file: {RUN_FILE}")
        return 1

    backup_path = backup(RUN_FILE)
    print(f"[OK] Backup created: {backup_path}")

    text = RUN_FILE.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "from cv_video_zone_metrics import ZoneMetrics\n",
        "from cv_video_zone_metrics import ZoneMetrics\nfrom cv_video_event_log import LiveEventCsvWriter\n",
        "import LiveEventCsvWriter",
    )

    text = replace_once(
        text,
        "            zone_metrics = ZoneMetrics()\n            events = []\n            app._ev_ref = events            # ← expose live events to HUD (per-class lines)\n",
        "            zone_metrics = ZoneMetrics()\n            events = []\n            live_event_writer = LiveEventCsvWriter(ev_dir / f\"{base_stem}_{run_tag}_events_live.csv\")\n            live_ev_i_saved = 0\n            try:\n                app._log(f\"Live events CSV: {live_event_writer.path}\")\n            except Exception:\n                pass\n            app._ev_ref = events            # ← expose live events to HUD (per-class lines)\n",
        "initialize live event writer",
    )

    text = replace_once(
        text,
        "            def _handle_frame(frame, frame_idx):\n                nonlocal cur_time_sec\n",
        "            def _handle_frame(frame, frame_idx):\n                nonlocal cur_time_sec, live_ev_i_saved\n",
        "make live event index nonlocal",
    )

    text = replace_once(
        text,
        "                    alert_loop,\n                )\n\n\n                try:\n                    zone_metrics.update_frame(\n",
        "                    alert_loop,\n                )\n\n                # Append newly created events immediately.\n                # This protects long runs from losing all event records on crash/abort.\n                try:\n                    total_live_events = len(events)\n                    if total_live_events > live_ev_i_saved:\n                        live_event_writer.write_events(events[live_ev_i_saved:total_live_events])\n                        live_ev_i_saved = total_live_events\n                except Exception as e:\n                    try:\n                        app._log(f\"[WARN] Live event CSV write failed: {e}\")\n                    except Exception:\n                        pass\n\n                try:\n                    zone_metrics.update_frame(\n",
        "append live events after counter update",
    )

    RUN_FILE.write_text(text, encoding="utf-8")
    print("[OK] Patched src/cv_video_run.py")
    print("[INFO] Added live CSV event logging while preserving final events CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

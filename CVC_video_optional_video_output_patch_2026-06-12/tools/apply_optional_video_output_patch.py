from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CV_VIDEO = SRC / "cv_video.py"
CV_RUN = SRC / "cv_video_run.py"
BACKUP_DIR = ROOT / "TEMP" / "patch_backups"


def backup(path: Path) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, BACKUP_DIR / f"{path.name}.bak_optional_video_output_{stamp}")


def write_if_changed(path: Path, text: str, original: str) -> bool:
    if text == original:
        print(f"[OK] No changes needed: {path}")
        return False
    backup(path)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[OK] Patched: {path}")
    return True


def patch_cv_video(text: str) -> str:
    original = text

    # 1) Runtime/UI variable, default ON to preserve current behaviour.
    marker = '        self.overlay_mode = tk.StringVar(value="centroid")\n'
    insert = marker + '        self.save_annotated_video = tk.BooleanVar(value=True)\n'
    if 'self.save_annotated_video = tk.BooleanVar' not in text:
        if marker not in text:
            raise RuntimeError("Could not find overlay_mode variable marker in cv_video.py")
        text = text.replace(marker, insert, 1)

    # 2) Keep value in adv_params too, so run summaries/config snapshots can read it consistently.
    marker = '            "overlay_frame_thickness": 2,\n'
    insert = marker + '            "save_annotated_video": True,\n'
    if '"save_annotated_video"' not in text:
        if marker not in text:
            raise RuntimeError("Could not find adv_params insertion marker in cv_video.py")
        text = text.replace(marker, insert, 1)

    # 3) Add a clear checkbox close to output path.
    marker = '        self._row_browse(root, "Output folder (default: ./output):", self.output_dir, self.browse_output, is_dir=True)\n'
    insert = marker + '''        out_opts = tk.Frame(root); out_opts.pack(fill="x", pady=(0, 3))\n        tk.Checkbutton(\n            out_opts,\n            text="Save annotated video (large files)",\n            variable=self.save_annotated_video\n        ).pack(side="left", padx=(26, 0))\n'''
    if 'text="Save annotated video (large files)"' not in text:
        if marker not in text:
            raise RuntimeError("Could not find output row marker in cv_video.py")
        text = text.replace(marker, insert, 1)

    # 4) Copy current checkbox value into adv_params before starting worker.
    marker = '            self.worker_done.clear()\n'
    insert = '''            try:\n                self.adv_params["save_annotated_video"] = bool(self.save_annotated_video.get())\n            except Exception:\n                pass\n\n            self.worker_done.clear()\n'''
    if 'self.adv_params["save_annotated_video"] = bool(self.save_annotated_video.get())' not in text:
        if marker not in text:
            raise RuntimeError("Could not find worker_done marker in cv_video.py")
        text = text.replace(marker, insert, 1)

    return text


def patch_cv_run(text: str) -> str:
    original = text

    # 1) Read UI option once at the start of run(), defaulting to True for backwards compatibility.
    marker = '''        selected_class_ids_set = set(selected_idx or [])  # respected everywhere\n        selected_class_ids_arr = np.fromiter(selected_class_ids_set, dtype=int) if selected_class_ids_set else None\n'''
    insert = '''        save_annotated_video = True\n        try:\n            save_var = getattr(app, "save_annotated_video", None)\n            save_annotated_video = bool(save_var.get()) if save_var is not None else bool(p.get("save_annotated_video", True))\n        except Exception:\n            save_annotated_video = bool(p.get("save_annotated_video", True))\n        try:\n            app._log(f"Output video: {'enabled' if save_annotated_video else 'disabled'}")\n        except Exception:\n            pass\n\n        selected_class_ids_set = set(selected_idx or [])  # respected everywhere\n        selected_class_ids_arr = np.fromiter(selected_class_ids_set, dtype=int) if selected_class_ids_set else None\n'''
    if 'save_annotated_video = True' not in text:
        if marker not in text:
            raise RuntimeError("Could not find selected_class_ids marker in cv_video_run.py")
        text = text.replace(marker, insert, 1)

    # 2) Replace writer setup with optional writer setup.
    pattern = re.compile(
        r'''            # ---------- open annotated video writer with original source FPS ----------\n.*?\n\n            tracker = _make_bytetrack''',
        re.DOTALL,
    )
    repl = '''            # ---------- optionally open annotated video writer with original source FPS ----------\n            # Size: use probed W,H from capture/first_frame (already set above)\n            size_wh = (int(W), int(H))\n\n            # FPS: prefer preserved source FPS that we set after opening cap\n            src_fps = float(getattr(app, "_current_src_fps", fps if "fps" in locals() else 30.0))\n\n            writer = None\n            out_path = None\n\n            if save_annotated_video:\n                try:\n                    # New API: open_video_writer_collision(out_dir, base_name, size_wh, fps, fourcc) -> (path, writer)\n                    # Pass the *root* output dir (helper will create output/videos/).\n                    out_path, writer = open_video_writer_collision(\n                        out_dir=outp,\n                        base_name=base_stem,\n                        size_wh=size_wh,\n                        fps=src_fps,\n                        fourcc="mp4v"\n                    )\n                except TypeError:\n                    # Legacy fallback: open_video_writer_collision(full_path, W, H, fps) -> (writer, out_path)\n                    full_path = (vids_dir / f"{base_stem}_annotated.mp4")\n                    writer, out_path = open_video_writer_collision(str(full_path), size_wh[0], size_wh[1], src_fps)\n\n                # validate writer only when video output is enabled\n                if not writer or not getattr(writer, "isOpened", lambda: False)():\n                    app._log(f"[ERR] Cannot open VideoWriter: {src_name} (fps={src_fps}, size={size_wh})")\n                    cap.release()\n                    continue\n            else:\n                try:\n                    app._log("[INFO] Annotated video saving disabled — CSV/JSON/preview/metrics still run.")\n                except Exception:\n                    pass\n\n            tracker = _make_bytetrack'''
    if '[INFO] Annotated video saving disabled' not in text:
        text, n = pattern.subn(repl, text, count=1)
        if n != 1:
            raise RuntimeError("Could not replace writer setup block in cv_video_run.py")

    # 3) Guard writer.write calls.
    old = '                    _writer_write_safe(writer, _ensure_bgr(ov), first_frame, (W, H), app)\n'
    new = '                    if save_annotated_video and writer is not None:\n                        _writer_write_safe(writer, _ensure_bgr(ov), first_frame, (W, H), app)\n'
    if old in text:
        text = text.replace(old, new, 1)

    old = '                _writer_write_safe(writer, _ensure_bgr(ov), frame, (W, H), app)\n'
    new = '                if save_annotated_video and writer is not None:\n                    _writer_write_safe(writer, _ensure_bgr(ov), frame, (W, H), app)\n'
    if old in text:
        text = text.replace(old, new, 1)

    # 4) Guard writer.release().
    old = '            writer.release()\n'
    new = '            if writer is not None:\n                writer.release()\n'
    if old in text:
        text = text.replace(old, new, 1)

    # 5) Include option in summary JSON.
    marker = '                "duration_s": float(processed / max(fps, 1e-6)),\n'
    insert = marker + '                "save_annotated_video": bool(save_annotated_video),\n                "annotated_video_path": str(out_path) if out_path else "",\n'
    if '"save_annotated_video": bool(save_annotated_video),' not in text:
        if marker not in text:
            raise RuntimeError("Could not find summary duration marker in cv_video_run.py")
        text = text.replace(marker, insert, 1)

    # 6) Include option in summary CSV total row.
    marker = '                    "duration_s": float(processed / max(fps, 1e-6)),\n                    "lines_cfg": len(lines_cfg),\n'
    insert = '                    "duration_s": float(processed / max(fps, 1e-6)),\n                    "save_annotated_video": bool(save_annotated_video),\n                    "annotated_video_path": str(out_path) if out_path else "",\n                    "lines_cfg": len(lines_cfg),\n'
    if '"save_annotated_video": bool(save_annotated_video),' not in text[text.find('sum_rows.append('):]:
        if marker not in text:
            raise RuntimeError("Could not find summary CSV duration marker in cv_video_run.py")
        text = text.replace(marker, insert, 1)

    return text


def main() -> int:
    if not CV_VIDEO.exists() or not CV_RUN.exists():
        print("[ERROR] Run this script from the repository root, after copying tools/ into the repo.", file=sys.stderr)
        return 2

    changed = []
    cv_video_text = CV_VIDEO.read_text(encoding="utf-8")
    new_cv_video = patch_cv_video(cv_video_text)
    if write_if_changed(CV_VIDEO, new_cv_video, cv_video_text):
        changed.append(str(CV_VIDEO.relative_to(ROOT)))

    cv_run_text = CV_RUN.read_text(encoding="utf-8")
    new_cv_run = patch_cv_run(cv_run_text)
    if write_if_changed(CV_RUN, new_cv_run, cv_run_text):
        changed.append(str(CV_RUN.relative_to(ROOT)))

    print("\nPatch complete.")
    if changed:
        print("Changed files:")
        for p in changed:
            print(f"  - {p}")
    else:
        print("No files changed; patch may already be applied.")
    print("\nBackup files are in TEMP/patch_backups/. Do not commit TEMP/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

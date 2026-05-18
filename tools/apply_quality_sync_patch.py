from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BACKUP_DIR = ROOT / "TEMP" / "patch_backups"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def backup(path: Path) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP_DIR / f"{path.name}.bak_quality_sync_{STAMP}")


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def regex_replace_once(text: str, pattern: str, repl: str, label: str) -> tuple[str, bool]:
    new_text, count = re.subn(pattern, repl, text, count=1, flags=re.DOTALL | re.MULTILINE)
    return new_text, count == 1


def patch_cv_video() -> list[str]:
    path = SRC / "cv_video.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    text = read(path)
    notes: list[str] = []

    if "def _sync_quality_preset_to_adv" in text:
        notes.append("cv_video.py already contains quality sync helper; skipped helper insertion.")
        return notes

    backup(path)

    # 1) Route the Quality slider through a sync-aware handler.
    old = 'tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality, command=lambda *_: self._update_preset_label()).pack(side="left", fill="x", expand=True, padx=8)'
    new = 'tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality, command=self._on_quality_slider_changed).pack(side="left", fill="x", expand=True, padx=8)'
    text2, ok = replace_once(text, old, new, "quality slider command")
    if not ok:
        pattern = r'tk\.Scale\(qf,\s*from_=1,\s*to=5,\s*orient="horizontal",\s*variable=self\.quality,\s*command=lambda \*_: self\._update_preset_label\(\)\)\.pack\('
        repl = 'tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality, command=self._on_quality_slider_changed).pack('
        text2, ok = regex_replace_once(text, pattern, repl, "quality slider command regex")
    if not ok:
        raise RuntimeError("Could not patch Quality slider command in cv_video.py")
    text = text2
    notes.append("cv_video.py: Quality slider now calls _on_quality_slider_changed().")

    # 2) Replace the preset label helper block with sync-aware helpers.
    helper_block = '''    def _current_quality_preset(self) -> dict:
        """Return the currently selected main Quality preset as a plain dict."""
        try:
            q = int(self.quality.get())
        except Exception:
            q = DEFAULT_QUALITY
        return dict(VIDEO_PRESETS.get(q, VIDEO_PRESETS[DEFAULT_QUALITY]))

    def _sync_quality_preset_to_adv(self, *, mark_manual_override: bool = False) -> dict:
        """Copy the main Quality preset into runtime advanced params and open Advanced fields.

        The main Quality slider is treated as the source of truth when the user moves it.
        Manual Advanced values are preserved until the slider is moved again.
        """
        p = self._current_quality_preset()
        for k in ("imgsz", "conf", "iou", "frame_skip", "track_buffer", "match_thresh", "min_hits"):
            if k in p:
                self.adv_params[k] = p[k]

        # False means: use main Quality preset as the current source.
        # True is set by Advanced Apply/Load and means: preserve manual Advanced values.
        self.advanced_override = bool(mark_manual_override)

        # If the Advanced window is currently open, update its fields immediately.
        var_map = {
            "imgsz": "v_imgsz",
            "conf": "v_conf",
            "iou": "v_iou",
            "frame_skip": "v_frame_skip",
            "track_buffer": "v_track_buffer",
            "match_thresh": "v_match_thresh",
            "min_hits": "v_min_hits",
        }
        for key, attr in var_map.items():
            var = getattr(self, attr, None)
            if var is not None and hasattr(var, "set") and key in p:
                try:
                    var.set(str(p[key]))
                except Exception:
                    pass
        return p

    def _on_quality_slider_changed(self, *_):
        """Handle main Quality slider changes and keep Advanced values in sync."""
        self._sync_quality_preset_to_adv(mark_manual_override=False)
        self._update_preset_label()

    def _update_preset_label(self):
        p = self._current_quality_preset()
        try:
            self.preset_label.config(text=(f"imgsz={p['imgsz']} conf={p['conf']} iou={p['iou']} "
                                           f"skip={p['frame_skip']} buf={p['track_buffer']} "
                                           f"match={p['match_thresh']} hits={p['min_hits']}"))
        except Exception:
            pass

'''
    pattern = r'^    def _update_preset_label\(self\):\n.*?^    def _toggle_all_classes\(self\):'
    repl = helper_block + '    def _toggle_all_classes(self):'
    text2, ok = regex_replace_once(text, pattern, repl, "preset helper block")
    if not ok:
        raise RuntimeError("Could not replace _update_preset_label block in cv_video.py")
    text = text2
    notes.append("cv_video.py: added sync-aware quality preset helpers.")

    write(path, text)
    return notes


def patch_advanced_ui() -> list[str]:
    path = SRC / "cv_video_advanced_ui.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    text = read(path)
    notes: list[str] = []

    if "def _apply_settings" in text and "advanced_override" in text:
        notes.append("cv_video_advanced_ui.py already contains Advanced override handling; skipped.")
        return notes

    backup(path)

    # 1) Do not overwrite manual Advanced values from the main Quality label when an override is active.
    old = '''    q = _extract_quality_from_main(app)
    for k, v in q.items():
        if k in defaults and v is not None:
            defaults[k] = v
'''
    new = '''    if not bool(getattr(app, "advanced_override", False)):
        q = _extract_quality_from_main(app)
        for k, v in q.items():
            if k in defaults and v is not None:
                defaults[k] = v
'''
    text2, ok = replace_once(text, old, new, "main quality extraction guard")
    if not ok:
        pattern = r'^    q = _extract_quality_from_main\(app\)\n    for k, v in q\.items\(\):\n        if k in defaults and v is not None:\n            defaults\[k\] = v\n'
        text2, ok = regex_replace_once(text, pattern, new, "main quality extraction guard regex")
    if not ok:
        raise RuntimeError("Could not patch main quality extraction guard in cv_video_advanced_ui.py")
    text = text2
    notes.append("cv_video_advanced_ui.py: Advanced values are no longer overwritten when manual override is active.")

    # 2) Make Apply explicit and set advanced_override=True.
    old = '    ttk.Button(bar, text="Apply", command=lambda: app.adv_params.update(_collect())).pack(side="left")'
    new = '''    def _apply_settings():
        data = _collect()
        app.adv_params.update(data)
        app.advanced_override = True
        try:
            if hasattr(app, "_log"):
                app._log("[ADV] Advanced settings applied.")
        except Exception:
            pass

    ttk.Button(bar, text="Apply", command=_apply_settings).pack(side="left")'''
    text2, ok = replace_once(text, old, new, "Apply button")
    if not ok:
        pattern = r'^    ttk\.Button\(bar, text="Apply", command=lambda: app\.adv_params\.update\(_collect\(\)\)\)\.pack\(side="left"\)'
        text2, ok = regex_replace_once(text, pattern, new, "Apply button regex")
    if not ok:
        raise RuntimeError("Could not patch Apply button in cv_video_advanced_ui.py")
    text = text2
    notes.append("cv_video_advanced_ui.py: Apply now marks Advanced values as the active source.")

    # 3) Loading a preset should also mark Advanced as active.
    old = '    app.adv_params.update(_collect())'
    new = '    _apply_settings()'
    text2, ok = replace_once(text, old, new, "Load preset apply")
    if not ok:
        raise RuntimeError("Could not patch preset load apply behavior in cv_video_advanced_ui.py")
    text = text2
    notes.append("cv_video_advanced_ui.py: loaded presets are applied through the same Advanced Apply path.")

    write(path, text)
    return notes


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Run this script from repository root or keep it in tools/. Missing: {SRC}")

    all_notes: list[str] = []
    all_notes.extend(patch_cv_video())
    all_notes.extend(patch_advanced_ui())

    print("Quality / Advanced sync patch applied.")
    print(f"Backups saved in: {BACKUP_DIR.relative_to(ROOT)}")
    for note in all_notes:
        print("-", note)
    print("\nNext test: start.bat → move Quality slider → open Advanced → verify imgsz/conf/iou/skip/buf/match/hits.")


if __name__ == "__main__":
    main()

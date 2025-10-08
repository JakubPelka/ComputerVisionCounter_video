# cv_video_run.py — custom alert sound (cross-platform), uses selected classes, freeze in seconds
from __future__ import annotations
from pathlib import Path
import time, re as _re, threading, subprocess, sys, os, shutil, signal
import cv2, numpy as np, pandas as pd
from collections import deque

from cv_video_gui import CounterEditor
from cv_video_overlay import draw_detections
from cv_video_core import (
    ensure_dir, device_auto_str, open_video_writer_collision,
    save_json_collision, save_csv_collision,
    VIDEO_PRESETS, DEFAULT_QUALITY,
    LINE_MIN_GAP_FRAMES_DEFAULT, ZONE_MIN_GAP_FRAMES_DEFAULT,
)

try:
    import supervision as sv
except Exception:
    sv = None

# ---------- Geometry ----------
def line_side(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    return (bx-ax)*(py-ay) - (by-ay)*(px-ax)
    

def _get_line_pts(ln: dict):
    """Return a list of points [(x,y),...] for a line (straight or polyline)."""
    if "pts" in ln and len(ln["pts"]) >= 2:
        return [(float(x), float(y)) for x, y in ln["pts"]]
    return [(float(ln["a"][0]), float(ln["a"][1])), (float(ln["b"][0]), float(ln["b"][1]))]

def _point_to_segment_dist2(a, b, p):
    """Squared distance from point p to segment a-b."""
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = bx-ax, by-ay
    wx, wy = px-ax, py-ay
    vv = vx*vx + vy*vy
    t = 0.0 if vv == 0 else max(0.0, min(1.0, (wx*vx + wy*vy)/vv))
    cx, cy = ax + t*vx, ay + t*vy
    dx, dy = px-cx, py-cy
    return dx*dx + dy*dy

def _polyline_side(pts, p):
    """Signed side using the *nearest* segment of the polyline to point p."""
    best_i = 0; best_d = 1e30
    for i in range(len(pts)-1):
        d2 = _point_to_segment_dist2(pts[i], pts[i+1], p)
        if d2 < best_d:
            best_d = d2; best_i = i
    a, b = pts[best_i], pts[best_i+1]
    return line_side(a, b, p)

def _polyline_cross_direction(prev_p, cur_p, pts):
    """
    If the motion segment prev_p->cur_p intersects ANY polyline segment,
    return 'ab' or 'ba' using that segment's orientation; otherwise None.
    """
    for i in range(len(pts)-1):
        a, b = pts[i], pts[i+1]
        if segments_intersect(prev_p, cur_p, a, b):
            ps = line_side(a, b, prev_p)
            cs = line_side(a, b, cur_p)
            if ps < 0 and cs > 0: return "ab"
            if ps > 0 and cs < 0: return "ba"
    return None


def segments_intersect(p1, p2, q1, q2):
    def _orient(a,b,c):
        v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        return 1 if v>0 else (-1 if v<0 else 0)
    def _on(a,b,c):
        return (min(a[0],b[0]) - 1e-6 <= c[0] <= max(a[0],b[0]) + 1e-6 and
                min(a[1],b[1]) - 1e-6 <= c[1] <= max(a[1],b[1]) + 1e-6)
    o1 = _orient(p1,p2,q1); o2 = _orient(p1,p2,q2)
    o3 = _orient(q1,q2,p1); o4 = _orient(q1,q2,p2)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and _on(p1,p2,q1): return True
    if o2 == 0 and _on(p1,p2,q2): return True
    if o3 == 0 and _on(q1,q2,p1): return True
    if o4 == 0 and _on(q1,q2,p2): return True
    return False

def point_in_polygon(p, poly):
    x, y = p; inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1)
        if cond: inside = not inside
    return inside

# ---------- Utils ----------
_URL_RE = _re.compile(r'^\s*(rtsp|rtsps|rtmp|http|https)://', flags=_re.I)
def _is_stream_source(src):
    if isinstance(src, (int,)): return True
    if isinstance(src, str) and _URL_RE.match(src): return True
    return False

def _ensure_bgr(img):
    if img is None:
        return img
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _preview(app, frame_bgr, frame_idx, fps, total_frames):
    try:
        if hasattr(app, "_update_preview"):
            app._update_preview(frame_bgr, frame_idx, fps, total_frames); return
        if hasattr(app, "_show_preview_bgr"):
            app._show_preview_bgr(frame_bgr); return
        if hasattr(app, "update_preview"):
            app.update_preview(frame_bgr, frame_idx, fps, total_frames); return
        if hasattr(app, "show_preview"):
            app.show_preview(frame_bgr); return
    except Exception:
        pass

def _parse_color(val):
    if val is None: return None
    if isinstance(val, (tuple, list)) and len(val) == 3:
        b,g,r = val; return (int(b), int(g), int(r))
    s = str(val).strip()
    if not s or s.lower() == "auto":
        return None
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16); g = int(s[3:5], 16); b = int(s[5:7], 16)
        return (b, g, r)
    try:
        parts = [int(x.strip()) for x in s.replace(";",",").split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    except Exception:
        pass
    return None

def _fmt_timecode(sec: float) -> str:
    if sec < 0: sec = 0.0
    s = int(round(sec))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# ---------- Tracker ----------
def _make_bytetrack(conf: float, track_buffer: int, match_thresh: float, min_hits: int):
    if sv is None:
        return None
    import inspect
    try:
        params = inspect.signature(sv.ByteTrack.__init__).parameters
    except Exception:
        try:
            return sv.ByteTrack()
        except Exception:
            return None
    kwargs = {}
    if "track_thresh" in params: kwargs["track_thresh"] = max(0.05, min(conf, 0.99))
    if "track_buffer" in params: kwargs["track_buffer"] = int(track_buffer)
    if "match_thresh" in params: kwargs["match_thresh"] = float(match_thresh)
    if "min_hits" in params: kwargs["min_hits"] = int(min_hits)
    if "mot20" in params: kwargs["mot20"] = False
    try:
        return sv.ByteTrack(**kwargs)
    except TypeError:
        try:
            return sv.ByteTrack()
        except Exception:
            return None

def _make_botsort(conf: float, track_buffer: int, match_thresh: float, min_hits: int):
    if sv is None:
        return None
    import inspect
    cand_names = ["BoTSORT", "BOTSORT", "BoTSort"]
    BotCls = None
    for nm in cand_names:
        BotCls = getattr(sv, nm, None)
        if BotCls is not None:
            break
    if BotCls is None:
        return None
    try:
        params = inspect.signature(BotCls.__init__).parameters
    except Exception:
        try:
            return BotCls()
        except Exception:
            return None
    kwargs = {}
    if "track_thresh" in params: kwargs["track_thresh"] = max(0.05, min(conf, 0.99))
    if "track_buffer" in params: kwargs["track_buffer"] = int(track_buffer)
    if "match_thresh" in params: kwargs["match_thresh"] = float(match_thresh)
    if "min_hits" in params: kwargs["min_hits"] = int(min_hits)
    if "mot20" in params: kwargs["mot20"] = False
    try:
        return BotCls(**kwargs)
    except TypeError:
        try:
            return BotCls()
        except Exception:
            return None

def _make_tracker(kind: str, conf: float, track_buffer: int, match_thresh: float, min_hits: int):
    k = (kind or "").strip().lower()
    if k in ("botsort", "bot-sort", "bot", "bts"):
        tr = _make_botsort(conf, track_buffer, match_thresh, min_hits)
        if tr is not None:
            return "BoT-SORT", tr
        bt = _make_bytetrack(conf, track_buffer, match_thresh, min_hits)
        return "ByteTrack (fallback)", bt
    return "ByteTrack", _make_bytetrack(conf, track_buffer, match_thresh, min_hits)

# ---------- Sound player (cross-platform, incl. robust Windows) ----------
class SoundPlayer:
    """
    Plays a custom sound file (wav/mp3/ogg/…).
    Backends (in order):
      - ffplay (best; supports stream_loop)
      - macOS: afplay
      - Linux: paplay/aplay (WAV likely)
      - Windows: winsound (WAV, async + loop)  ⟵ robust & built-in
      - Windows: PowerShell (WAV via SoundPlayer; MP3 via WMP COM)
      - simpleaudio (WAV only, pure Python)
    """
    def __init__(self, path: str | None):
        self.path = str(path) if path else None
        self.proc: subprocess.Popen | None = None
        self.play_obj = None  # simpleaudio
        self.loop_thread: threading.Thread | None = None
        self._loop_stop = threading.Event()
        self._lock = threading.Lock()
        self._ffplay = shutil.which("ffplay")
        self._afplay = shutil.which("afplay") if sys.platform == "darwin" else None
        self._paplay = shutil.which("paplay") if sys.platform.startswith("linux") else None
        self._aplay  = shutil.which("aplay")  if sys.platform.startswith("linux") else None
        self._is_win = sys.platform.startswith("win")
        try:
            from subprocess import CREATE_NEW_PROCESS_GROUP as _CF
            self._win_createflags = _CF
        except Exception:
            self._win_createflags = 0
        # winsound (Windows)
        self._winsound = None
        if self._is_win:
            try:
                import winsound  # type: ignore
                self._winsound = winsound
            except Exception:
                self._winsound = None
        # simpleaudio optional
        self._sa = None
        try:
            import simpleaudio as sa  # type: ignore
            self._sa = sa
        except Exception:
            self._sa = None

    def describe_backends(self) -> str:
        backs = []
        if self._ffplay: backs.append("ffplay")
        if self._afplay: backs.append("afplay")
        if self._paplay: backs.append("paplay")
        if self._aplay:  backs.append("aplay")
        if self._winsound: backs.append("winsound")
        backs.append("PowerShell" if self._is_win else "-")
        if self._sa: backs.append("simpleaudio")
        return ", ".join(b for b in backs if b and b != "-") or "none"

    def _stop_proc(self):
        if self.proc and self.proc.poll() is None:
            try:
                if self._is_win:
                    self.proc.terminate()
                else:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
        self.proc = None

    def stop(self):
        with self._lock:
            self._loop_stop.set()
            # stop winsound loop if used
            if self._winsound:
                try:
                    self._winsound.PlaySound(None, self._winsound.SND_PURGE)
                except Exception:
                    pass
            if self.loop_thread and self.loop_thread.is_alive():
                self._stop_proc()
            self.loop_thread = None
            if self.play_obj:
                try: self.play_obj.stop()
                except Exception: pass
                self.play_obj = None
            self._stop_proc()

    def _spawn_once(self):
        """Non-blocking play once via best available backend."""
        if not self.path: return
        # Preferred: ffplay (broad codecs)
        if self._ffplay:
            try:
                self.proc = subprocess.Popen(
                    [self._ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", self.path],
                    preexec_fn=(os.setsid if not self._is_win else None),
                    creationflags=(self._win_createflags if self._is_win else 0),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ); return
            except Exception:
                self.proc = None
        # macOS afplay
        if self._afplay:
            try:
                self.proc = subprocess.Popen(
                    [self._afplay, self.path],
                    preexec_fn=(os.setsid if not self._is_win else None),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ); return
            except Exception:
                self.proc = None
        # Linux paplay/aplay (WAV likely)
        for cmd in (self._paplay, self._aplay):
            if cmd:
                try:
                    self.proc = subprocess.Popen(
                        [cmd, self.path],
                        preexec_fn=(os.setsid if not self._is_win else None),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    ); return
                except Exception:
                    self.proc = None
        # Windows: winsound (WAV)
        if self._winsound and self.path.lower().endswith(".wav"):
            try:
                self._winsound.PlaySound(self.path, self._winsound.SND_FILENAME | self._winsound.SND_ASYNC)
                return
            except Exception:
                pass
        # Windows PowerShell backends (WAV/MP3/etc.)
        if self._is_win:
            ps_path = self.path.replace("'", "''")
            if self.path.lower().endswith(".wav"):
                script = f"[System.Reflection.Assembly]::LoadWithPartialName('System.Media') | Out-Null; " \
                         f"$p = New-Object System.Media.SoundPlayer('{ps_path}'); $p.PlaySync()"
            else:
                script = f"$w = New-Object -ComObject WMPlayer.OCX; " \
                         f"$m = $w.newMedia('{ps_path}'); $w.URL = '{ps_path}'; " \
                         f"$w.controls.play(); while ($w.playState -ne 1) {{ Start-Sleep -Milliseconds 100 }}; $w.close()"
            try:
                self.proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", script],
                    creationflags=self._win_createflags,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ); return
            except Exception:
                self.proc = None
        # simpleaudio (WAV only)
        if self._sa and self.path.lower().endswith(".wav"):
            try:
                wave_obj = self._sa.WaveObject.from_wave_file(self.path)
                self.play_obj = wave_obj.play()
                return
            except Exception:
                self.play_obj = None

    def play_once(self):
        self.stop()
        self._spawn_once()

    def start_loop(self):
        # winsound loop (Windows, WAV) is perfect
        if self._winsound and self.path and self.path.lower().endswith(".wav"):
            try:
                self._winsound.PlaySound(self.path, self._winsound.SND_FILENAME | self._winsound.SND_ASYNC | self._winsound.SND_LOOP)
                return
            except Exception:
                pass
        # ffplay native loop
        if self._ffplay:
            try:
                self._loop_stop.clear()
                self.proc = subprocess.Popen(
                    [self._ffplay, "-nodisp", "-loglevel", "quiet", "-autoexit", "-stream_loop", "-1", self.path],
                    preexec_fn=(os.setsid if not self._is_win else None),
                    creationflags=(self._win_createflags if self._is_win else 0),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                def _watch():
                    while not self._loop_stop.is_set(): time.sleep(0.1)
                    self._stop_proc()
                self.loop_thread = threading.Thread(target=_watch, daemon=True); self.loop_thread.start()
                return
            except Exception:
                self.proc = None
        # Fallback loop: respawn when finished
        self._loop_stop.clear()
        def _loop():
            while not self._loop_stop.is_set():
                self._spawn_once()
                while not self._loop_stop.is_set():
                    alive = False
                    if self.proc and self.proc.poll() is None: alive = True
                    if self.play_obj and self.play_obj.is_playing(): alive = True
                    if not alive: break
                    time.sleep(0.1)
            self._stop_proc()
        self.loop_thread = threading.Thread(target=_loop, daemon=True); self.loop_thread.start()

# ---------- Drawing helpers ----------
def _draw_lines_zones(frame, lines_cfg, zones_cfg, frame_color, frame_thickness):
    th = int(frame_thickness if frame_thickness is not None else 2)
    if th <= 0:
        return

    # Lines (straight & polylines)
    for ln in (lines_cfg or []):
        pts = _get_line_pts(ln)
        col = frame_color if frame_color is not None else (0,165,255)
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                a = (int(pts[i-1][0]), int(pts[i-1][1]))
                b = (int(pts[i][0]),   int(pts[i][1]))
                if i == len(pts)-1:
                    try:
                        cv2.arrowedLine(frame, a, b, col, max(1, th), tipLength=0.08)
                    except Exception:
                        cv2.line(frame, a, b, col, th, cv2.LINE_AA)
                else:
                    cv2.line(frame, a, b, col, th, cv2.LINE_AA)
            # A/B markers
            a0 = (int(pts[0][0]), int(pts[0][1])); b0 = (int(pts[-1][0]), int(pts[-1][1]))
            cv2.circle(frame, a0, 4, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, b0, 4, col, -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "A", (a0[0]+6, a0[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.putText(frame, "B", (b0[0]+6, b0[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # Zones
    for zn in (zones_cfg or []):
        pts = np.array(zn["pts"], dtype=np.int32)
        if len(pts) >= 3:
            col = frame_color if frame_color is not None else (0,165,255)
            cv2.polylines(frame, [pts], True, col, th, cv2.LINE_AA)



def _draw_trails(frame, trails, trace_color, trace_thickness):
    th = int(trace_thickness if trace_thickness is not None else 2)
    if not trails or th <= 0:
        return
    for tid, dq in trails.items():
        if len(dq) < 2:
            continue
        if trace_color is None:
            r = (37 * tid) % 256; g = (91 * tid) % 256; b = (157 * tid) % 256
            col = (int(b), int(g), int(r))
        else:
            col = trace_color
        pts = np.array(dq, dtype=np.int32)
        cv2.polylines(frame, [pts], False, col, th, cv2.LINE_AA)


def _ab_dir_hint_for_line(ln: dict) -> str:
    """
    Human hint for what A->B means (uses the same cross-product sign as the counter).
    Vertical-ish: Left->Right / Right->Left. Horizontal-ish: Up->Down / Down->Up.
    """
    ax, ay = ln["a"]; bx, by = ln["b"]
    dx, dy = bx - ax, by - ay
    mx, my = (ax + bx) * 0.5, (ay + by) * 0.5

    def s(px, py):
        # same as line_side(a,b,p): (bx-ax)*(py-ay) - (by-ay)*(px-ax)
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    if abs(dx) < abs(dy):               # vertical-ish -> Left/Right
        sl = s(mx - 10, my)             # left of the line
        sr = s(mx + 10, my)             # right of the line
        return "Left->Right" if sl < sr else "Right->Left"
    else:                                # horizontal-ish -> Up/Down
        su = s(mx, my - 10)             # above (smaller y)
        sd = s(mx, my + 10)             # below (larger y)
        return "Up->Down" if su < sd else "Down->Up"


def _draw_hud_panel(
    img,
    lines: list[str],
    anchor: str = "br",  # "tl" | "tr" | "bl" | "br"
    margin: int = 12,
    pad: int = 8,
    font = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.6,
    color = (255, 255, 255),
    bgcolor = (0, 0, 0),
    alpha: float = 0.55,
    thickness: int = 2,
):
    """Draw a semi-transparent black box with text lines."""
    if not lines:
        return
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    maxw = max((w for (w, h) in sizes), default=0)
    lineh = max((h for (w, h) in sizes), default=14)
    panel_w = maxw + 2 * pad
    panel_h = len(lines) * (lineh + 6) - 6 + 2 * pad  # 6 px spacing

    H, W = img.shape[:2]
    if anchor == "tl":
        x0, y0 = margin, margin
    elif anchor == "tr":
        x0, y0 = W - margin - panel_w, margin
    elif anchor == "bl":
        x0, y0 = margin, H - margin - panel_h
    else:  # "br"
        x0, y0 = W - margin - panel_w, H - margin - panel_h

    # background (semi-transparent)
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + panel_w, y0 + panel_h), bgcolor, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

    # text
    y = y0 + pad + lineh
    for t in lines:
        cv2.putText(img, t, (x0 + pad, y), font, scale, color, thickness, cv2.LINE_AA)
        y += lineh + 6









def _draw_counts_labels(frame, lines_cfg, line_counts, zones_cfg, zone_counts):
    """
    Draw a compact HUD with results in the **bottom-right** on a black background.
    Uses ASCII-only text (A->B). No transparency (solid black) to maximize legibility.
    """
    # 1) Build text lines
    hud_lines = []
    for i, ln in enumerate(lines_cfg or []):
        ab = int(line_counts[i].get("ab", 0))
        ba = int(line_counts[i].get("ba", 0))
        # small human hint for A->B direction
        try:
            hint = _ab_dir_hint_for_line(ln)
        except Exception:
            hint = ""
        suffix = f"  ({hint})" if hint else ""
        hud_lines.append(f"{ln['name']}:  A->B {ab}  |  B->A {ba}{suffix}")

    for i, zn in enumerate(zones_cfg or []):
        zin  = int(zone_counts[i].get("in", 0))
        zout = int(zone_counts[i].get("out", 0))
        hud_lines.append(f"{zn['name']}:  IN {zin}  |  OUT {zout}")

    if not hud_lines:
        return  # nothing to draw

    # 2) Layout (bottom-right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    th    = 2
    color = (255, 255, 255)   # white text
    margin = 12
    pad    = 8
    line_gap = 6

    # Measure panel size
    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t in hud_lines]
    maxw  = max(w for w, h in sizes)
    lineh = max(h for w, h in sizes)
    panel_w = maxw + 2 * pad
    panel_h = len(hud_lines) * (lineh + line_gap) - line_gap + 2 * pad

    H, W = frame.shape[:2]
    x0 = max(0, W - margin - panel_w)
    y0 = max(0, H - margin - panel_h)

    # 3) Solid black background for readability
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)

    # 4) Draw text lines
    y = y0 + pad + lineh
    for t in hud_lines:
        cv2.putText(frame, t, (x0 + pad, y), font, scale, color, th, cv2.LINE_AA)
        y += lineh + line_gap




# ---------- Anchors & events ----------
def _anchor_from_box(b, mode: str, ghost_margin: int = 0):
    if mode == "bottom":
        return 0.5*(b[0]+b[2]), max(0.0, b[3] - float(ghost_margin))
    return 0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])

def _process_frame_counting(app, frame_idx, fps, names,
                            lines_cfg, zones_cfg,
                            det_boxes, det_confs, det_cids, det_ids,
                            last_anchor, line_states, line_counts, zone_states, zone_counts, events,
                            line_min_gap, zone_min_gap, anchor_mode, ghost_margin,
                            alert_enabled, selected_class_ids_set,
                            alert_freeze_ms, alert_when_inside,
                            event_time_sec, timecode_str, clock_str,
                            sound_player: SoundPlayer | None, alert_loop: bool):
    anchors = [_anchor_from_box(b, anchor_mode, ghost_margin) for b in det_boxes]
    now_ms = int(time.time()*1000)
    frame_active_ids = set()  # IDs that should keep loop playing in this frame

    for (tid, b, s, cid, (cx,cy)) in zip(det_ids, det_boxes, det_confs, det_cids, anchors):
        # Lines: one-shot sound on crossing
        # Lines: single beep when crossing (straight or polyline)
        for li, ln in enumerate(lines_cfg or []):
            st = line_states[li].get(tid, {"last_side": None, "last_frame": -9999})
            prev_c = last_anchor.get(tid, (cx,cy))
            crossed = False; direction = None

            if "pts" in ln and len(ln["pts"]) >= 2:
                pts_line = _get_line_pts(ln)
                if st["last_side"] is None:
                    st["last_side"] = _polyline_side(pts_line, (cx,cy))
                else:
                    direction = _polyline_cross_direction(prev_c, (cx,cy), pts_line)
                    if direction is not None and (frame_idx - st["last_frame"] >= line_min_gap):
                        crossed = True
                # keep last_side updated (so next frame isn't None)
                st["last_side"] = _polyline_side(pts_line, (cx,cy))
            else:
                a = (ln["a"][0], ln["a"][1]); b2 = (ln["b"][0], ln["b"][1])
                cur_side = line_side(a, b2, (cx,cy))
                prev_side = st["last_side"]
                if prev_side is not None:
                    if segments_intersect(prev_c, (cx,cy), a, b2):
                        if prev_side < 0 and cur_side > 0: direction = "ab"
                        elif prev_side > 0 and cur_side < 0: direction = "ba"
                        if direction is not None and (frame_idx - st["last_frame"] >= line_min_gap):
                            crossed = True
                st["last_side"] = cur_side

            if crossed:
                st["last_frame"] = frame_idx
                line_states[li][tid] = st
                line_counts[li][direction] += 1
                events.append({
                    "frame": int(frame_idx),
                    "time_sec": float(event_time_sec),
                    "timecode": timecode_str,
                    "clock": clock_str,
                    "track_id": int(tid),
                    "class_id": int(cid),
                    "class_name": (names[cid] if isinstance(names, dict) else names[cid]),
                    "event_type": f"line_{direction}",
                    "counter_name": ln["name"],
                    "conf": float(s)
                })
                if alert_enabled:
                    cname = (names[cid] if isinstance(names, dict) else names[cid]).lower()
                    if (not alert_classes_set) or (cname in alert_classes_set):
                        _beep(alert_freq, alert_dur)
            else:
                line_states[li][tid] = st


        # Zones: active while inside/outside (depending on mode)
        for zi, zn in enumerate(zones_cfg or []):
            sstate = zone_states[zi].get(tid, {"inside": False, "last_change": -9999})
            inside_now = point_in_polygon((cx,cy), zn["pts"])
            if inside_now != sstate["inside"]:
                if frame_idx - sstate["last_change"] >= zone_min_gap:
                    sstate["inside"] = inside_now
                    sstate["last_change"] = frame_idx
                    zone_states[zi][tid] = sstate
                    ev = "zone_in" if inside_now else "zone_out"
                    if inside_now: zone_counts[zi]["in"] += 1
                    else: zone_counts[zi]["out"] += 1
                    events.append({
                        "frame": int(frame_idx),
                        "time_sec": float(event_time_sec),
                        "timecode": timecode_str,
                        "clock": clock_str,
                        "track_id": int(tid),
                        "class_id": int(cid),
                        "class_name": (names[cid] if isinstance(names, dict) else names[cid]),
                        "event_type": ev,
                        "counter_name": zn["name"],
                        "conf": float(s)
                    })
            else:
                zone_states[zi][tid] = sstate

            # Sound condition per object → aggregate in frame_active_ids
            if alert_enabled and (cid in selected_class_ids_set):
                want_inside = bool(alert_when_inside)
                cond = (sstate.get("inside", False) is True) if want_inside else (sstate.get("inside", False) is False)
                if cond:
                    frame_active_ids.add(tid)

        last_anchor[tid] = (cx,cy)

    # Frame-level sound handling (immediate start/stop)
    if alert_enabled and sound_player:
        if alert_loop:
            if frame_active_ids:
                if not app._alert_state.get("looping", False):
                    if now_ms - app._alert_state.get("last_ms", 0) >= int(alert_freeze_ms):
                        sound_player.start_loop()
                        app._alert_state["last_ms"] = now_ms
                        app._alert_state["looping"] = True
                        try: app._log("[ALERT] loop start")
                        except Exception: pass
            else:
                if app._alert_state.get("looping", False):
                    sound_player.stop()
                    app._alert_state["looping"] = False
                    try: app._log("[ALERT] loop stop")
                    except Exception: pass
        else:
            # one-shot pings while condition holds, freeze-limited
            if frame_active_ids and now_ms - app._alert_state.get("last_ms", 0) >= int(alert_freeze_ms):
                sound_player.play_once()
                app._alert_state["last_ms"] = now_ms
                try: app._log("[ALERT] ping")
                except Exception: pass

    return anchors

# ---------- Main ----------
def run(app, sources, outp: Path, selected_idx):
    t0 = time.time()
    try:
        vids_dir = ensure_dir(outp / "videos")
        ev_dir   = ensure_dir(outp / "events")
        summ_dir = ensure_dir(outp / "summary")
        cnt_dir  = ensure_dir(outp / "counters")
        ensure_dir(outp / "temp")

        p = VIDEO_PRESETS.get(int(app.quality.get()), VIDEO_PRESETS.get(DEFAULT_QUALITY)).copy()
        if getattr(app, "advanced_override", False):
            p.update(app.adv_params)
        imgsz = int(p["imgsz"]); conf = float(p["conf"]); iou = float(p["iou"])
        frame_skip = int(p["frame_skip"]); stride = max(1, frame_skip + 1)
        track_buffer = int(p["track_buffer"]); match_thresh = float(p["match_thresh"]); min_hits = int(p["min_hits"])
        line_min_gap = int(p.get("line_min_gap", LINE_MIN_GAP_FRAMES_DEFAULT))
        zone_min_gap = int(p.get("zone_min_gap", ZONE_MIN_GAP_FRAMES_DEFAULT))
        device = device_auto_str()

        names = app.model.names
        id2name = names if isinstance(names, dict) else {i:nm for i,nm in enumerate(names)}
        # Selected class IDs from main window drive alert filtering
        selected_class_ids_set = set(int(i) for i in selected_idx)

        anchor_mode = getattr(app, "anchor_mode", None).get() if hasattr(app, "anchor_mode") else "bottom"
        overlay_mode = getattr(app, "overlay_mode", None).get() if hasattr(app, "overlay_mode") else "centroid"
        ghost_margin = int(getattr(app, "ghost_margin", None).get() if hasattr(app, "ghost_margin") else 0)

        # TRACE & FRAME (ADVANCED)
        trace_on = getattr(app, "trace_enabled", None).get() if hasattr(app, "trace_enabled") else True
        trace_len = int(getattr(app, "trace_len", None).get() if hasattr(app, "trace_len") else 24)
        trace_color = _parse_color(p.get("trace_color", None))
        trace_thickness = int(p.get("trace_thickness", 2)) if str(p.get("trace_thickness","")).strip() != "" else 2

        frame_color = _parse_color(p.get("overlay_frame_color", None))
        frame_thickness = int(p.get("overlay_frame_thickness", 2)) if str(p.get("overlay_frame_thickness","")).strip() != "" else 2

        # ALERTS — read from UI vars first (works even if user didn't hit Apply)
        alert_enabled = bool(getattr(app, "alert_enabled", None).get()) if hasattr(app, "alert_enabled") else bool(p.get("alert_enabled", False))
        alert_sound_path = ""
        if hasattr(app, "alert_sound"):
            try: alert_sound_path = str(app.alert_sound.get()).strip()
            except Exception: alert_sound_path = ""
        if not alert_sound_path:
            alert_sound_path = str(p.get("alert_sound", "")).strip()
        alert_loop = bool(getattr(app, "alert_loop", None).get()) if hasattr(app, "alert_loop") else bool(p.get("alert_loop", True))
        if hasattr(app, "alert_freeze_s"):
            try:
                alert_freeze_ms = 1000 * int(app.alert_freeze_s.get())
            except Exception:
                alert_freeze_ms = 1000 * int(p.get("alert_freeze_s", 2))
        else:
            alert_freeze_ms = 1000 * int(p.get("alert_freeze_s", 2))
        alert_when_inside = int(p.get("alert_zone_inside", 1))  # 1=in zone, 0=outside

        sound_player = SoundPlayer(alert_sound_path if alert_sound_path else None)

        app._log(f"Param: imgsz={imgsz}, conf={conf}, iou={iou}, frame_skip={frame_skip}, "
                 f"track_buffer={track_buffer}, match={match_thresh}, hits={min_hits}, device={device}")
        tracker_kind = (getattr(app, "tracker_kind", None).get() if hasattr(app, "tracker_kind") else "bytetrack")
        tracker_name, _tracker_obj_preview = _make_tracker(tracker_kind, conf, track_buffer, match_thresh, min_hits)
        app._log(
            "Tracker: {tn} | Selected classes: {cls} | Alert={onoff} {mode} {snd}; freeze={fz:.1f}s; zone_mode={zmode} | Sound backends: {bk}".format(
                tn=tracker_name,
                cls=", ".join(id2name[i] for i in sorted(selected_class_ids_set)) if selected_class_ids_set else "(none)",
                onoff=("ON" if alert_enabled else "OFF"),
                mode="(loop)" if alert_loop else "(ping)",
                snd=f"(file: {Path(alert_sound_path).name})" if alert_sound_path else "(no file)",
                fz=alert_freeze_ms/1000.0,
                zmode=("INSIDE" if alert_when_inside else "OUTSIDE"),
                bk=sound_player.describe_backends() if sound_player else "none"
            )
        )

        for vi, source in enumerate(sources):
            src_name = (str(source) if not isinstance(source, (int,)) else f"cam_{source}")
            app._log(f"\n=== {vi+1}/{len(sources)}: {src_name} ===")

            is_stream = _is_stream_source(source)
            cap = cv2.VideoCapture(source if is_stream or isinstance(source, (int,)) else str(source))
            if not cap or not cap.isOpened():
                app._log(f"[WARN] Cannot open: {src_name}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (not is_stream and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0) else None
            fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps>1e-3 else 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            start_perf = time.perf_counter()
            start_epoch = time.time()

            # --- Counter editor ---
            if is_stream:
                base_stem = _re.sub(r'[^A-Za-z0-9_]+','_', src_name if isinstance(source, str) else f"cam_{source}")
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, frame_bgr=None, default_cfg_path=default_cfg_path, live_cap=cap)
                app.wait_window(editor)
                lines_cfg = editor.lines[:]; zones_cfg = editor.zones[:]
                first_frame = None
            else:
                ok, first_frame = cap.read()
                if not ok or first_frame is None:
                    app._log(f"[ERR] No first frame: {src_name}")
                    cap.release(); continue
                base_stem = (Path(src_name).stem if isinstance(source, (str,Path)) else f"cam_{source}")
                default_cfg_path = cnt_dir / f"{base_stem}.json"
                editor = CounterEditor(app, frame_bgr=first_frame, default_cfg_path=default_cfg_path, live_cap=None)
                app.wait_window(editor)
                lines_cfg = editor.lines[:]; zones_cfg = editor.zones[:]

            if not lines_cfg and not zones_cfg:
                app._log("[WARN] No lines or zones — skipping.")
                cap.release(); continue

            # --- Writer ---
            fps_out = max(1.0, fps / float(stride))
            writer, out_path = open_video_writer_collision(vids_dir / f"{base_stem}_annotated.mp4", W, H, fps_out)
            if not writer or not writer.isOpened():
                app._log(f"[ERR] Cannot open VideoWriter: {src_name}")
                cap.release(); continue

            tracker_name, tracker = _make_tracker(tracker_kind, conf, track_buffer, match_thresh, min_hits)

            last_anchor = {}
            line_states = [{ } for _ in lines_cfg]
            line_counts = [{"ab":0,"ba":0} for _ in lines_cfg]
            zone_states = [{ } for _ in zones_cfg]
            zone_counts = [{"in":0,"out":0} for _ in zones_cfg]
            events = []
            trails = {} if (getattr(app, "trace_enabled", None).get() if hasattr(app, "trace_enabled") else True) else None

            # per-run alert state (for freeze + loop)
            app._alert_state = {"last_ms": 0, "looping": False}

            def _frame_timing(is_stream_local: bool) -> tuple[float, str, str]:
                if is_stream_local:
                    sec = time.perf_counter() - start_perf
                    tc = _fmt_timecode(sec)
                    clk = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_epoch + sec))
                    return sec, tc, clk
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    sec = float(pos_ms) / 1000.0
                else:
                    cur_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if cur_idx and cur_idx > 0 and fps > 0:
                        sec = float(cur_idx) / float(fps)
                    else:
                        sec = 0.0
                tc = _fmt_timecode(sec)
                return sec, tc, ""

            def _handle_frame(frame, frame_idx):
                res = app.model(frame, imgsz=imgsz, conf=conf, iou=iou,
                                device=device, classes=list(sorted(selected_class_ids_set)), verbose=False)[0]

                det_boxes, det_confs, det_cids, det_ids = [], [], [], []
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    if tracker is not None and len(xyxy) > 0 and sv is not None:
                        dets = sv.Detections(
                            xyxy=xyxy.astype(np.float32),
                            confidence=confs.astype(np.float32),
                            class_id=cls.astype(np.int32)
                        )
                        dets = tracker.update_with_detections(dets)
                        if dets.tracker_id is not None:
                            det_ids = dets.tracker_id.astype(int).tolist()
                            det_boxes = dets.xyxy.astype(float).tolist()
                            det_confs = dets.confidence.astype(float).tolist()
                            det_cids  = dets.class_id.astype(int).tolist()
                    if not det_ids:
                        det_boxes = xyxy.astype(float).tolist()
                        det_confs = confs.astype(float).tolist()
                        det_cids  = cls.astype(int).tolist()
                        det_ids   = list(range(1, len(det_boxes)+1))

                sec, tc, clk = _frame_timing(is_stream)

                anchors = _process_frame_counting(
                    app, frame_idx, fps, id2name,
                    lines_cfg, zones_cfg,
                    det_boxes, det_confs, det_cids, det_ids,
                    last_anchor, line_states, line_counts, zone_states, zone_counts, events,
                    line_min_gap, zone_min_gap, anchor_mode, ghost_margin,
                    alert_enabled, selected_class_ids_set,
                    alert_freeze_ms, alert_when_inside,
                    sec, tc, clk,
                    sound_player if alert_enabled and sound_player and sound_player.path else None,
                    alert_loop
                )

                if trails is not None:
                    for tid, a in zip(det_ids, anchors):
                        dq = trails.get(tid)
                        if dq is None:
                            dq = deque(maxlen=max(2, int((getattr(app, "trace_len", None).get() if hasattr(app, "trace_len") else 24))))
                            trails[tid] = dq
                        dq.append((int(a[0]), int(a[1])))

                overlay = frame
                try:
                    draw_detections(overlay, det_boxes, det_confs, det_cids, det_ids,
                                    id2name, (overlay_mode or "centroid"),
                                    None, True, anchors)
                except Exception:
                    pass

                frame_color = _parse_color(p.get("overlay_frame_color", None))
                frame_thickness = int(p.get("overlay_frame_thickness", 2)) if str(p.get("overlay_frame_thickness","")).strip() != "" else 2
                trace_color = _parse_color(p.get("trace_color", None))
                trace_thickness = int(p.get("trace_thickness", 2)) if str(p.get("trace_thickness","")).strip() != "" else 2

                _draw_lines_zones(overlay, lines_cfg, zones_cfg, frame_color, frame_thickness)
                _draw_trails(overlay, trails, trace_color, trace_thickness)
                _draw_counts_labels(overlay, lines_cfg, line_counts, zones_cfg, zone_counts)

                return overlay

            processed = 0
            if not is_stream and 'first_frame' in locals() and first_frame is not None:
                if (processed % stride) == 0:
                    ov = _handle_frame(first_frame, processed)
                    ov = _ensure_bgr(ov)
                    writer.write(ov)
                    _preview(app, ov, processed, fps, total_frames)
                processed += 1

            while True:
                if app.abort_event.is_set():
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if (processed % stride) != 0:
                    processed += 1
                    continue

                ov = _handle_frame(frame, processed)
                ov = _ensure_bgr(ov)
                writer.write(ov)
                _preview(app, ov, processed, fps, total_frames)
                processed += 1

            # stop any leftover looping sound when source finishes
            try:
                if sound_player:
                    sound_player.stop()
            except Exception:
                pass

            cap.release()
            writer.release()

            # Save events + summary
            ev_df = pd.DataFrame(events)
            ev_path = save_csv_collision(ev_df, ev_dir / f"{base_stem}_events.csv")
            app._log(f"Saved events: {ev_path}")

            summary = {
                "source": src_name,
                "frames": int(processed),
                "fps": float(fps),
                "lines": [{"name": ln["name"], **line_counts[i]} for i,ln in enumerate(lines_cfg)],
                "zones": [{"name": zn["name"], **zone_counts[i]} for i,zn in enumerate(zones_cfg)],
                "advanced": {
                    "trace_color": p.get("trace_color", None),
                    "trace_thickness": int(p.get("trace_thickness", 2)),
                    "overlay_frame_color": p.get("overlay_frame_color", None),
                    "overlay_frame_thickness": int(p.get("overlay_frame_thickness", 2)),
                    "alert_zone_inside": int(p.get("alert_zone_inside", 1)),
                    "alert_sound": alert_sound_path,
                    "alert_loop": alert_loop,
                    "alert_freeze_s": int(alert_freeze_ms/1000)
                }
            }
            sum_path = save_json_collision(summary, summ_dir / f"{base_stem}_summary.json")
            app._log(f"Saved summary: {sum_path}")

        app._set_progress(100.0, "Done.")

    except Exception as e:
        try: app._log(f"[ERROR] {e}")
        except Exception: print(e)
    finally:
        try:
            app.worker_done.set()
            app.btn_start.config(state="normal")
            app.btn_abort.config(state="disabled")
        except Exception:
            pass
        app._log(f"\nDone in {time.time()-t0:.1f}s")

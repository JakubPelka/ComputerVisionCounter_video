# cv_video_sound.py
from __future__ import annotations
import subprocess, threading, signal, shutil, sys, os, time

class SoundPlayer:
    """
    Plays a custom sound file (wav/mp3/ogg/…).
    Backends in order: ffplay, macOS afplay, Linux paplay/aplay, Windows winsound,
    Windows PowerShell (WAV/MP3), simpleaudio (WAV).
    """
    def __init__(self, path: str | None):
        self.path = str(path) if path else None
        self.proc: subprocess.Popen | None = None
        self.play_obj = None
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
        self._winsound = None
        if self._is_win:
            try:
                import winsound  # type: ignore
                self._winsound = winsound
            except Exception:
                self._winsound = None
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
                if self._is_win: self.proc.terminate()
                else: os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
        self.proc = None

    def stop(self):
        with self._lock:
            self._loop_stop.set()
            if self._winsound:
                try: self._winsound.PlaySound(None, self._winsound.SND_PURGE)
                except Exception: pass
            if self.loop_thread and self.loop_thread.is_alive():
                self._stop_proc()
            self.loop_thread = None
            if self.play_obj:
                try: self.play_obj.stop()
                except Exception: pass
                self.play_obj = None
            self._stop_proc()

    # -- one shot
    def _spawn_once(self):
        if not self.path: return
        if self._ffplay:
            try:
                self.proc = subprocess.Popen(
                    [self._ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", self.path],
                    preexec_fn=(os.setsid if not self._is_win else None),
                    creationflags=(self._win_createflags if self._is_win else 0),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ); return
            except Exception: self.proc = None
        if self._afplay:
            try:
                self.proc = subprocess.Popen(
                    [self._afplay, self.path],
                    preexec_fn=(os.setsid if not self._is_win else None),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ); return
            except Exception: self.proc = None
        for cmd in (self._paplay, self._aplay):
            if cmd:
                try:
                    self.proc = subprocess.Popen(
                        [cmd, self.path],
                        preexec_fn=(os.setsid if not self._is_win else None),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    ); return
                except Exception: self.proc = None
        if self._winsound and self.path.lower().endswith(".wav"):
            try:
                self._winsound.PlaySound(self.path, self._winsound.SND_FILENAME | self._winsound.SND_ASYNC)
                return
            except Exception: pass
        if self._is_win:
            ps_path = self.path.replace("'", "''")
            if self.path.lower().endswith(".wav"):
                script = (
                    "[System.Reflection.Assembly]::LoadWithPartialName('System.Media') | Out-Null; "
                    f"$p = New-Object System.Media.SoundPlayer('{ps_path}'); $p.PlaySync()"
                )
            else:
                script = (
                    "$w = New-Object -ComObject WMPlayer.OCX; "
                    f"$m = $w.newMedia('{ps_path}'); $w.URL = '{ps_path}'; "
                    "$w.controls.play(); while ($w.playState -ne 1) { Start-Sleep -Milliseconds 100 }; $w.close()"
                )
            try:
                self.proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", script],
                    creationflags=self._win_createflags,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ); return
            except Exception: self.proc = None
        if self._sa and self.path.lower().endswith(".wav"):
            try:
                wave_obj = self._sa.WaveObject.from_wave_file(self.path)
                self.play_obj = wave_obj.play()
                return
            except Exception: self.play_obj = None

    def play_once(self):
        self.stop()
        self._spawn_once()

    # -- loop
    def start_loop(self):
        if self._winsound and self.path and self.path.lower().endswith(".wav"):
            try:
                self._winsound.PlaySound(
                    self.path,
                    self._winsound.SND_FILENAME | self._winsound.SND_ASYNC | self._winsound.SND_LOOP
                ); return
            except Exception: pass
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
            except Exception: self.proc = None

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

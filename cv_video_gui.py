# cv_video_gui.py — EN: ScrollableFrame, AppUIMixin, CounterEditor (lines & zones)
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import cv2
from pathlib import Path


class ScrollableFrame(ttk.Frame):
    """Reusable scrollable container with Canvas + inner Frame."""
    def __init__(self, parent, height=240, width=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        if width:
            self.canvas.config(width=width)
        # stretch inner width with canvas width
        self.canvas.bind("<Configure>", self._on_canvas_config)

    def _on_canvas_config(self, event):
        try:
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        except Exception:
            pass


class AppUIMixin:
    """Currently a placeholder; kept for compatibility/imports."""
    pass


class CounterEditor(tk.Toplevel):
    """
    Lightweight editor to draw counting lines and zones on a reference frame.
    Produces:
      self.lines = [ {name, a:(x,y), b:(x,y)} , ... ]
      self.zones = [ {name, pts:[(x,y), ...]} , ... ]
    """
    def __init__(self, parent, frame_bgr, default_cfg_path: Path | None = None):
        super().__init__(parent)
        self.title("Counter Editor")
        self.transient(parent)
        self.grab_set()

        self.default_cfg_path = Path(default_cfg_path) if default_cfg_path else None

        # image
        self._src_bgr = frame_bgr.copy()
        rgb = cv2.cvtColor(self._src_bgr, cv2.COLOR_BGR2RGB)
        self._img = Image.fromarray(rgb)
        self._tk = ImageTk.PhotoImage(self._img)

        # state
        self.mode = tk.StringVar(value="line")  # "line"|"zone"
        self.lines = []
        self.zones = []
        self._tmp_pts = []          # points being drawn (for line -> 2 points; for zone -> N points)
        self._preview_items = []    # canvas item ids to clear
        self._line_count = 0
        self._zone_count = 0

        # UI
        self._build_ui()

        # mouse bindings
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Button-3>", self._on_right_click)  # finish polygon

    def _build_ui(self):
        body = ttk.Frame(self); body.pack(fill="both", expand=True, padx=8, pady=8)

        # toolbar
        tb = ttk.Frame(body); tb.pack(fill="x", pady=(0,6))
        ttk.Label(tb, text="Mode:").pack(side="left")
        ttk.Radiobutton(tb, text="Draw line", variable=self.mode, value="line").pack(side="left", padx=6)
        ttk.Radiobutton(tb, text="Draw zone (polygon)", variable=self.mode, value="zone").pack(side="left", padx=6)
        ttk.Button(tb, text="Undo", command=self._undo).pack(side="left", padx=(12,4))
        ttk.Button(tb, text="Clear", command=self._clear).pack(side="left", padx=(4,12))
        ttk.Button(tb, text="Load…", command=self._load).pack(side="left", padx=4)
        ttk.Button(tb, text="Save…", command=self._save).pack(side="left", padx=4)
        ttk.Button(tb, text="OK", command=self._ok).pack(side="right", padx=4)
        ttk.Button(tb, text="Cancel", command=self._cancel).pack(side="right", padx=4)

        # canvas
        self.canvas = tk.Canvas(body, bg="#111", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        # show image
        self._img_id = self.canvas.create_image(0, 0, image=self._tk, anchor="nw")
        self.canvas.config(width=self._img.width, height=self._img.height)

        # legend
        info = ttk.Label(body, text="Left click: add point | Right click: finish polygon | Undo removes last point/shape")
        info.pack(fill="x", pady=(6,0))

        self._redraw_all()

    # ---------- drawing helpers ----------
    def _redraw_all(self):
        self.canvas.delete("all")
        self._preview_items.clear()
        self._img_id = self.canvas.create_image(0, 0, image=self._tk, anchor="nw")

        # existing lines
        for ln in self.lines:
            a = (int(ln["a"][0]), int(ln["a"][1]))
            b = (int(ln["b"][0]), int(ln["b"][1]))
            self.canvas.create_line(a[0], a[1], b[0], b[1], fill="#00FFFF", width=3, arrow=tk.LAST)
            self.canvas.create_text(min(a[0], b[0]) + 6, min(a[1], b[1]) - 8,
                                    text=ln["name"], fill="#00FFFF", anchor="w")

        # existing zones
        for zn in self.zones:
            pts = [(int(x), int(y)) for (x,y) in zn["pts"]]
            if len(pts) >= 2:
                self.canvas.create_polygon(*sum(pts, ()), outline="#FFA500", width=2, fill="", smooth=False)
            # label
            if pts:
                cx = int(sum(p[0] for p in pts) / len(pts))
                cy = int(sum(p[1] for p in pts) / len(pts))
                self.canvas.create_text(cx, cy, text=zn["name"], fill="#FFA500")

        # temp preview
        if self._tmp_pts:
            if self.mode.get() == "line":
                if len(self._tmp_pts) == 1:
                    # just a node
                    x, y = self._tmp_pts[0]
                    self._preview_items.append(self.canvas.create_oval(x-3, y-3, x+3, y+3, outline="#0F0", fill="#0F0"))
                elif len(self._tmp_pts) >= 2:
                    x1,y1 = self._tmp_pts[0]; x2,y2 = self._tmp_pts[-1]
                    self._preview_items.append(self.canvas.create_line(x1,y1,x2,y2, fill="#0F0", width=2, dash=(5,3)))
            else:
                # polygon
                pts = [(int(x), int(y)) for (x,y) in self._tmp_pts]
                if len(pts) == 1:
                    x,y = pts[0]
                    self._preview_items.append(self.canvas.create_oval(x-3, y-3, x+3, y+3, outline="#0F0", fill="#0F0"))
                elif len(pts) >= 2:
                    flat = sum(pts, ())
                    self._preview_items.append(self.canvas.create_line(*flat, fill="#0F0", width=2, dash=(4,2)))

    def _on_click(self, event):
        x, y = int(event.x), int(event.y)
        if self.mode.get() == "line":
            if len(self._tmp_pts) == 0:
                self._tmp_pts.append((x,y))
            elif len(self._tmp_pts) == 1:
                self._tmp_pts.append((x,y))
                # finalize line
                self._line_count += 1
                name = f"Line {self._line_count}"
                a = self._tmp_pts[0]; b = self._tmp_pts[1]
                self.lines.append({"name": name, "a": a, "b": b})
                self._tmp_pts.clear()
            else:
                self._tmp_pts.clear()
        else:
            # polygon mode
            self._tmp_pts.append((x,y))
        self._redraw_all()

    def _on_right_click(self, event):
        # finish polygon
        if self.mode.get() == "zone" and len(self._tmp_pts) >= 3:
            self._zone_count += 1
            name = f"Zone {self._zone_count}"
            self.zones.append({"name": name, "pts": self._tmp_pts.copy()})
            self._tmp_pts.clear()
            self._redraw_all()

    def _on_motion(self, event):
        # update preview (last point follows mouse)
        if not self._tmp_pts:
            return
        if self.mode.get() == "line":
            if len(self._tmp_pts) == 1:
                self._tmp_pts = [self._tmp_pts[0], (int(event.x), int(event.y))]
            else:
                self._tmp_pts[-1] = (int(event.x), int(event.y))
        else:
            # polygon preview: last temp point follows
            if len(self._tmp_pts) >= 1:
                if len(self._tmp_pts) >= 2:
                    self._tmp_pts[-1] = (int(event.x), int(event.y))
        self._redraw_all()

    # ---------- commands ----------
    def _undo(self):
        if self._tmp_pts:
            self._tmp_pts.pop()
        elif self.mode.get() == "line" and self.lines:
            self.lines.pop()
        elif self.mode.get() == "zone" and self.zones:
            self.zones.pop()
        self._redraw_all()

    def _clear(self):
        if messagebox.askyesno("Clear", "Remove all lines and zones?"):
            self.lines.clear()
            self.zones.clear()
            self._tmp_pts.clear()
            self._redraw_all()

    def _load(self):
        initfile = None
        if self.default_cfg_path:
            initfile = str(self.default_cfg_path)
        path = filedialog.askopenfilename(
            title="Load counters (JSON)",
            initialfile=initfile if initfile and Path(initfile).exists() else None,
            filetypes=[("JSON","*.json"), ("All","*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.lines = []
            self.zones = []
            for ln in data.get("lines", []):
                if "a" in ln and "b" in ln:
                    self.lines.append({
                        "name": ln.get("name", f"Line {len(self.lines)+1}"),
                        "a": tuple(ln["a"]),
                        "b": tuple(ln["b"]),
                    })
            for zn in data.get("zones", []):
                pts = zn.get("pts", [])
                if pts:
                    self.zones.append({
                        "name": zn.get("name", f"Zone {len(self.zones)+1}"),
                        "pts": [tuple(p) for p in pts],
                    })
            self._redraw_all()
        except Exception as e:
            messagebox.showerror("Load", f"Failed to load JSON:\n{e}")

    def _save(self):
        initfile = None
        if self.default_cfg_path:
            initfile = str(self.default_cfg_path)
        path = filedialog.asksaveasfilename(
            title="Save counters (JSON)",
            defaultextension=".json",
            initialfile=Path(initfile).name if initfile else "counters.json",
            filetypes=[("JSON","*.json")]
        )
        if not path:
            return
        data = {
            "lines": self.lines,
            "zones": self.zones,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Save", f"Failed to save JSON:\n{e}")

    def _ok(self):
        # return lines/zones to caller
        self.grab_release()
        self.destroy()

    def _cancel(self):
        if messagebox.askyesno("Cancel", "Discard changes and close?"):
            self.lines = []
            self.zones = []
            self.grab_release()
            self.destroy()

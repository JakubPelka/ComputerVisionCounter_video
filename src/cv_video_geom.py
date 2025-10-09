# cv_video_geom.py
from __future__ import annotations

# ---- Line/polyline helpers ---------------------------------------------------

def get_line_pts(ln: dict):
    """Return [(x,y), ...] for a line (straight or polyline)."""
    if "pts" in ln and len(ln["pts"]) >= 2:
        return [(float(x), float(y)) for x, y in ln["pts"]]
    return [(float(ln["a"][0]), float(ln["a"][1])), (float(ln["b"][0]), float(ln["b"][1]))]

def line_side(a, b, p) -> float:
    """Signed side of point p relative to segment a->b (cross product)."""
    ax, ay = a; bx, by = b; px, py = p
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

def _point_to_segment_dist2(a, b, p) -> float:
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx*vx + vy*vy
    t = 0.0 if vv == 0 else max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
    cx, cy = ax + t*vx, ay + t*vy
    dx, dy = px - cx, py - cy
    return dx*dx + dy*dy

def polyline_side(pts, p) -> float:
    """Side sign using the *nearest* polyline segment to p."""
    best_i, best_d = 0, 1e30
    for i in range(len(pts) - 1):
        d2 = _point_to_segment_dist2(pts[i], pts[i+1], p)
        if d2 < best_d:
            best_d, best_i = d2, i
    a, b = pts[best_i], pts[best_i+1]
    return line_side(a, b, p)

def segments_intersect(p1, p2, q1, q2) -> bool:
    def _orient(a,b,c):
        v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        return 1 if v > 0 else (-1 if v < 0 else 0)
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

def polyline_cross_direction(prev_p, cur_p, pts):
    """
    If motion segment prev->cur intersects ANY polyline segment,
    return 'ab' or 'ba' using that segment’s orientation; else None.
    """
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i+1]
        if segments_intersect(prev_p, cur_p, a, b):
            ps = line_side(a, b, prev_p)
            cs = line_side(a, b, cur_p)
            if ps < 0 and cs > 0: return "ab"
            if ps > 0 and cs < 0: return "ba"
    return None

def point_in_polygon(p, poly) -> bool:
    x, y = p; inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1 + 1e-12) + x1)
        if cond: inside = not inside
    return inside

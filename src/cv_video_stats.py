# cv_video_stats.py
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Iterable, List, Dict, Optional

class StatsAggregator:
    """Per-class 'now' and 'max this run' counters.

    - Initialize with class name list (id -> name).
    - Optionally pass `selected_ids` to limit counting.
    - Call `update_from_cids(cids)` every frame.
    - Read `now_named()` and `max_named()` for HUD.
    """
    def __init__(self, class_names: List[str], selected_ids: Optional[Iterable[int]] = None):
        self.class_names = list(class_names)
        self.selected_ids = set(selected_ids) if selected_ids else None
        self._now = Counter()
        self._max = defaultdict(int)

    def update_from_cids(self, cids) -> None:
        self._now.clear()
        if cids is None:
            return
        try:
            import numpy as _np
            arr = _np.asarray(cids).astype(int)
            if self.selected_ids is not None:
                if arr.size == 0:
                    return
                mask = _np.isin(arr, _np.fromiter(self.selected_ids, dtype=int))
                arr = arr[mask]
            if arr.size:
                unique, counts = _np.unique(arr, return_counts=True)
                for cid, cnt in zip(unique.tolist(), counts.tolist()):
                    self._now[int(cid)] += int(cnt)
        except Exception:
            for v in cids:
                cid = int(v)
                if (self.selected_ids is not None) and (cid not in self.selected_ids):
                    continue
                self._now[cid] += 1

        for cid, n in self._now.items():
            if n > self._max[cid]:
                self._max[cid] = n

    def now_named(self) -> Dict[str, int]:
        return { self._name(cid): n for cid, n in self._now.items() }

    def max_named(self) -> Dict[str, int]:
        return { self._name(cid): n for cid, n in self._max.items() if n > 0 }

    def _name(self, cid: int) -> str:
        try:
            return self.class_names[int(cid)]
        except Exception:
            return str(cid)

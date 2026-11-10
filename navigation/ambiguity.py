"""Accumulated anonymous range geometry. No RGB or simulator semantics.

Fixed-height adaptation of thesis 4.2.3: both flanks support an apparent gap.
Range gaps remain hypotheses; only an executed verification authorizes crossing.
"""
from collections import deque
import numpy as np


class RangeMemory:
    def __init__(self, cfg):
        self.cfg = cfg; self.frames = deque(); self.last_time = -1.

    def add(self, p):
        if not np.isfinite(p.time) or p.time <= self.last_time: return False
        self.last_time = p.time
        points = p.points[p.valid & np.isfinite(p.points).all(axis=1)]
        points = points[(points[:, 2] > .60) & (points[:, 2] < 1.95)]
        self.frames.append((p.time, points.copy()))
        while self.frames and p.time-self.frames[0][0] > self.cfg.scan_history_s: self.frames.popleft()
        return True

    def evidence(self, opening, p):
        x, y = opening['center_xy_m']; width = opening['width_m']
        origin = p.scan_origin if p.scan_origin is not None else p.position
        dy = p.directions[:, 1]
        t = (y-origin[1])/np.where(abs(dy) > .01, dy, np.nan)
        at = origin[None, :] + t[:, None]*p.directions
        selected = ((t > .05) & (t < self.cfg.scan_range-.10) & (abs(at[:, 0]-x) < min(.36, width*.28))
                    & (at[:, 2] > .70) & (at[:, 2] < 1.85))
        idx = np.flatnonzero(selected); usable = idx[p.valid[idx]]
        valid_ratio = len(usable)/max(1, len(idx))
        panel = float(np.mean(abs(p.distances[usable]-t[usable]) < .10)) if len(usable) else 0.
        clear = float(np.mean(p.distances[usable]-t[usable] > .30)) if len(usable) else 0.
        occluded = float(np.mean(p.distances[usable] < t[usable]-.18)) if len(usable) else 1.
        history = [a for stamp, a in self.frames if stamp <= p.time]
        all_points = np.concatenate(history, axis=0) if history else np.empty((0, 3))
        near = all_points[abs(all_points[:, 1]-y) < .11, :2]
        # Coarse aperture location is public; its contents never enter this fit.
        left = near[(near[:, 0] > x-width/2-.50) & (near[:, 0] < x-width/2+.10)]
        right = near[(near[:, 0] > x+width/2-.10) & (near[:, 0] < x+width/2+.50)]
        def unique(points):
            return len(np.unique(np.round(points/.018).astype(int), axis=0)) if len(points) else 0
        nl, nr = unique(left), unique(right)
        supports = min(nl, nr) >= self.cfg.ambiguity_min_side_points and min(nl, nr)/max(1, max(nl, nr)) > .10
        normal = np.array([0., 1.]); center = np.array([x, y]); residual = None
        if supports:
            sides = np.concatenate([left, right]); mid = sides.mean(axis=0)
            _, _, v = np.linalg.svd(sides-mid, full_matrices=False); tangent = v[0]
            normal = np.array([-tangent[1], tangent[0]])
            if normal[1] < 0: normal = -normal
            residual = float(np.median(abs((sides-mid)@normal)))
            if abs(normal[1]) > .7: center[1] = mid[1]-(x-mid[0])*normal[0]/normal[1]
            else: supports = False
            supports = supports and residual < .045
        in_front = .35 < y-p.position[1] < 2.8
        quality = len(idx) >= 5 and valid_ratio >= .50 and occluded < .20
        ambiguous = bool(in_front and quality and supports and panel < self.cfg.ambiguity_gap_fraction)
        return dict(time_s=p.time, ray_count=len(idx), usable_rays=len(usable), valid_ratio=valid_ratio,
                    panel_support=panel, clearance_support=clear, occlusion_support=occluded,
                    left_support_points=nl, right_support_points=nr, side_support_confirmed=bool(supports),
                    normal_xy=normal.tolist(), center_xy_m=center.tolist(), plane_residual_m=residual,
                    ambiguous_gap=ambiguous, quality_valid=bool(quality),
                    source='accumulated anonymous range / bilateral supports; gap is not free-space proof')

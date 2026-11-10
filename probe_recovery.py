"""Rotor-only local release: stop at the first sustained, plan-able safe state."""
import math
import numpy as np
from navigation.door_clearance import collision_bounds
from navigation.physics import yaw_error


def monotone_exit(planner, start, end):
    """Exit existing inflation overlaps; never cross a new obstacle."""
    points = np.linspace(start, end, max(25, int(np.linalg.norm(end-start)/.01)+2))
    if np.any(points < planner.origin) or np.any(points > planner.upper):
        return False
    for c, r, h in planner.boxes:
        depth = np.maximum(0., np.min(h-np.abs((points-c)@r), axis=1))
        if depth[0] > 0:
            if np.any(np.diff(depth) > 1e-8):
                return False
        elif depth.max() > 1e-8:
            return False
    return True


class ProbeRecovery:
    # Engineering flight/clearance gates, NOT measurement-error calibration.
    dwell = .25
    timeout = 8.

    def __init__(self, experiment):
        self.e = experiment
        self.started = float(experiment.world.data.time)
        self.start_xy = experiment.position().copy()
        self.anchor = None
        self.stable_since = None
        self.ready = False
        self.last_status = {}
        self.next_check = 0.
        if not hasattr(experiment, 'recovery_frames'):
            experiment.recovery_frames = []

    def geometry(self, p, q):
        e = self.e
        exit_planner = e.planner(q, 'bypass')
        current_planner = e.planner(q, 'target')
        clear = not exit_planner.collides(p)[0] and not current_planner.collides(p)[0]
        return clear, current_planner, exit_planner

    def exit_target(self, p, q, current, exit_planner):
        # Select the nearest feasible local egress, not a fixed displacement.
        normal = np.array([-np.sin(q), np.cos(q)]) if self.e.kind == 'door' else np.array([0., 1.])
        directions = [-normal, np.array([0., -1.])]
        toward = self.e.staging-p
        if np.linalg.norm(toward) > .01:
            directions.append(toward/np.linalg.norm(toward))
        for distance in np.arange(.04, 1.25, .04):
            for direction in directions:
                end = p+distance*direction
                # An extra .02m clearance avoids braking on a raster boundary.
                probes = end+np.array([[0,0],[.02,0],[-.02,0],[0,.02],[0,-.02]])
                if exit_planner.collides(probes).any() or current.collides(probes).any():
                    continue
                if not monotone_exit(current, p, end):
                    continue
                try:
                    exit_planner.plan(end, self.e.goal)
                except ValueError:
                    continue
                return end
        raise ValueError('no local monotone clearance egress')

    def command(self):
        e, w = self.e, self.e.world
        p, v, R, gyro = w.observed()
        q, _ = e.state()
        if w.data.time >= self.next_check:
            self.next_check = float(w.data.time)+.05
            clear, current, exit_planner = self.geometry(p[:2], q)
            bounds = collision_bounds(w)
            actual = np.max([np.maximum(abs(lo), abs(hi)) for lo, hi in bounds], axis=0)
            envelope_ok = bool(np.all(actual <= e.envelope.half-e.cfg.DOOR_POSITION_MARGIN))
            detached = w.force < .1 and e.last_force < .1
            stable = (np.linalg.norm(v) < .065 and np.linalg.norm(gyro) < .16
                      and R[2,2] >= math.cos(math.radians(10))
                      and abs(yaw_error(R, 0.)) < .10 and abs(p[2]-1.25) < .065)
            exit_ok = False
            if clear:
                # Brake in place immediately, including while yaw recovers.
                if self.anchor is None:
                    self.anchor = p[:2].copy()
                try:
                    exit_planner.plan(p[:2], e.goal)
                    # Both formal branches must be computable at THIS pose.
                    e.plan(p[:2], e.target, 'target')
                    exit_ok = True
                except ValueError:
                    pass
            else:
                self.anchor = None
            if self.anchor is None:
                self.anchor = self.exit_target(p[:2], q, current, exit_planner)
            ok = bool(detached and stable and envelope_ok and clear and exit_ok)
            if not ok:
                self.stable_since = None
            elif self.stable_since is None:
                self.stable_since = float(w.data.time)
            self.ready = bool(ok and w.data.time-self.stable_since >= self.dwell)
            self.last_status = dict(detached=detached, stable=bool(stable),
                 envelope_ok=envelope_ok, clearance=bool(clear), exit_path=exit_ok,
                 dwell_s=0. if self.stable_since is None else float(w.data.time-self.stable_since))
            e.recovery_frames.append(dict(time=float(w.data.time), **self.last_status,
                position_xy=p[:2].tolist(), speed_mps=float(np.linalg.norm(v)),
                gyro_radps=float(np.linalg.norm(gyro)), contact_N=float(w.force), ready=self.ready))
        w.target = np.r_[self.anchor, 1.25]
        # Limit retreat speed and acceleration; all actuation remains on rotors.
        delta = self.anchor-p[:2]
        desired = delta*1.5
        speed = np.linalg.norm(desired)
        if speed > .25:
            desired *= .25/speed
        return w.mass*3.8*(desired-v[:2])

    def record(self):
        return dict(**self.last_status, elapsed_s=float(self.e.world.data.time)-self.started,
                    displacement_m=float(np.linalg.norm(self.e.position()-self.start_xy)),
                    position_xy=self.e.position().tolist(), trigger='state_ready')

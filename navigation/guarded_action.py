"""Online action migrated from B; no Experiment/reset or hidden parameters.
Only preparation uses current geometry; approach/excite/unload commands and
solved-interval instrumentation preserve probe_design.execute's equations.
"""
from dataclasses import dataclass
import numpy as np
from .physics import clip_norm, PhysicsFailure
from .interactions import panel_geometry, probe
from .door_dynamics import ProbeTrace

@dataclass(frozen=True)
class Action:
    id: str
    speed: float = .2
    arm: float = .835
    hold: float = .8
    coast: float = 1.2
    force: float = 2.2
    oblique_deg: float = 0.
    ramp: float = 0.
    legacy: bool = False
    unload_angle_deg: object = None


GUARDED = Action('guarded_coast', hold=1.4, coast=1.2, unload_angle_deg=20.)

def execute(w, opening, inst, action, prepare):
    """Measure every solved interval, including the first collision.

    Contact onset uses PRE-step velocity (not a stale post-step sensor).
    Coast actively backs the UAV away, while continuing to measure actual
    contacts. It is not assumed to have zero torque. No force is applied to door.
    """
    old = w.after_step_observer
    start = float(w.data.time)
    records = [[float(w.data.time), *inst.state()]]
    inputs, force, points, own_speed, cmd = [], [], [], [], []
    onset = None
    pre_velocity = w.observed()[1].copy()
    raw_command = np.zeros(2)
    safe_peak = 0.
    preparation = True

    def sample(world):
        nonlocal onset, safe_peak
        if old:
            old(world)
        tau, cs = inst.contact()
        f = sum(float(np.linalg.norm(c['force_on_door_world'])) for c in cs)
        safe_peak = max(safe_peak, f)
        if f > 1e-6 and onset is None:
            onset = max(0,len(inputs)-int(getattr(inst,'pretrigger_intervals',0)))
        inputs.append(tau)
        force.append(f)
        points.append(cs)
        q, v = inst.state()
        n = np.array([-np.sin(q), np.cos(q)])
        # mj_step1 evaluated this velocity sensor at the pre-integration state.
        own_speed.append(float(pre_velocity[:2]@n))
        cmd.append(raw_command.copy())
        records.append([float(world.data.time), q, v])

    def step(f, yaw):
        nonlocal pre_velocity, raw_command
        pre_velocity = w.observed()[1].copy()
        raw_command = np.array(f)
        w.step(clip_norm(f, w.cfg.probe_force_max), yaw=yaw)

    w.after_step_observer = sample
    try:
        if action.legacy:
            probe(w, opening, inst.state)
        else:
            prepare(w, opening, inst, action)
            if onset is not None:
                raise PhysicsFailure('preparation_contact (recorded; no state reset)')
            begin = float(w.data.time)
            w.stage = 'DESIGN_APPROACH'
            while onset is None:
                if w.data.time-begin > 12.:
                    raise PhysicsFailure('contact_timeout')
                q, _ = inst.state()
                t = np.array([np.cos(q), np.sin(q)])
                n = np.array([-np.sin(q), np.cos(q)])
                pivot = panel_geometry(opening, q)[0]
                pos, vel, _, _ = w.observed()
                ft = 4.*float((pivot+action.arm*t-pos[:2])@t)-3.*float(vel[:2]@t)
                fn = w.mass*3.2*(action.speed-float(vel[:2]@n))
                w.target = np.r_[pos[:2]+.1*n, 1.25]
                step(ft*t+fn*n, q)
            origin = float(w.data.time)
            release_pos = None
            w.stage = 'DESIGN_EXCITE'
            while w.data.time-origin < action.hold+action.coast-1e-9:
                elapsed = float(w.data.time-origin)
                q, _ = inst.state()
                t = np.array([np.cos(q), np.sin(q)])
                n = np.array([-np.sin(q), np.cos(q)])
                pos, vel, _, _ = w.observed()
                pivot = panel_geometry(opening, q)[0]
                active = (elapsed < action.hold-1e-9 and release_pos is None and
                          (action.unload_angle_deg is None or q < np.radians(action.unload_angle_deg)))
                if active:
                    ft = 4.*float((pivot+action.arm*t-pos[:2])@t)-3.*float(vel[:2]@t)
                    fn = action.force*min(1., elapsed/action.ramp) if action.ramp else action.force
                    ft += fn*np.tan(np.radians(action.oblique_deg))
                    w.target = np.r_[pos[:2], 1.25]
                    step(ft*t+fn*n, q)
                else:
                    if release_pos is None:
                        release_pos = pos.copy()
                        release_pos[:2] -= .28*n
                    w.stage = 'DESIGN_UNLOAD_OBSERVE'
                    w.target = release_pos.copy()
                    step(4.*(release_pos[:2]-pos[:2])-3.*vel[:2], q)
        a = np.asarray(records)
        # New probes condition on the state immediately before first contact;
        # retaining this interval preserves the impact impulse exactly.
        i = 0 if action.legacy else onset
        trace = ProbeTrace(a[i:, 0]-a[i, 0], np.r_[inputs[i:], inputs[-1]],
                           a[i:, 1], a[i:, 2])
        trace.validate()
        return dict(trace=trace, force=np.asarray(force[i:]+[force[-1]]),
                    contacts=points[i:], commands=np.asarray(cmd[i:]+[cmd[-1]]),
                    peak_force=max(force, default=0.), duration=float(w.data.time)-start,
                    contact_speed=None if onset is None else own_speed[onset],
                    wall_peak=w.wall_peak, min_height=w.min_height, max_tilt=w.max_tilt)
    except PhysicsFailure as e:
        e.diagnostics = dict(peak_force=safe_peak, duration=float(w.data.time)-start,
                             wall_peak=w.wall_peak, max_tilt=w.max_tilt, min_height=w.min_height)
        raise
    finally:
        w.after_step_observer = old

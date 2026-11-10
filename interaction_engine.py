"""Fixed-altitude rotor-only NAMO experiment. Run this file in the existing PyCharm env."""
import runtime
import argparse
from dataclasses import replace
from functools import partial
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent

from configuration import Config, architecture
from navigation.door_clearance import VehicleEnvelope, DoorClearance
from navigation.geometry import Planner, rectangle
from navigation.interactions import approach, panel_geometry, recover
from navigation.physics import World, PhysicsFailure, clip_norm
from scene.build import build_xml
import mujoco

from belief import ACTIONS, Belief
from decision_rule import evaluate
from probe_recovery import ProbeRecovery
from local_hover import LocalHoverRecovery, deferred_egress, hover_planner
from copy import deepcopy

DOOR_WIDTH = 1.1825
ARM = .835
BOX_HALF = np.array([DOOR_WIDTH/2, .25])
BOX_TRAVEL = 2.
DAMPING = .05
IMPACT_GUARD = 8.
PROBE_SPEED = .15
CASES = {
    'light-door': ('door', .15, .35),
    'heavy-door': ('door', .42, 2.40),
    'near-door': ('door', .32, 1.35),
    'boundary-push': ('door', .32, 1.35),
    'boundary-avoid': ('door', .32, 1.35),
    'majority-boundary-push': ('door', .32, 1.35),
    'majority-boundary-avoid': ('door', .32, 1.35),
    'light-box': ('box', 1.1, .62),
    'heavy-box': ('box', 2.1, 3.10),
    'short-detour': ('door', .15, .35),
}
CASE_RATES = {'boundary-push': 3.70, 'boundary-avoid': 3.72,
              'majority-boundary-push':3.81,'majority-boundary-avoid':3.82}
STATE_CASE_RATES = {'majority-boundary-push':4.085,'majority-boundary-avoid':4.099}
LOCAL_CASE_RATES = {'majority-boundary-push':4.499,'majority-boundary-avoid':4.514,
                    'boundary-avoid':4.397,'boundary-push':4.377}


def scene(arch, cfg, *args, kind, inertia, resistance, **kwargs):
    contents = {'Oglass': 'glass'}
    if kind == 'door':
        contents['Otarget'] = 'door'
    xml, truth = build_xml(arch, cfg, door_parameters={'Otarget': (inertia, resistance)},
                           observed_contents=contents, visual=True)
    if kind == 'door':
        return xml, truth
    root = ET.fromstring(xml)
    x, y = arch['openings'][1]['center_xy_m']
    body = ET.SubElement(root.find('worldbody'), 'body', name='panel_Otarget', pos=f'{x} {y} 1')
    ET.SubElement(body, 'joint', name='hinge_Otarget', type='slide', axis='0 1 0',
                  range=f'0 {BOX_TRAVEL}', frictionloss=str(resistance), damping=str(DAMPING))
    ET.SubElement(body, 'geom', name='surface_Otarget', type='box',
                  size=f'{BOX_HALF[0]} {BOX_HALF[1]} .99', mass=str(inertia),
                  rgba='.76 .35 .13 1', friction='.3 .005 .0001', solref='.012 1')
    return ET.tostring(root, encoding='unicode'), truth


def make_map(name):
    arch = architecture()
    arch['start_xyz_m'] = [0.24, 3.50, 1.25]
    arch['goal_xyz_m'] = [0.24, 8.15, 1.25]
    bypass_x = 4.55
    if name == 'short-detour':
        # Start and goal align with the already open corridor.
        arch['start_xyz_m'][0] = arch['goal_xyz_m'][0] = bypass_x
    arch['internal_walls'] = [dict(id='partition', y_m=5.4,
                                  opening_ids=['Oglass', 'Otarget', 'Obypass'])]
    arch['openings'] = [dict(id=oid, center_xy_m=[x, 5.4], width_m=width, normal_xy=[0, 1])
                        for oid, x, width in [('Oglass', -3.5, 1.65),
                                              ('Otarget', 0., 1.3625),
                                              ('Obypass', bypass_x, 1.75)]]
    return arch


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


class Experiment:
    def __init__(self, args):
        self.args = args
        kind, inertia, resistance = CASES[args.case]
        inertia = inertia if args.inertia is None else args.inertia
        resistance = resistance if args.resistance is None else args.resistance
        self.truth_parameters = [inertia, resistance]
        self.kind = kind
        self.arm = ARM if kind == 'door' else 1.
        self.arch = make_map(args.case)
        self.opening = self.arch['openings'][1]
        self.x, self.y = self.opening['center_xy_m']
        self.staging = (approach(self.opening)[:2] if kind == 'door'
                        else np.array([self.x, self.y-1.03]))
        self.goal = np.array(self.arch['goal_xyz_m'][:2])
        self.cap = args.force_cap
        self.cfg = replace(Config(), grid_resolution=.06, flight_speed=.75, push_force_max=self.cap)
        self.world = World(self.arch, self.cfg, scene_builder=partial(
            scene, kind=kind, inertia=inertia, resistance=resistance))
        m = self.world.model
        self.joint = m.joint('hinge_Otarget').id
        self.surface = m.geom('surface_Otarget').id
        self.envelope = VehicleEnvelope.from_world(self.world)
        height = self.arch['flight_height_m']
        if height-self.envelope.half[2] <= 0 or height+self.envelope.half[2] >= self.cfg.DOOR_CLEAR_HEIGHT:
            raise PhysicsFailure('vehicle envelope exceeds floor/lintel clearance')
        self.belief = Belief()
        self.prior = Belief()
        self.dashboard = None
        self.next_evidence = 0.
        self.retreat_preview = False
        self.next_route_preview = 0.
        self.last_decision = {}
        self.validation = {}
        self.events, self.trace, self.frames = [], [], []
        self.sensor_frames = []
        self.paths, self.snapshots = {}, []
        self.work, self.peak = 0., 0.
        self.probed = self.committed = self.crossed = False
        self.viewer = None
        self.snapshot_saved = False
        self.world.callback = self.capture
        self.target = self.compute_target()
        self.world.after_step_observer = self.measure
        self.rows = None
        self.last_force = 0.
        self.status = 'INIT'
        self.probe_recovery = None
        self.probe_recovery_error = None
        self.recovery_frames = []

    def state(self):
        d, j = self.world.data, self.joint
        return float(d.qpos[self.world.model.jnt_qposadr[j]]), float(d.qvel[self.world.model.jnt_dofadr[j]])

    def position(self):
        return self.world.observed()[0][:2]

    def panel(self, q):
        if self.kind == 'door':
            return panel_geometry(self.opening, q)[2]
        return rectangle([self.x, self.y+q], BOX_HALF)

    def planner(self, q, route):
        sealed = ['Oglass']
        if route == 'bypass':
            sealed.append('Otarget')
        if route == 'target':
            sealed.append('Obypass')
        return Planner(self.arch, self.cfg, sealed=sealed, panels=[] if q is None else [self.panel(q)],
                       footprint=self.envelope.half[:2])

    def plan(self, start, q, route):
        path, _ = self.planner(q, route).plan(start, self.goal)
        return path, path_length(path)

    def compute_target(self):
        if self.kind == 'door':
            result = DoorClearance(self.world).solve(self.opening)
            if not result['feasible']:
                raise PhysicsFailure(result['reason'])
            # Geometry solver already includes panel-angle uncertainty.
            return result['theta_clear']
        for q in np.arange(.04, BOX_TRAVEL, .04):
            try:
                self.plan(self.staging, q-.025, 'target')
                self.plan(self.staging, q+.025, 'target')
                return float(q)
            except ValueError:
                pass
        raise PhysicsFailure('no box translation creates a passage')

    def measure(self, w):
        collecting = self.rows is not None
        q, v = self.state()
        prev = self.rows[-1][1] if collecting else q
        dq = q-prev
        total_force = 0.
        u = 0.
        pivot = w.data.xanchor[self.joint]
        for i in range(w.data.ncon):
            contact = w.data.contact[i]
            a, b = int(contact.geom1), int(contact.geom2)
            if not ((a == self.surface and b in w.robot_geoms) or
                    (b == self.surface and a in w.robot_geoms)):
                continue
            mujoco.mj_contactForce(w.model, w.data, i, w.wrench)
            sign = 1 if b == self.surface else -1
            force = sign * contact.frame.reshape(3, 3).T @ w.wrench[:3]
            total_force += np.linalg.norm(force)
            if self.kind == 'door':
                r = contact.pos[:2] - pivot[:2]
                c, s = np.cos(dq), np.sin(dq)
                displacement = np.r_[(np.array([[c, -s], [s, c]]) - np.eye(2)) @ r, 0.]
                u += np.cross(contact.pos-pivot, force)[2]
            else:
                displacement = np.array([0., dq, 0.])
                u += force[1]
            self.work += max(0., float(force@displacement))
        if collecting and w.data.time >= self.next_evidence and len(self.rows)>51:
            preview = deepcopy(self.belief)
            preview.update(self.local_rows(self.rows))
            self.evidence_snapshot(preview, 'live_'+w.stage)
            self.next_evidence = float(w.data.time)+.10
        self.peak = max(self.peak, total_force)
        self.last_force = total_force
        if collecting:
            self.rows.append([float(w.data.time), q, v, float(u), float(total_force)])

    def capture(self, w):
        p = w.observed()[0]
        if self.retreat_preview and w.data.time >= self.next_route_preview:
            self.evidence_snapshot(self.belief, 'retreat_route_preview')
            self.next_route_preview = float(w.data.time)+.10
        readings = {w.model.sensor(i).name: w.data.sensor(w.model.sensor(i).name).data.tolist()
                    for i in range(w.model.nsensor)}
        readings.update(rotor_controls=w.data.ctrl.tolist(),
                        total_contact_N=float(self.last_force), work_J=float(self.work),
                        object_local_s_m=self.state()[0]*self.arm,
                        object_local_v_mps=self.state()[1]*self.arm)
        self.sensor_frames.append(dict(time=float(w.data.time), stage=w.stage, readings=readings))
        self.frames.append([float(w.data.time), *map(float, p), self.state()[0],
                            self.last_force, self.work, w.stage])
        if self.dashboard is not None:
            self.dashboard.update(self)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
            time.sleep(.05/self.args.playback)
        if (self.args.snapshot and not self.snapshot_saved and w.stage == 'SHORTCUT'
                and abs(p[1]-self.y) < 1.30):
            from PIL import Image
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [self.x, self.y+.25, 1.1]
            camera.distance, camera.azimuth, camera.elevation = 4.8, -59., -51.
            self.args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            with mujoco.Renderer(w.model, height=620, width=980) as renderer:
                renderer.update_scene(w.data, camera=camera)
                Image.fromarray(renderer.render()).save(self.args.snapshot)
            self.snapshot_saved = True
        if w.wall_peak > .35:
            raise PhysicsFailure(f'unintended wall/jamb contact: {w.wall_peak:.2f} N')
        if w.stage in ('SHORTCUT', 'BYPASS', 'BYPASS_EXIT') and w.force > .35:
            raise PhysicsFailure('unexpected contact during free flight')
        if w.stage == 'BYPASS_EXIT' and hover_planner(self,self.state()[0]).collides(p[:2])[0]:
            raise PhysicsFailure('local exit clearance changed during execution')

    def event(self, name, **values):
        self.events.append(dict(time=float(self.world.data.time), event=name, **values))
        details = ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}'
                           for k, v in values.items() if not isinstance(v, (list, dict)))
        print(f'{self.world.data.time:6.2f}  {name:20s} {details}', flush=True)

    def fly(self, path, stage):
        self.paths[stage] = np.asarray(path).tolist()
        # Fixed yaw for consistency with the derived envelope, including passage.
        self.world.fly(path[1:], yaw=0., speed=.45 if stage == 'SHORTCUT' else .70,
                       stage=stage, position_tolerance=.02)

    def interact(self, stage, fraction, hold, coast, travel_goal):
        w = self.world
        q0, v0 = self.state()
        self.rows = [[float(w.data.time), q0, v0, 0., 0.]]
        work0 = self.work
        onset = None
        release = None
        begin = float(w.data.time)
        reason = 'timeout'
        stall_since = None
        stream = stage == 'PROBE' and self.args.policy == 'adaptive'
        while w.data.time-begin < 24.:
            q, velocity = self.state()
            pos, vel, _, _ = w.observed()
            if self.kind == 'door':
                tangent = np.array([np.cos(q), np.sin(q)])
                normal = np.array([-np.sin(q), np.cos(q)])
                point = panel_geometry(self.opening, q)[0]+ARM*tangent-.035*normal
            else:
                tangent, normal = np.eye(2)
                point = np.array([self.x, self.y+q-BOX_HALF[1]])
            if onset is None and self.last_force > .08:
                onset = w.data.time
                if hasattr(self,'contact_started'):self.contact_started(stage)
            if (stream and release is None and onset is not None and self.last_force <= IMPACT_GUARD
                    and (w.data.time-onset >= hold or q+.28*max(velocity, 0.) >= travel_goal)):
                self.finish_trace('PROBE', 'online_decision', work0)
                _, _, _, _, _, moments = self.decision(self.staging)
                stream = False
                self.rows = [[float(w.data.time), q, velocity, 0., 0.]]
                work0 = self.work
                if moments[0] > 0:
                    self.committed = True
                    self.event('route_choice', route='continue_contact')
                    stage = 'PUSH'
                    onset = begin = w.data.time
                    hold, coast, fraction = 15., 2.25, 1.
                    travel_goal = self.target+(.025 if self.kind == 'door' else .055)
                else:
                    self.event('route_choice', route='unload_and_bypass')
                    hold = 0.
            if release is None:
                if self.last_force > IMPACT_GUARD:
                    reason = 'impact_guard'
                    release = w.data.time
                elif q >= travel_goal or (stage == 'PROBE' and q + .28*max(velocity, 0.) >= travel_goal):
                    reason = 'geometry_or_stroke'
                    release = w.data.time
                elif onset is not None and w.data.time-onset >= hold:
                    reason = 'excitation_end'
                    release = w.data.time
                elif onset is None and w.data.time-begin > 11.:
                    reason = 'no_contact'
                    release = w.data.time
                if stage == 'PUSH' and onset is not None and release is None:
                    if abs(velocity) < .009 and self.last_force > .60*self.cap:
                        stall_since = w.data.time if stall_since is None else stall_since
                        if w.data.time-stall_since >= .80:
                            reason, release = 'stall', w.data.time
                    else:
                        stall_since = None
                if release is not None:
                    retreat_distance = .39 if stage == 'PUSH' and self.kind == 'door' else .28
                    retreat = np.r_[pos[:2]-retreat_distance*normal, 1.25]
                    if stage == 'PROBE' and self.args.probe_recovery != 'fixed':
                        cls = LocalHoverRecovery if self.args.probe_recovery == 'local' else ProbeRecovery
                        self.probe_recovery = cls(self)
                        self.event('probe_release', recovery=self.args.probe_recovery, release_reason=reason)
            if release is not None:
                w.stage = stage+'_UNLOAD'
                w.target = retreat
                if self.last_force > IMPACT_GUARD:
                    reason = 'impact_guard'
                if stage == 'PROBE' and self.args.probe_recovery != 'fixed':
                    try:
                        f = self.probe_recovery.command()
                    except ValueError as exc:
                        self.probe_recovery_error = str(exc)
                        break
                    if (self.args.probe_recovery == 'local'
                            and self.probe_recovery.last_status.get('detached')
                            and self.probe_recovery.last_status.get('clearance')):
                        w.stage = 'PROBE_HOVER'
                    if self.probe_recovery.ready:
                        break
                    if self.args.probe_recovery == 'local' and w.data.time-release > self.probe_recovery.timeout:
                        self.probe_recovery_error = 'local hover state timeout'
                        break
                    if self.args.probe_recovery == 'state' and w.data.time-release >= coast:
                        break
                    w.step(clip_norm(f, self.cap), yaw=0.)
                    continue
                if w.data.time-release >= coast:
                    break
                f = 4*(retreat[:2]-pos[:2])-5*vel[:2]
            else:
                w.stage = stage if onset is not None else stage+'_APPROACH'
                w.target = np.r_[pos[:2], 1.25]
                ft = 4*float((point-pos[:2])@tangent)-3*float(vel[:2]@tangent)
                gap = float((point-pos[:2])@normal)
                speed = np.clip(.65*(gap-.26), .055, PROBE_SPEED)
                if onset is None and hasattr(self,'approach_speed'):
                    speed=self.approach_speed(gap,pos,vel,stage)
                # Target a geometric standoff first, then close along the local
                # normal. Tangential alignment alone can cut across a jamb.
                if self.kind == 'door' and onset is None and q > .22 and gap > .50:
                    entry = point-.43*normal
                    safe_x = np.clip(entry[0], self.x-.26, self.x+.26)
                    f = 3.2*(clip_norm(1.4*(np.array([safe_x, entry[1]])-pos[:2]), .22)-vel[:2])
                    w.step(clip_norm(f, self.cap), yaw=0.)
                    continue
                damping = 2.5
                if stage == 'PUSH':
                    damping += 6.*np.clip(1-(travel_goal-q)*self.arm/.42, 0., 1.)
                active_fraction = fraction
                if stage == 'PROBE' and self.args.action == 'ramp_coast' and onset is not None:
                    active_fraction *= min(1., max(.15, (w.data.time-onset)/.4))
                fn = (w.mass*3.2*(speed-float(vel[:2]@normal)) if onset is None else
                      max(0., self.cap*active_fraction-damping*float(vel[:2]@normal)))
                f = ft*tangent + fn*normal
            w.step(clip_norm(f, self.cap), yaw=q if self.kind == 'door' else 0.)
        self.finish_trace(stage, reason, work0)
        return reason

    def finish_trace(self, stage, reason, work0):
        a = np.asarray(self.rows)
        self.rows = None
        self.trace.extend([*row, stage] for row in a.tolist())
        self.belief.update(self.local_rows(a))
        self.evidence_snapshot(self.belief, stage.lower())
        self.event(stage.lower(), reason=reason, displacement=self.state()[0]-a[0, 1], work_J=self.work-work0,
                   peak_N=float(a[:, 4].max()))

    def retreat(self, stage):
        if stage == 'AFTER_PROBE' and self.args.probe_recovery != 'fixed':
            self.recover_probe()
            return
        w = self.world
        self.retreat_preview = stage == 'AFTER_PROBE'
        if self.retreat_preview:
            self.evidence_snapshot(self.belief, 'retreat_route_preview')
        p = self.position()
        planner = self.planner(self.state()[0], 'target')
        if planner.collides(p)[0] or w.force >= .1:
            if self.kind == 'door':
                recover(w, self.opening, self.state, bypass=True)
            else:
                w.fly([self.staging], yaw=0., speed=.35, stage=stage)
        w.align_yaw(0.)
        # A sealed-route planner cannot start within the target opening. Complete
        # a real backward flight to the public safe staging pose before replanning.
        if stage == 'AFTER_PROBE' and self.planner(self.state()[0], 'bypass').collides(w.observed()[0][:2])[0]:
            w.fly([self.staging], yaw=0., speed=.35, stage=stage, position_tolerance=.02)
        w.hold(w.observed()[0], .40, stage, yaw=0.)
        self.retreat_preview = False

    def recover_probe(self):
        w = self.world
        cls = LocalHoverRecovery if self.args.probe_recovery == 'local' else ProbeRecovery
        recovery = self.probe_recovery or cls(self)
        self.retreat_preview = True
        w.stage = 'PASSAGE_RELEASE' if getattr(recovery,'policy',None)=='direct_passage_connection' else 'AFTER_PROBE_STATE'
        fallback = False
        try:
            while True:
                try:
                    if self.probe_recovery_error:
                        raise ValueError(self.probe_recovery_error)
                    f = recovery.command()
                    if recovery.ready:
                        self.event('probe_recovery_ready', fallback_used=fallback, **recovery.record())
                        return
                    if w.data.time-recovery.started > recovery.timeout:
                        raise ValueError('local recovery state timeout')
                except ValueError as exc:
                    if fallback:
                        raise PhysicsFailure('probe recovery failed after staging fallback: '+str(exc))
                    self.event('probe_recovery_fallback', reason=str(exc))
                    fallback = True
                    # Use the established physical recovery only after local failure.
                    if self.kind == 'door':
                        recover(w, self.opening, self.state, bypass=True)
                    w.fly([self.staging], yaw=0., speed=.35, stage='PROBE_RECOVERY_FALLBACK',
                          position_tolerance=.02)
                    self.probe_recovery_error = None
                    recovery = cls(self)
                    self.probe_recovery = recovery
                    continue
                if getattr(recovery,'policy',None)!='direct_passage_connection':w.stage='AFTER_PROBE_STATE'
                w.step(clip_norm(f, self.cap), yaw=getattr(recovery,'yaw_command',0.))
        finally:
            self.retreat_preview = False

    def local_rows(self, rows):
        a = np.asarray(rows).copy()
        a[:, 1:3] *= self.arm
        a[:, 3] /= self.arm
        return a

    def evidence_snapshot(self, belief, label):
        snap = belief.snapshot()
        snap.update(self.last_decision)
        if label.startswith('live_PROBE') or label in ('probe', 'retreat_route_preview'):
            try:
                _, _, metrics = self.route_metrics(self.position())
                snap.update(metrics)
                snap['route_preview_status'] = 'current_geometry'
            except ValueError as exc:
                snap.update(push_limit=None, route_preview_status='unavailable: '+str(exc))
        snap.update(label=label, time=float(self.world.data.time))
        limit = snap.get('push_limit')
        prob, converged = belief.probability(limit) if limit is not None else (None, False)
        snap.update(push_probability=prob, integration_converged=converged)
        snap.update(evaluate(prob,converged,self.args.decision_rule,self.args.alpha,self.args.tie_margin))
        self.snapshots.append(snap)
        return snap

    def route_metrics(self, start):
        q, _ = self.state()
        prefix = None
        comparison_start = start
        if self.args.probe_recovery == 'local' and (
                self.planner(q,'bypass').collides(start)[0] or self.planner(q,'target').collides(start)[0]):
            # Hypothetical common connector in the geometry comparison. It is
            # NOT flown before the decision, nor when continuing contact.
            prefix = deferred_egress(self,np.asarray(start),q)
            comparison_start = prefix[-1]
        elif self.planner(q, 'bypass').collides(start)[0]:
            # Both alternatives pay the same explicitly planned retreat prefix.
            # No stale starting-point utility and no interpolated boundary.
            prefix, _ = self.planner(q, 'target').plan(start, self.staging)
            comparison_start = self.staging
        bypass, bypass_length = self.plan(comparison_start, q, 'bypass')
        candidate, candidate_length = self.plan(comparison_start, self.target, 'target')
        if prefix is not None:
            length = path_length(prefix)
            bypass_length += length; candidate_length += length
            bypass = np.vstack((prefix, bypass[1:]))
            candidate = np.vstack((prefix, candidate[1:]))
        stroke = max(self.target-q, 0.)*self.arm
        # Conservative explicit flight allowance for tracking the object, unloading
        # and realigning. Both branches share the current start and final goal.
        manoeuvre = (2*stroke+.56) if stroke > 0 else 0.
        gain = bypass_length-candidate_length-manoeuvre
        economic = gain/(self.args.rate*stroke) if self.args.rate*stroke > 0 else None
        limit = min(self.cap, economic) if economic is not None else self.cap
        if gain <= 0: limit = 0.
        metrics = dict(capacity_limit=self.cap, economic_limit=economic,
             push_limit=limit, gain=gain, stroke=stroke, expected_delta=0.,
             manoeuvre_allowance_m=manoeuvre, alpha=self.args.alpha,
             bypass_length_m=bypass_length, candidate_length_m=candidate_length,
             deferred_exit_m=0. if prefix is None else path_length(prefix),
             deferred_exit_path=[] if prefix is None else prefix.tolist(),
             exit_execution='after_bypass_choice' if self.args.probe_recovery=='local' else 'before_decision',
             geometry_q=q, geometry_xy=np.asarray(start).tolist())
        return bypass, candidate, metrics

    def decision(self, start, label='replan'):
        bypass, candidate, self.last_decision = self.route_metrics(start)
        metrics = self.last_decision
        bypass_length, candidate_length = metrics['bypass_length_m'], metrics['candidate_length_m']
        gain, stroke, limit = metrics['gain'], metrics['stroke'], metrics['push_limit']
        manoeuvre = metrics['manoeuvre_allowance_m']
        snap = self.evidence_snapshot(self.belief, label)
        if label == 'replan' and self.args.pause_at_decision and self.dashboard is not None:
            self.dashboard.pause_for_decision(self)
        prob = snap['push_probability']
        push = snap['decision_push']
        self.event('global_comparison', bypass_m=bypass_length, shortcut_m=candidate_length,
             manoeuvre_m=manoeuvre, gain_m=gain, remaining_m=stroke, F_star_N=limit,
             p_ok=prob, push=push, belief_status=self.belief.status,
             decision_rule=self.args.decision_rule,decision_label=snap['decision_label'],
             decision_reason=snap['decision_reason'])
        return bypass, candidate, gain, push

    def run(self):
        w = self.world
        w.hold(self.arch['start_xyz_m'], 1.3, 'SETTLE', yaw=0.)
        start = self.position()
        bypass, candidate, gain, _ = self.decision(start, 'start')
        self.paths['BYPASS_OPTION'] = bypass.tolist()
        self.paths['SHORTCUT_OPTION'] = candidate.tolist()
        gain = path_length(bypass)-path_length(candidate)  # G0: pure optimistic geometry
        if self.args.policy == 'bypass' or gain <= 0:
            self.event('skip_probe', reason='bypass_baseline_or_no_geometric_gain')
            self.fly(bypass, 'BYPASS'); return
        to_stage, _ = self.planner(self.state()[0], 'bypass').plan(start, self.staging)
        staged_bypass, _ = self.plan(self.staging, self.state()[0], 'bypass')
        approach_extra = max(0., path_length(to_stage)+path_length(staged_bypass)-path_length(bypass))
        action = next(a for a in ACTIONS if a.name == self.args.action)
        probe_stroke = min(self.target, np.radians(20) if self.kind == 'door' else .30)
        probe_budget = self.args.rate*self.cap*probe_stroke*self.arm
        if self.args.policy != 'always-push' and gain <= approach_extra+probe_budget:
            self.event('skip_probe', reason='geometric_gain_below_approach_and_probe_budget')
            self.fly(bypass, 'BYPASS'); return
        self.fly(to_stage, 'TO_INTERACTION')
        reason = ''
        if self.args.policy != 'always-push':
            self.probed = True
            self.event('probe_choice', action=action.name, approach_extra_m=approach_extra,
                       probe_work_budget_m=probe_budget)
            reason = self.interact('PROBE', action.fraction, action.hold, action.coast, probe_stroke)
            self.retreat('AFTER_PROBE')
        bypass, _, gain, push = self.decision(self.position())
        if (not push and self.args.policy != 'always-push') or reason in ('impact_guard','no_contact'):
            route_reason=reason if reason in ('impact_guard','no_contact') else self.snapshots[-1]['decision_reason']
            self.event('route_choice', route='bypass', reason=route_reason,
                       decision_label=self.snapshots[-1]['decision_label'])
            if self.args.probe_recovery == 'local':
                # Replan at actual time, then execute only the chosen exit.
                prefix = deferred_egress(self,self.position(),self.state()[0])
                if len(prefix)>1:
                    self.paths['BYPASS_EXIT'] = prefix.tolist()
                    self.event('bypass_exit_start', distance_m=path_length(prefix))
                    w.fly(prefix[1:],yaw=0.,speed=.15,stage='BYPASS_EXIT',position_tolerance=.02)
                    self.event('bypass_exit_end')
                bypass,_ = self.plan(self.position(),self.state()[0],'bypass')
            self.fly(bypass, 'BYPASS'); return
        self.committed = True
        self.event('route_choice', route='push' if self.state()[0]<self.target else 'cross')
        reason = ('already_clear' if self.state()[0]>=self.target else
                  self.interact('PUSH', 1., 15., 2.25,
                       self.target+(.025 if self.kind=='door' else .055)))
        self.finish_crossing(reason)

    def finish_crossing(self, reason):
        self.retreat('AFTER_PUSH')
        q, velocity = self.state()
        try:
            if reason == 'impact_guard' or abs(velocity) > .05 or q < self.target:
                raise ValueError('stable target clearance not confirmed')
            path, distance = self.plan(self.position(), q, 'target')
        except ValueError as exc:
            self.event('fallback', reason=str(exc))
            self.world.fly([self.staging], yaw=0., speed=.35, stage='FALLBACK_RETREAT', position_tolerance=.02)
            path, _ = self.plan(self.position(), q, 'bypass')
            self.fly(path, 'BYPASS')
        else:
            self.event('clearance_verified', observed_q=q, path_m=distance)
            self.fly(path, 'SHORTCUT')
            self.crossed = True

    def result(self, failure):
        frames = np.array([f[1:3] for f in self.frames])
        distance = path_length(frames) if len(frames) > 1 else 0.
        return dict(case=self.args.case, policy=self.args.policy, force_cap_N=self.cap,
                    rate_m_per_J=self.args.rate, success=failure is None and
                    float(np.linalg.norm(self.position()-self.goal)) < .07, failure=failure,
                    probed=self.probed, committed=self.committed, shortcut=self.crossed,
                    work_J=self.work, contact_peak_N=self.peak, wall_peak_N=self.world.wall_peak,
                    path_m=distance, realised_cost_m=distance+self.args.rate*self.work,
                    target_q=self.target, object_kind=self.kind, envelope_m=self.envelope.half.tolist(),
                    events=self.events, paths=self.paths, frames=self.frames, trace=self.trace,
                    snapshots=self.snapshots, force_replay=self.validation,
                    sensor_frames=self.sensor_frames,
                    truth_for_evaluation=[self.truth_parameters[0]/self.arm**2, self.truth_parameters[1]/self.arm],
                    action=self.args.action, alpha=self.args.alpha, measurement_mode='simulation_reference',
                    decision_rule=self.args.decision_rule,tie_margin=self.args.tie_margin,
                    probe_recovery=self.args.probe_recovery,
                    recovery_frames=self.recovery_frames,
                    map=self.arch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, default='light-door')
    parser.add_argument('--policy',
                        choices=['consistent', 'bypass', 'always-push'],
                        default='consistent')
    parser.add_argument('--action', choices=[a.name for a in ACTIONS], default='guarded_coast')
    parser.add_argument('--probe-recovery', choices=['local','state','fixed'], default='local',
                        help='local: hover nearby, exit only after bypass choice; state/fixed: historical controls')
    parser.add_argument('--alpha', type=float, default=.10)
    parser.add_argument('--decision-rule', choices=['majority','cautious'], default='majority',
                        help='majority: p>0.5 continues; cautious: p>=1-alpha continues')
    parser.add_argument('--tie-margin', type=float, default=0.,
                        help='Optional symmetric undecided band around .5; default exact tie only')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--close-on-done', action='store_true')
    parser.add_argument('--pause-at-decision', action='store_true', help='Pause dashboard after post-probe replanning; Space resumes')
    parser.add_argument('--inertia', type=float, help='Evaluator only: true inertia or mass')
    parser.add_argument('--resistance', type=float, help='Evaluator only: true Coulomb resistance')
    parser.add_argument('--force-cap', type=float, default=2.2, help='Sustained horizontal command limit (N)')
    parser.add_argument('--rate', type=float, default=None,
                        help='Lambda m/J; default .7; majority-boundary local 4.499/4.514, state 4.085/4.099, fixed 3.81/3.82')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--snapshot', type=Path, help='Render a PNG during shortcut passage')
    parser.add_argument('--log-only', action='store_true', help='Write JSON without HTML')
    parser.add_argument('--playback', type=float, default=2.)
    parser.add_argument('--output', type=Path, default=HERE/'results')
    args = parser.parse_args()
    if args.rate is None:
        rates = LOCAL_CASE_RATES if args.probe_recovery=='local' else STATE_CASE_RATES if args.probe_recovery=='state' else CASE_RATES
        args.rate = rates.get(args.case,CASE_RATES.get(args.case,.70))
    if not 0 < args.alpha < 1: parser.error('alpha must be between 0 and 1')
    if not 0<=args.tie_margin<.5:parser.error('tie-margin must be in [0,.5)')
    if not 0 < args.force_cap <= 2.2 or args.rate < 0 or args.playback <= 0:
        parser.error('require 0 < force-cap <= 2.2, rate >= 0 and playback > 0')
    if ((args.inertia is not None and args.inertia <= 0) or
            (args.resistance is not None and args.resistance < 0)):
        parser.error('require inertia > 0 and resistance >= 0')
    experiment = Experiment(args)
    failure = None
    if not args.headless and not args.viewer:
        from dashboard import Dashboard
        experiment.dashboard = Dashboard(experiment, args.playback)
    try:
        if args.viewer:
            from mujoco.viewer import launch_passive
            with launch_passive(experiment.world.model, experiment.world.data) as viewer:
                experiment.viewer = viewer
                viewer.cam.lookat[:] = [0., 5.6, .7]
                viewer.cam.distance = 17.
                viewer.cam.azimuth, viewer.cam.elevation = 90., -64.
                experiment.run()
        else:
            experiment.run()
    except (PhysicsFailure, ValueError, KeyboardInterrupt) as exc:
        failure = str(exc) or 'viewer closed'
        experiment.event('failure', reason=failure)
    result = experiment.result(failure)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f'{args.case}-{args.policy}-{args.action}-A{args.alpha:g}-F{args.force_cap:g}-L{args.rate:g}'
    if args.inertia is not None or args.resistance is not None:
        mass, resistance = experiment.truth_parameters
        stem += f'-M{mass:g}-R{resistance:g}'
    stem += f'-rule-{args.decision_rule}-tie{args.tie_margin:g}'
    if args.probe_recovery != 'fixed':
        stem += '-recovery-'+args.probe_recovery
    temp = args.output/(stem+'.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    temp.replace(args.output/(stem+'.json'))
    if not args.log_only:
        from report import save_report
        save_report(args.output/(stem+'.html'), result)
    if experiment.dashboard is not None:
        experiment.dashboard.finish(experiment, args.close_on_done)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())


"""Local measured-shape hover clearance and deferred full-envelope egress."""
from itertools import product
from types import SimpleNamespace
import math
import numpy as np
import mujoco
from navigation.geometry import Planner
from navigation.door_clearance import collision_bounds
from navigation.interactions import panel_geometry
from navigation.physics import yaw_error
from probe_recovery import ProbeRecovery, monotone_exit


def hover_planner(e, q):
    """Own per-primitive swept bounds, actual objects; no virtual closed target."""
    w=e.world
    p,v,R,gyro=w.observed()
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,R.ravel())
    if quat[0]<0:quat=-quat
    angle=2*math.acos(float(np.clip(quat[0],-1.,1.)))
    rotations=[]
    for t in np.linspace(0.,1.,max(2,int(math.ceil(angle/.035))+1)):
        z=(1-t)*quat+t*np.array([1.,0.,0.,0.]);z/=np.linalg.norm(z)
        mat=np.empty(9);mujoco.mju_quat2Mat(mat,z);rotations.append(mat.reshape(3,3))
    shapes=[]
    # These are own measured geometries, not the obstacle's hidden dynamics.
    for lo,hi in collision_bounds(w,body_frame=True):
        vertices=np.array([np.where(s,hi,lo) for s in product((False,True),repeat=3)])
        swept=np.concatenate([vertices@r.T for r in rotations])
        # Cover sampled attitude interpolation and future flex/position changes.
        pad=e.cfg.DOOR_POSITION_MARGIN+e.cfg.DOOR_FLEX_MARGIN+float(np.linalg.norm(vertices,axis=1).max())*.035
        lower=swept.min(axis=0)-pad;upper=swept.max(axis=0)+pad
        shapes.append(((lower[:2]+upper[:2])/2,(upper[:2]-lower[:2])/2))
    _,speed=e.state()
    if e.kind=='door':
        margin=np.radians(e.cfg.DOOR_ANGLE_MARGIN_DEG)+abs(speed)*.2
        angles=np.linspace(max(0.,q-margin),min(e.cfg.DOOR_ANGLE_MAX,q+margin),
                           max(2,int(math.ceil(2*margin/np.radians(1.)))+1))
        spacing=float(np.max(np.diff(angles)))
        sweep_pad=2*(e.opening['width_m']+.06)*math.sin(spacing/4)
        panels=[(c,r,h+sweep_pad) for c,r,h in [panel_geometry(e.opening,a)[2] for a in angles]]
    else:
        future=q+speed*.2
        c,r,h=e.panel((q+future)/2)
        panels=[(c,r,h+np.array([0.,abs(future-q)/2+e.cfg.DOOR_POSITION_MARGIN]))]
    mission=getattr(e,'mission',None);known=[]
    if mission is None:sealed=['Oglass','Obypass']
    else:
        sealed=[]
        for oid,record in mission.memory.records.items():
            if oid==e.oid:continue
            if record.traversability=='blocked':sealed.append(oid)
            elif record.semantic=='door':known.append(panel_geometry(record.geometry,record.angle)[2])
            elif record.semantic=='box':
                x,y=record.geometry['center_xy_m'];known.append((np.array([x,y+record.linear_position]),np.eye(2),np.array([.59125,.25])))
    planner=Planner(e.arch,e.cfg,sealed=sealed,panels=known+panels,footprint=np.full(2,1e-8))
    raw=planner.boxes
    # Minkowski obstacles for each individual swept primitive. Keep empty space
    # between the arms instead of enclosing the entire UAV in a flight square.
    planner.boxes=[(c-offset,r,h+half@np.abs(r)) for c,r,h in raw for offset,half in shapes]
    planner.solid_boxes=planner.boxes[:(len(raw)-len(panels))*len(shapes)]
    planner.hover_shapes=[dict(offset=o.tolist(),half=h.tolist()) for o,h in shapes]
    return planner


def contact_release_clear(planner, start, end, outward):
    """Back out of existing contact padding, never enter walls or new panels.

    A primitive may already extend past a padded panel's midplane during
    contact. Requiring its minimum penetration to monotonically decrease would
    incorrectly forbid backing out. Allow that EXISTING convex panel overlap
    only along the measured outward normal; wall/jamb rules stay strict.
    """
    if float((end-start)@outward)<=0:return False
    solids=SimpleNamespace(origin=planner.origin,upper=planner.upper,boxes=planner.solid_boxes)
    if not monotone_exit(solids,start,end):return False
    points=np.linspace(start,end,40)
    for c,r,h in planner.boxes[len(planner.solid_boxes):]:
        inside=np.all(np.abs((points-c)@r)<=h+1e-9,axis=1)
        if not inside[0] and inside.any():return False
    return not planner.collides(end)[0]


def retreat_directions(e, p, q):
    normal=np.array([-np.sin(q),np.cos(q)]) if e.kind=='door' else np.array([0.,1.])
    result=[-normal,np.array([0.,-1.])]
    toward=e.staging-p
    if np.linalg.norm(toward)>.01:result.append(toward/np.linalg.norm(toward))
    # A nearby jamb may require backing away with a small lateral component.
    # All candidates still pass the same swept-body and monotone wall checks.
    outward=-normal
    for angle in (15,-15,30,-30,45,-45):
        a=np.radians(angle);result.append(np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@outward)
    return result


def deferred_egress(e, start, q, local=None):
    """PLAN a connector to the full flight envelope; never move the vehicle."""
    start=np.asarray(start,float)
    current=e.planner(q,'target');bypass=e.planner(q,'bypass')
    if not current.collides(start)[0] and not bypass.collides(start)[0]:
        return np.array([start])
    local=hover_planner(e,q) if local is None else local
    if local.collides(start)[0]:raise ValueError('local hover clearance not yet established')
    for distance in np.arange(.04,1.25,.04):
        for direction in retreat_directions(e,start,q):
            end=start+distance*direction
            probes=end+np.array([[0,0],[.03,0],[-.03,0],[0,.03],[0,-.03]])
            if current.collides(probes).any() or bypass.collides(probes).any():continue
            if not local.clear(start,end) or not monotone_exit(current,start,end):continue
            try:
                bypass_path,_=e.plan(end,q,'bypass')
                candidate_path,_=e.plan(end,e.target,'target')
            except ValueError:continue
            e._exit_certificate=(end.copy(),bypass_path.copy(),candidate_path.copy())
            if hasattr(e,'planning_stats'):e.planning_stats['exit_certificate_replans']=e.planning_stats.get('exit_certificate_replans',0)+1
            return np.array([start,end])
    raise ValueError('no safe deferred exit from local hover pose')

def checked_exit(e,start,q,local):
    """Reuse a route certificate only after CURRENT collision checks pass.

    Recovery does not need a new optimal route every 50 ms. This validates the
    actual connector, end margins and both full paths against current geometry.
    Formal cost comparison still performs a fresh optimal-route calculation.
    """
    certificate=getattr(e,'_exit_certificate',None)
    if certificate is not None:
        end,bp,cp=certificate;current=e.planner(q,'target');bypass=e.planner(q,'bypass')
        future=e.planner(e.target,'target')
        probes=end+np.array([[0,0],[.03,0],[-.03,0],[0,.03],[0,-.03]])
        clear=(not local.collides(start)[0] and not current.collides(probes).any()
            and not bypass.collides(probes).any() and local.clear(start,end) and monotone_exit(current,start,end))
        if clear and all(bypass.clear(a,b) for a,b in zip(bp,bp[1:])) and all(future.clear(a,b) for a,b in zip(cp,cp[1:])):
            if hasattr(e,'planning_stats'):e.planning_stats['exit_certificate_reuses']=e.planning_stats.get('exit_certificate_reuses',0)+1
            return np.array([start,end])
    return deferred_egress(e,start,q,local)


class LocalHoverRecovery(ProbeRecovery):
    """Brake as soon as a nearby hover is safe; bypass retreat stays hypothetical."""
    def command(self):
        e,w=self.e,self.e.world
        p,v,R,gyro=w.observed();q,_=e.state()
        if w.data.time>=self.next_check:
            self.next_check=float(w.data.time)+.05
            local=hover_planner(e,q)
            clear=not local.collides(p[:2])[0]
            detached=w.force<.1 and e.last_force<.1
            if not detached and np.linalg.norm(v)<.02:
                if self.contact_stall_since is None:self.contact_stall_since=float(w.data.time)
            else:self.contact_stall_since=None
            contact_stall=0. if self.contact_stall_since is None else float(w.data.time-self.contact_stall_since)
            bounds=collision_bounds(w)
            actual=np.max([np.maximum(abs(lo),abs(hi)) for lo,hi in bounds],axis=0)
            envelope_ok=bool(np.all(actual<=e.envelope.half-e.cfg.DOOR_POSITION_MARGIN))
            speed_ok=np.linalg.norm(v)<.065;gyro_ok=np.linalg.norm(gyro)<.16
            tilt_ok=R[2,2]>=math.cos(math.radians(10));yaw_ok=abs(yaw_error(R,0.))<.10;height_ok=abs(p[2]-1.25)<.065
            stable=bool(speed_ok and gyro_ok and tilt_ok and yaw_ok and height_ok)
            if not detached and stable and contact_stall>1.:
                raise ValueError('persistent contact with no local unloading progress; use safe recovery fallback')
            exit_ok=False;planned_exit=0.;exit_checked=False
            if clear and detached:
                # Freeze a local braking point. Do NOT chase the bypass start.
                if self.anchor is None or self.translating:
                    stopping=p[:2]+.30*v[:2]
                    self.anchor=stopping if local.clear(p[:2],stopping) else p[:2].copy()
                self.translating=False
                if stable and envelope_ok:
                    exit_checked=True
                    try:
                        prefix=checked_exit(e,p[:2],q,local)
                        planned_exit=float(np.linalg.norm(np.diff(prefix,axis=0),axis=1).sum())
                        exit_ok=True
                    except ValueError:pass
            else:
                self.anchor=None;self.translating=True
                # Only seek local hover clearance. A full route is checked AFTER
                # braking, and its connector is flown only if bypass is chosen.
                directions=retreat_directions(e,p[:2],q)
                candidates=np.array([p[:2]+distance*direction for distance in np.arange(.02,.81,.02)
                                     for direction in directions])
                for end in candidates[~local.collides(candidates)]:
                    if not contact_release_clear(local,p[:2],end,directions[0]):continue
                    self.anchor=end;break
                if self.anchor is None:raise ValueError('no nearby local hover clearance')
            ok=bool(detached and stable and envelope_ok and clear and exit_ok)
            if not ok:self.stable_since=None
            elif self.stable_since is None:self.stable_since=float(w.data.time)
            self.ready=bool(ok and w.data.time-self.stable_since>=self.dwell)
            self.last_status=dict(detached=bool(detached),stable=bool(stable),envelope_ok=envelope_ok,
                clearance=bool(clear),exit_path=exit_ok,local_hover=True,exit_deferred=True,
                speed_ok=bool(speed_ok),gyro_ok=bool(gyro_ok),tilt_ok=bool(tilt_ok),yaw_ok=bool(yaw_ok),height_ok=bool(height_ok),
                exit_path_checked=exit_checked,
                contact_stall_s=contact_stall,
                planned_exit_m=planned_exit,dwell_s=0. if self.stable_since is None else float(w.data.time-self.stable_since))
            feedback=dict(time=float(w.data.time),**self.last_status,
                position_xy=p[:2].tolist(),speed_mps=float(np.linalg.norm(v)),
                gyro_radps=float(np.linalg.norm(gyro)),contact_N=float(w.force),ready=self.ready,
                tilt_deg=float(np.degrees(np.arccos(np.clip(R[2,2],-1.,1.)))),
                yaw_error_rad=abs(yaw_error(R,0.)),height_error_m=abs(float(p[2]-1.25)),
                hover_shapes=getattr(local,'hover_shapes',[]))
            e.recovery_frames.append(feedback)
            if hasattr(e,'on_recovery_status'):e.on_recovery_status(feedback)
        w.target=np.r_[self.anchor,1.25]
        desired=1.25*(self.anchor-p[:2]);speed=np.linalg.norm(desired)
        if speed>.15:desired*=.15/speed
        # Match the existing flight position-loop gain and limit requested tilt
        # during braking; chasing a close anchor with high gains excites flex.
        acceleration=3.2*(desired-v[:2]);magnitude=np.linalg.norm(acceleration)
        if magnitude>.8:acceleration*=.8/magnitude
        return w.mass*acceleration

    def __init__(self,e):
        super().__init__(e)
        self.translating=True
        self.contact_stall_since=None

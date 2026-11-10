"""Brake and recover attitude directly toward the certified passage connection.

There is no intermediate hover anchor. A connection stays fixed while valid,
with forward upgrades if object coasting removes a backward release need.
Avoidance keeps its original recovery controller.
"""
import math
import numpy as np
from passage_completion import passage_geometry,current_passage,certificate_for
from local_hover import contact_release_clear,retreat_directions
from probe_recovery import ProbeRecovery
from navigation.door_clearance import collision_bounds
from navigation.physics import yaw_error,PhysicsFailure
from motion_reference import SmoothFlightReference,append_motion_sample


def certify_target(e,geometry,end):
    q,_,p,planner,local,field=geometry;end=np.asarray(end,float)
    probes=end+np.array([[0,0],[.03,0],[-.03,0],[0,.03],[0,-.03]])
    if planner.collides(probes).any() or local.collides(end)[0]:raise ValueError('connection endpoint lacks current envelope clearance')
    outward=retreat_directions(e,p,q)[0]
    if not local.clear(p,end) and not contact_release_clear(local,p,end,outward):raise ValueError('connection segment lost swept-body clearance')
    field.cost(end)
    route,_=planner.plan(end,e.goal)
    return certificate_for(e,geometry,end,route)


def select_target(e,geometry):
    base=current_passage(e,geometry);p=geometry[2];velocity=e.world.observed()[1][:2]
    speed=float(np.linalg.norm(velocity));direction=velocity/max(speed,1e-8)
    acceleration=min(.8,e.cap/e.world.mass)
    # Include a short rotor-attitude response allowance in the candidate
    # estimate. This is a control lookahead, not a measurement error bound.
    stop=p+velocity*(.25+speed/(2*max(acceleration,.1)))
    path=np.array(base['passage_path']);candidates=[path[0],stop];travel=0.
    # Check route samples and the measured stopping estimate using the same
    # current-geometry certificate. Prefer enough room for braking.
    for a,b in zip(path,path[1:]):
        length=float(np.linalg.norm(b-a))
        for distance in np.arange(.05,length+.0001,.05):
            if travel+distance>.8:break
            candidates.append(a+(b-a)*(distance/max(length,1e-9)))
        travel+=length
        if travel>.8:break
        candidates.append(b)
    ranked=[]
    for end in candidates:
        if np.linalg.norm(end-p)>1.2:continue
        behind=max(0.,float((stop-end)@direction)) if speed>.04 else 0.
        score=float(np.linalg.norm(end-p))+4*behind
        ranked.append((score,end))
    for _,end in sorted(ranked,key=lambda item:item[0]):
        try:return certify_target(e,geometry,end),stop
        except ValueError:continue
    raise ValueError('no braking-aware certified passage connection')


class PassageConnectionRecovery(ProbeRecovery):
    """One connection target for unloading, braking, attitude and handoff."""
    policy='direct_passage_connection'

    def __init__(self,e):
        super().__init__(e);self.target=None;self.certificate=None;self.handoff_certificate=None
        self.target_history=[];self.plan_count=0;self.translating=True;self.contact_stall_since=None
        self.allow_forward_upgrade=False
        self.progress_direction=(e.goal-self.start_xy)/max(float(np.linalg.norm(e.goal-self.start_xy)),1e-8)
        self.reference=getattr(e,'flight_reference',None) or SmoothFlightReference(e.world)
        self.yaw_command=self.reference.heading;self.connecting=False;self.actual_attachment=False

    def command(self):
        e,w=self.e,self.e.world;p,v,R,gyro=w.observed();now=float(w.data.time)
        if e.last_force>8.:raise PhysicsFailure('contact impact during direct passage release')
        if now>=self.next_check:
            self.next_check=now+.10
            geometry=passage_geometry(e);local=geometry[4];replanned=False;reason='initial_connection'
            if self.target is not None:
                try:self.certificate=certify_target(e,geometry,self.target)
                except ValueError as exc:reason=str(exc);self.target=None
            if self.target is not None and self.allow_forward_upgrade:
                # The object can coast away while we brake. If its CURRENT
                # geometry now admits a forward connection, avoid returning to
                # an obsolete backward release endpoint. Require at least
                # 10 cm of certified forward improvement; never move it back.
                try:
                    improved,stop=select_target(e,geometry);end=np.array(improved['release_endpoint'])
                    if ((end-self.target)@self.progress_direction>.10
                            and (end-self.start_xy)@self.progress_direction>=-.01):
                        self.target=None
                        reason='newly_clear_forward_connection'
                except ValueError:pass
            if self.target is None:
                self.certificate,self.stop_point=select_target(e,geometry)
                self.target=np.array(self.certificate['release_endpoint']);self.plan_count+=1;replanned=True
                if self.plan_count==1:self.allow_forward_upgrade=bool((self.target-self.start_xy)@self.progress_direction<-.04)
                self.target_history.append(dict(time=now,target_xy=self.target.tolist(),reason=reason))
                e.event('passage_connection_selected',target_xy=self.target.tolist(),
                    measured_velocity_xy=v[:2].tolist(),estimated_stopping_xy=self.stop_point.tolist(),
                    reason=reason,selection=self.plan_count,hover_anchor_used=False)
            clear=not local.collides(p[:2])[0];detached=w.force<.1 and e.last_force<.1
            bounds=collision_bounds(w);actual=np.max([np.maximum(abs(lo),abs(hi)) for lo,hi in bounds],axis=0)
            envelope_ok=bool(np.all(actual<=e.envelope.half-e.cfg.DOOR_POSITION_MARGIN))
            speed_ok=np.linalg.norm(v)<.065;gyro_ok=np.linalg.norm(gyro)<.16
            tilt_ok=R[2,2]>=math.cos(math.radians(10));yaw_ok=abs(yaw_error(R,0.))<.10;height_ok=abs(p[2]-1.25)<.065
            stable=bool(speed_ok and gyro_ok and tilt_ok and yaw_ok and height_ok)
            path_ok=False;self.handoff_certificate=None;self.actual_attachment=False
            if clear and detached and envelope_ok:
                try:
                    handoff=current_passage(e,geometry)
                    # The ACTUAL current pose must attach directly. No extra
                    # connector is deferred to a later, separate flight stage.
                    if np.linalg.norm(np.diff(np.array(handoff['release_path']),axis=0))<1e-6:
                        self.handoff_certificate=handoff;path_ok=True;self.actual_attachment=True
                except ValueError:pass
            ok=bool(detached and stable and envelope_ok and clear and path_ok)
            if not ok:self.stable_since=None
            elif self.stable_since is None:self.stable_since=now
            self.ready=bool(ok and now-self.stable_since>=self.dwell)
            self.last_status=dict(detached=bool(detached),stable=stable,envelope_ok=envelope_ok,
                clearance=bool(clear),exit_path=path_ok,local_hover=False,exit_deferred=False,
                speed_ok=bool(speed_ok),gyro_ok=bool(gyro_ok),tilt_ok=bool(tilt_ok),yaw_ok=bool(yaw_ok),height_ok=bool(height_ok),
                exit_path_checked=bool(clear and detached and stable and envelope_ok),contact_stall_s=0.,
                planned_exit_m=float(np.linalg.norm(self.target-p[:2])),
                dwell_s=0. if self.stable_since is None else now-self.stable_since,
                recovery_policy=self.policy,connection_target_xy=self.target.tolist(),hover_anchor_used=False,
                motion_mode=self.reference.last.get('motion_mode','BRAKE_TURN'),
                connection_swept_clear=True,connection_motion_margin_checked=self.certificate['motion_margin_checked'],
                connection_plan_count=self.plan_count,connection_replanned=replanned)
            feedback=dict(time=now,**self.last_status,position_xy=p[:2].tolist(),
                speed_mps=float(np.linalg.norm(v)),gyro_radps=float(np.linalg.norm(gyro)),contact_N=float(w.force),ready=self.ready,
                tilt_deg=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))),yaw_error_rad=abs(yaw_error(R,0.)),
                height_error_m=abs(float(p[2]-1.25)),hover_shapes=getattr(local,'hover_shapes',[]))
            e.recovery_frames.append(feedback)
            if hasattr(e,'on_recovery_status'):e.on_recovery_status(feedback)
        w.target=np.r_[self.target,1.25]
        if self.actual_attachment:
            desired=np.zeros(2);mode='BRAKE_TURN'
        else:
            aligned=(np.linalg.norm(v)<.18 and np.linalg.norm(gyro)<.35
                and R[2,2]>math.cos(math.radians(10)) and abs(yaw_error(R,0.))<.18)
            self.connecting=self.connecting or aligned
            desired=1.25*(self.target-p[:2]) if self.connecting else np.zeros(2)
            speed=float(np.linalg.norm(desired))
            if speed>.18:desired*=.18/speed
            mode='CONNECT' if self.connecting else 'BRAKE_TURN'
        w.stage='PASSAGE_CONNECT' if mode=='CONNECT' else 'PASSAGE_RELEASE'
        force=self.reference.command(w,desired,0.,mode);self.yaw_command=self.reference.heading
        append_motion_sample(e,self.reference)
        return force

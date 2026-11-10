"""Debounced contact / detached coasting / guarded reapproach states."""
from types import SimpleNamespace
import numpy as np
from local_hover import hover_planner
from probe_recovery import monotone_exit
from navigation.physics import clip_norm,yaw_error
from motion_reference import SmoothFlightReference,append_motion_sample


class ContactTracking:
    off_force=.03
    on_force=.10
    off_dwell=.04
    on_dwell=.02

    def __init__(self,e):
        self.e=e;self.state='contact';self.off_since=None;self.on_since=None
        self.started=float(e.world.data.time);self.hold_heading=None
        self.next_geometry_check=0.;self.cruise_heading_safe=False

    def update(self):
        e,w=self.e,self.e.world;now=float(w.data.time);force=e.last_force
        # Initial touch can bounce before an informative movement window even
        # exists. It remains contact acquisition. Arm the detached-push logic
        # once the online decision has committed to continued interaction.
        if self.state=='contact' and not e.committed:
            self.off_since=None;return self.state
        if self.state=='contact':
            self.off_since=now if force<self.off_force and self.off_since is None else self.off_since
            if force>=self.off_force:self.off_since=None
            if self.off_since is not None and now-self.off_since>=self.off_dwell-1e-9:
                self.state='coast';self.started=now
                e.flight_reference=SmoothFlightReference(w)
                self.hold_heading=e.flight_reference.heading
                p,v,_,_=w.observed()
                e.event('contact_lost',low_force_since=self.off_since,confirmation_s=now-self.off_since,
                    contact_N=float(force),position_xy=p[:2].tolist(),velocity_xy=v[:2].tolist(),
                    speed_mps=float(np.linalg.norm(v)),previous_horizontal_N=w.last_horizontal.tolist())
        else:
            self.on_since=now if force>self.on_force and self.on_since is None else self.on_since
            if force<=self.on_force:self.on_since=None
            if self.on_since is not None and now-self.on_since>=self.on_dwell-1e-9:
                self.state='contact';self.off_since=None;self.on_since=None
                e.event('contact_reacquired',contact_N=float(force),confirmation_s=self.on_dwell)
                # Resume the contact controller through an explicit force ramp.
                e.contact_blend_time=now;e.contact_blend_force=w.last_horizontal.copy()
        return self.state

    def command(self,point,tangent,normal):
        e,w=self.e,self.e.world;p,v,R,g=w.observed();_,object_speed=e.state()
        goal=np.zeros(2);mode='COAST_BRAKE';q=e.state()[0]
        if w.data.time>=self.next_geometry_check:
            self.next_geometry_check=float(w.data.time)+.10
            local=hover_planner(e,q)
            self.cruise_heading_safe=not local.collides(p[:2])[0]
        # Coasting observations remain useful. Recontact is only needed once
        # the object stops short, the current majority still supports pushing,
        # and own attitude/speed permit another gentle approach.
        can_reapproach=(e.online_choice=='continue' and abs(object_speed)*e.arm<.02
            and np.linalg.norm(v)<.12 and np.linalg.norm(g)<.30 and R[2,2]>.985)
        if can_reapproach:
            gap=float((point-p[:2])@normal)
            goal=clip_norm(.7*float((point-p[:2])@tangent)*tangent,.06)
            if abs(yaw_error(R,q if e.kind=='door' else 0.))<.15:
                goal+=float(np.clip(.6*(gap-.22),0.,.055))*normal
            local=hover_planner(e,q)
            solids=SimpleNamespace(origin=local.origin,upper=local.upper,boxes=local.solid_boxes)
            if not monotone_exit(solids,p[:2],p[:2]+goal*.25):goal[:]=0.
            mode='REAPPROACH'
        w.stage='ONLINE_'+mode
        heading=(q if e.kind=='door' else 0.) if mode=='REAPPROACH' else (0. if self.cruise_heading_safe else self.hold_heading)
        force=e.flight_reference.command(w,goal,heading,mode)
        append_motion_sample(e,e.flight_reference)
        e.mission.status='脱开滑行 / 制动并继续观测' if mode=='COAST_BRAKE' else '脱开后低速重新接触'
        e.mission.reason='接触力已消失，暂停续推偏置；运动窗口仍在线更新权重与边界'
        e.mission.action='Observe' if mode=='COAST_BRAKE' else 'Approach'
        return force,e.flight_reference.heading

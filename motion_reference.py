"""Continuous velocity, acceleration and yaw references after contact loss.

Uses own measured velocity/attitude and the preceding rotor force command.
No object dynamics parameter or simulator reset enters this controller.
"""
import math
import numpy as np
from navigation.physics import clip_norm,yaw_error


class SmoothFlightReference:
    velocity_accel_limit=.35
    acceleration_limit=.80
    jerk_limit=12.0
    yaw_rate_limit=.60
    yaw_accel_limit=1.20

    def __init__(self,w):
        _,v,R,g=w.observed()
        self.velocity=v[:2].copy()
        self.acceleration=w.last_horizontal.copy()/w.mass
        self.heading=math.atan2(R[1,0],R[0,0])
        self.yaw_rate=float(np.clip(g[2],-self.yaw_rate_limit,self.yaw_rate_limit))
        self.time=float(w.data.time);self.last={}
        self.started=self.time;self.initial_velocity=self.velocity.copy()
        self.max_velocity_slew=0.;self.max_force_jerk=0.;self.max_yaw_accel=0.

    def command(self,w,goal_velocity,goal_heading,mode):
        now=float(w.data.time);dt=max(0.,min(.02,now-self.time));self.time=now
        _,v,R,_=w.observed();goal=np.asarray(goal_velocity,float)
        previous=self.velocity.copy()
        self.velocity+=clip_norm(goal-self.velocity,self.velocity_accel_limit*dt)
        feedforward=(self.velocity-previous)/dt if dt>1e-9 else np.zeros(2)
        # Own attitude predicts the acceleration already being produced by the
        # rotors. Anticipate it instead of continuing a stale braking command
        # until measured velocity has already reversed.
        actual_acceleration=R[:2,2]*float(np.sum(w.ctrl))/w.mass
        predicted_velocity=v[:2]+.25*actual_acceleration
        desired=clip_norm(feedforward+1.6*(self.velocity-predicted_velocity),self.acceleration_limit)
        old_acceleration=self.acceleration.copy()
        self.acceleration+=clip_norm(desired-self.acceleration,self.jerk_limit*dt)
        error=(goal_heading-self.heading+math.pi)%(2*math.pi)-math.pi
        rate=float(np.clip(1.4*error,-self.yaw_rate_limit,self.yaw_rate_limit))
        old_yaw_rate=self.yaw_rate
        self.yaw_rate+=float(np.clip(rate-self.yaw_rate,-self.yaw_accel_limit*dt,self.yaw_accel_limit*dt))
        self.heading+=self.yaw_rate*dt
        if dt>1e-9:
            self.max_velocity_slew=max(self.max_velocity_slew,float(np.linalg.norm(self.velocity-previous)/dt))
            self.max_force_jerk=max(self.max_force_jerk,float(np.linalg.norm(self.acceleration-old_acceleration)/dt))
            self.max_yaw_accel=max(self.max_yaw_accel,abs(self.yaw_rate-old_yaw_rate)/dt)
        self.last=dict(time=now,motion_mode=mode,velocity_xy=v[:2].tolist(),
            reference_velocity_xy=self.velocity.tolist(),goal_velocity_xy=goal.tolist(),
            acceleration_command_xy=self.acceleration.tolist(),yaw_reference=self.heading,
            attitude_acceleration_xy=actual_acceleration.tolist(),predicted_velocity_xy=predicted_velocity.tolist(),
            yaw_reference_rate=self.yaw_rate,reference_dt_s=dt,
            reference_started_time=self.started,initial_reference_velocity_xy=self.initial_velocity.tolist(),
            max_velocity_slew=self.max_velocity_slew,max_force_jerk=self.max_force_jerk,max_yaw_accel=self.max_yaw_accel,
            reference_accel_limit=self.velocity_accel_limit,jerk_limit=self.jerk_limit)
        return w.mass*self.acceleration


def append_motion_sample(e,reference):
    now=float(e.world.data.time)
    if now<getattr(e,'next_motion_sample',0.):return
    e.next_motion_sample=now+.05
    p,v,R,g=e.world.observed()
    e.motion_frames.append(dict(**reference.last,position_xy=p[:2].tolist(),stage=e.world.stage,
        speed_mps=float(np.linalg.norm(v)),contact_N=float(e.last_force),
        tilt_deg=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))),
        yaw_error_rad=abs(yaw_error(R,0.)),object_q=float(e.state()[0])))

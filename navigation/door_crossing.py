"""Continuous rotor recovery, stable panel, then guarded geometric passage."""
import math
import numpy as np
from .interactions import recover
from .door_clearance import collision_bounds
from .physics import PhysicsFailure, yaw_error


def cross_door(m,rec,panel,*,measurement='lidar panel fit'):
    w=m.world;o=rec.geometry;solver=m.clearance_solver;cfg=m.cfg;oid=rec.stable_id
    solution=solver.solve(o)
    if not solution['feasible']:raise PhysicsFailure('door_has_no_vehicle_clearance')
    m.status='净空通行 / 脱离与对正'
    # Only the existing rotor backoff; no state reset or further door actuation.
    recover(w,o,panel)
    w.align_yaw(0.,stage='ALIGN_DOOR_PASS')
    anchor=w.observed()[0];deadline=w.data.time+4.;stable_since=None
    w.target=anchor.copy();w.stage='WAIT_DOOR_CLEAR'
    while w.data.time<deadline:
        w.step(yaw=0.)
        if w.step_count%25:continue
        angle,omega=panel()
        if abs(omega)<=cfg.DOOR_PASS_MAX_OMEGA:
            if stable_since is None:stable_since=w.data.time
            if w.data.time-stable_since>=.2:break
        else:stable_since=None
    else:raise PhysicsFailure('door_not_stable_for_passage')
    if measurement=='lidar panel fit' and w.data.time-rec.angle_time>.25:
        raise PhysicsFailure('stale_panel_geometry_before_crossing')
    angle,omega=panel()
    if angle+1e-9<solution['theta_clear']:raise PhysicsFailure('door_clearance_lost_before_passage')
    path=solver.path(o,angle,w.observed()[0]);m.path=path
    event=dict(angle_rad=angle,omega_rad_s=omega,theta_clear=solution['theta_clear'],path=path,
               footprint_half_m=solver.envelope.half[:2],measurement=measurement,
               source='post-recovery measured panel + same computed envelope')
    m.memory.event(w.data.time,oid,'computed_clearance_passage_start',**event)
    m.status='满足机体净空 / 穿门';m.reason='机体碰撞包络 + 距离余量 + 当前门扇路径检查'
    previous=m.crossing_guard
    cached_angle=angle;cached_motion=abs(omega)*cfg.DOOR_PASS_LOOKAHEAD
    planner=solver.planner(o,angle,tracking_allowance=False,extra_angle_margin=cached_motion)

    def guard(world):
        nonlocal planner,cached_angle,cached_motion
        if previous:previous(world)
        p,v,R,g=world.observed();q,omega_now=panel()
        # Keep the actual own flexible body inside the envelope used to plan.
        bounds=collision_bounds(world)
        actual=np.max([np.maximum(np.abs(lo),np.abs(hi)) for lo,hi in bounds],axis=0)
        if np.any(actual>solver.envelope.half-cfg.DOOR_POSITION_MARGIN+1e-6):
            raise PhysicsFailure('actual_vehicle_exceeds_passage_envelope')
        if math.degrees(math.acos(float(np.clip(R[2,2],-1,1))))>cfg.DOOR_PASS_TILT_DEG+1e-6:
            raise PhysicsFailure('passage_tilt_limit')
        if abs(yaw_error(R,0.))>np.radians(cfg.DOOR_PASS_YAW_DEG):
            raise PhysicsFailure('passage_yaw_limit')
        motion=abs(omega_now)*cfg.DOOR_PASS_LOOKAHEAD
        if abs(q-cached_angle)>=np.radians(.25) or abs(motion-cached_motion)>=np.radians(.1):
            planner=solver.planner(o,q,tracking_allowance=False,extra_angle_margin=motion)
            cached_angle=q;cached_motion=motion
            if abs(omega_now)>cfg.DOOR_PASS_MAX_OMEGA:
                m.memory.event(w.data.time,oid,'passage_motion_envelope',angle_rad=q,
                               measured_omega_rad_s=omega_now,extra_angle_margin_rad=motion,
                               measurement=measurement)
        index=int(np.argmin(np.linalg.norm(path-world.target[:2],axis=1)))
        remaining=np.vstack((p[:2],path[index:]))
        if any(not planner.clear(a,b) for a,b in zip(remaining[:-1],remaining[1:])):
            m.memory.event(w.data.time,oid,'passage_geometry_guard_blocked',angle_rad=q,
                           position=p,remaining_path=remaining,footprint_half_m=planner.footprint)
            raise PhysicsFailure('remaining_door_passage_path_blocked')
    m.crossing_guard=guard
    try:w.fly(path[1:],yaw=0.,speed=min(cfg.flight_speed,cfg.DOOR_PASS_SPEED),stage='DOOR_PASS',
              position_tolerance=min(.015,max(.005,cfg.DOOR_POSITION_MARGIN/2)))
    finally:m.crossing_guard=previous
    rec.traversability='free';rec.verification_state='verified_free';rec.confirmation='confirmed'
    m.memory.event(w.data.time,oid,'door_geometry_clear',estimated_angle_rad=panel()[0],
                   theta_clear=solution['theta_clear'],source='completed guarded geometric passage')
    m.save_key(oid+'_crossed')

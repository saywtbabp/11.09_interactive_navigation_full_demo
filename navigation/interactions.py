"""Rotor-driven Probe/Push/backoff; panel state is supplied through an observation interface."""
import math
import numpy as np
from .physics import clip_norm,PhysicsFailure
from .geometry import Planner,rectangle

def panel_geometry(opening,angle):
    x,y=opening['center_xy_m'];width=opening['width_m']-.18
    sensed=opening.get('sensed_geometry')
    pivot=np.array(sensed['hinge_xy_m'] if sensed else [x-width/2,y+.035]);c,s=math.cos(angle),math.sin(angle)
    rot=np.array([[c,-s],[s,c]])
    offset=sensed['leaf_center_offset_m'] if sensed else -.035
    return pivot,width,(pivot+rot@np.array([width/2,offset]),rot,np.array([width/2,.025]))

def approach(opening):
    x,y=opening['center_xy_m'];return np.array([x+.24375,y-.95,1.25])

def contact_pose(opening):
    x,y=opening['center_xy_m'];return np.array([x+.24375,y-.4025,1.25])

def merge_pose(opening):
    x,y=opening['center_xy_m'];return np.array([x+.24375,y+1.70,1.25])

def probe(w,opening,panel_state,progress=None):
    cfg=w.cfg;anchor=contact_pose(opening);w.target=anchor.copy();w.stage='PROBE'
    angle0=panel_state()[0];start=float(w.data.time);peak=0.;samples=[];next_sample=.05
    while w.data.time-start<cfg.probe_duration-1e-9:
        pos,vel,_,_=w.observed()
        f=clip_norm(np.array([4*(anchor[0]-pos[0])-3*vel[0],max(0.,cfg.probe_force_max-1.2*vel[1])]),cfg.probe_force_max)
        w.step(f,yaw=0.);peak=max(peak,w.force)
        if w.data.time-start>=next_sample-1e-9:
            angle=panel_state()[0];z=np.array([peak,angle-angle0]);samples.append(z)
            if progress:progress(float(w.data.time-start),z)
            next_sample+=.05
    return np.array([peak,panel_state()[0]-angle0]),np.asarray(samples)

def push(w,opening,panel_state):
    start=w.data.time;cfg=w.cfg;braking=False;anchor=None;eligible=panel_state()[0]<math.radians(20)
    solver=getattr(w,'door_clearance_solver',None)
    target=cfg.THETA_CLEAR
    if solver:
        result=solver.solve(opening)
        if not result['feasible']:return False
        target=result['theta_clear']
    while w.data.time-start<cfg.push_timeout:
        angle,omega=panel_state();pos,vel,R,gyro=w.observed()
        if angle>=target:return True
        if eligible and not braking and angle>=math.radians(20) and omega>=.75:
            braking=True;anchor=pos.copy()
        elif braking and omega<.15 and np.linalg.norm(vel)<.35 and R[2,2]>math.cos(math.radians(15)):
            braking=False
        if braking:
            w.stage='PUSH_BRAKE';w.target=anchor.copy()
            f=clip_norm(2*(anchor[:2]-pos[:2])-3*vel[:2],cfg.push_force_max);w.step(f)
        else:
            w.stage='PUSH';pivot,width,_=panel_geometry(opening,angle)
            tangent=np.array([math.cos(angle),math.sin(angle)]);normal=np.array([-math.sin(angle),math.cos(angle)])
            contact=pivot+(.24375+width/2)*tangent-.035*normal
            ft=4*float((contact-pos[:2])@tangent)-3*float(vel[:2]@tangent)
            fn=max(0.,cfg.push_force_max-1.2*float(vel[:2]@normal))
            f=clip_norm(ft*tangent+fn*normal,cfg.push_force_max)
            w.target=np.r_[contact-.15*normal,1.25];w.step(f,yaw=angle)
    return False

def recover(w,opening,panel_state,bypass=False):
    pos,vel,R,gyro=w.observed();angle=panel_state()[0];normal=np.array([-math.sin(angle),math.cos(angle)])
    point=pos.copy();point[:2]-=.38*normal;point[0]=opening['center_xy_m'][0]+.24375
    if bypass:point[1]=min(point[1],opening['center_xy_m'][1]-.95)
    point[2]=1.25;w.target=point;w.stage='BACKOFF'
    w.yaw_target=math.atan2(R[1,0],R[0,0]);start=w.data.time;stable_since=None
    while w.data.time-start<10:
        w.step()
        if w.step_count%25:continue
        pos,vel,R,gyro=w.observed();angle=panel_state()[0]
        stable=(w.force<.1 and R[2,2]>math.cos(math.radians(15)) and np.linalg.norm(gyro)<.8
                and np.linalg.norm(vel)<.35 and abs(pos[2]-1.25)<.15)
        if not stable:stable_since=None;continue
        if stable_since is None:stable_since=w.data.time
        if w.data.time-stable_since<.15:continue
        panel=panel_geometry(opening,angle)[2]
        try:
            solver=getattr(w,'door_clearance_solver',None)
            planner=(solver.planner(opening,angle) if solver and not bypass else
                     Planner(w.arch,w.cfg,panels=[panel],row=opening['center_xy_m'][1]))
            goal=approach(opening) if bypass else merge_pose(opening)
            path,_=planner.plan(pos,goal)
            return path
        except ValueError:stable_since=None
    raise PhysicsFailure('recovery_timeout')

def route(w,opening,panel_state,kind,geo):
    """Return comparable time to stable common M; incomplete routes cost infinity."""
    start=float(w.data.time);reason='';success=False
    try:
        if kind=='door':
            if w.observed()[0][1]<opening['center_xy_m'][1]-.60:
                w.fly([contact_pose(opening)],yaw=0.,speed=.35)
            if not push(w,opening,panel_state):raise PhysicsFailure('push_timeout')
            path=recover(w,opening,panel_state)
            w.fly(path[1:],yaw=0.)
        else:
            if w.force>.1 or w.observed()[0][1]>opening['center_xy_m'][1]-.65:
                path=recover(w,opening,panel_state,bypass=True);w.fly(path[1:],yaw=0.)
            planner=Planner(w.arch,w.cfg,sealed=[opening['id']],row=opening['center_xy_m'][1])
            path,_=planner.plan(w.observed()[0],merge_pose(opening));w.fly(path[1:])
        success=True
        if w.wall_peak>.1:success=False;reason='unexpected_wall_contact'
    except (PhysicsFailure,ValueError) as exc:reason=str(exc)
    elapsed=float(w.data.time-start)
    return dict(time_s=elapsed if success else float('inf'),elapsed_s=elapsed,success=success,reason=reason,
                contact_peak_N=w.contact_peak,wall_peak_N=w.wall_peak,position=w.observed()[0].tolist())

def observed_mechanical_state(w,opening,panel_state):
    """Explicit ideal full-body proprioception of OWN flexible vehicle.

    Five free-body states represent a simulation instrument, not VIO. No door
    qpos/qvel, environmental parameters or unrelated model state are copied.
    Door state is the independently fitted lidar estimate.
    """
    import mujoco
    robot=[]
    for jid in range(w.model.njnt):
        if w.model.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE and int(w.model.jnt_bodyid[jid]) in w.robot_ids:
            q=int(w.model.jnt_qposadr[jid]);v=int(w.model.jnt_dofadr[jid])
            robot.append(dict(joint=mujoco.mj_id2name(w.model,mujoco.mjtObj.mjOBJ_JOINT,jid),
                              position_quaternion=w.data.qpos[q:q+7].tolist(),velocity=w.data.qvel[v:v+6].tolist()))
    a,omega=panel_state()
    return dict(robot=robot,panel_angle_rad=float(a),panel_speed_rad_s=float(omega),time=float(w.data.time),
                yaw_reference=float(w.yaw_reference),
                source='ideal own multibody proprioception + lidar panel geometry; not environmental state clone')

def initialize_prediction(w,state,opening,theta):
    """One-time candidate clone initialization, never an online motion action."""
    import mujoco
    m,d=w.model,w.data;mujoco.mj_resetData(m,d)
    oid=opening['id'];jid=m.joint('hinge_'+oid).id;qa=int(m.jnt_qposadr[jid]);da=int(m.jnt_dofadr[jid])
    body=m.body('panel_'+oid).id;I,friction=theta;width=opening['width_m']-.18
    mass=float(I/(width**2/3+.05**2/12+.035**2));m.body_mass[body]=mass
    m.body_inertia[body]=mass/12*np.array([.05**2+4,width**2+4,width**2+.05**2])
    m.dof_frictionloss[da]=friction;mujoco.mj_setConst(m,d)
    for r in state['robot']:
        j=m.joint(r['joint']).id;q=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j])
        d.qpos[q:q+7]=r['position_quaternion'];d.qvel[v:v+6]=r['velocity']
    d.qpos[qa]=state['panel_angle_rad'];d.qvel[da]=state['panel_speed_rad_s']
    d.time=state['time'];w.yaw_reference=w.yaw_target=state['yaw_reference'];w.step_count=0
    w.contact_peak=w.wall_peak=0.;mujoco.mj_forward(m,d)
    return lambda:(float(d.qpos[qa]),float(d.qvel[da]))

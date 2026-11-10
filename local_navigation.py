"""Rolling, common-endpoint local legs; no distant-goal cost-to-go."""
import numpy as np


def leg_for_opening(arch,cfg,opening):
    y=float(opening['center_xy_m'][1]);bounds=arch['bounds_wall_centerlines_m']
    rows=sorted(float(r['y_m']) for r in arch['internal_walls'])
    previous=max([bounds['ymin']]+[r for r in rows if r<y-.01])
    following=min([bounds['ymax']]+[r for r in rows if r>y+.01])
    # Common merge point for BOTH alternatives. Never move it with the UAV or
    # choose it from hidden object classes / physical parameters.
    distance=min(cfg.local_merge_distance_m,following-y-.85)
    if distance<.65:raise ValueError('corridor too short for a local merge point')
    goal=np.array([(bounds['xmin']+bounds['xmax'])/2,y+distance])
    lower=previous+.10 if previous>bounds['ymin'] else bounds['ymin']
    region=(bounds['xmin'],bounds['xmax'],lower,goal[1]+.05)
    return dict(row_y=y,goal_xy=goal.tolist(),region=list(region),scope='current_wall_to_common_local_merge')


def next_leg(arch,cfg,position):
    y=float(position[1]);rows=sorted(arch['internal_walls'],key=lambda r:r['y_m'])
    row=next((r for r in rows if r['y_m']>y-.55),None)
    if row is None:
        return dict(row_y=None,goal_xy=list(arch['goal_xyz_m'][:2]),region=None,scope='final_free_flight_only')
    opening=next(o for o in arch['openings'] if o['id']==row['opening_ids'][0])
    return leg_for_opening(arch,cfg,opening)


def cruise(world,path,planner,*,speed=.45,lookahead=.55,stage='CRUISE',stop_when=None,
           reference=None,exit_yaw=None):
    """Follow a collision-checked polyline without settling at its vertices.

    Sensing runs in World.step; stop_when may compare costs while moving. It
    interrupts only for rerouting, never inserts a hover or resets velocity.
    Precision contact/release/verification use their existing controllers.
    A supplied reference carries acceleration limits across path boundaries.
    With exit_yaw, the endpoint is a fly-through marker and speed is sustained
    through the marker for the next leg.
    """
    from navigation.physics import PhysicsFailure, clip_norm
    points=np.asarray(path,dtype=float)[:,:2]
    if len(points)<2:return True
    world.stage=stage;index=0;start=float(world.data.time)
    budget=20.+4*float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())/max(speed,.1)
    while world.data.time-start<budget:
        p=world.observed()[0][:2]
        if np.linalg.norm(p-points[-1])<.08:return True
        # Progress along the actual segment; do not stop to confirm a vertex.
        a,b=points[index:index+2];delta=b-a;length=float(np.linalg.norm(delta))
        fraction=float(np.clip((p-a)@delta/max(length*length,1e-12),0.,1.))
        if index<len(points)-2 and (fraction>=.995 or np.linalg.norm(p-b)<.10):
            index+=1;continue
        projected=a+fraction*delta;remaining=lookahead;targets=[];cursor=projected
        for end in points[index+1:]:
            d=float(np.linalg.norm(end-cursor))
            if d>remaining:
                targets.append(cursor+(end-cursor)*(remaining/d));break
            targets.append(end);remaining-=d;cursor=end
            if remaining<=1e-8:break
        # Never cut an inflated wall corner just to maintain cruise speed.
        target=next((q for q in reversed(targets) if planner.clear(p,q)),None)
        if target is None:
            target=next((projected+t*(b-projected) for t in (.5,.25,0.) if planner.clear(p,projected+t*(b-projected))),None)
        if target is None:raise PhysicsFailure('cruise lost current-geometry corridor clearance')
        world.target=np.r_[target,1.25]
        world.stage=stage
        if reference is None:
            world.step(yaw=0.,speed=speed)
        else:
            offset=target-p
            if exit_yaw is None:
                velocity=clip_norm(1.25*offset,speed)
            else:
                # This waypoint marks clearance of the frame. Fly past it
                # with a forward reference for the following corridor leg.
                velocity=speed*offset/max(float(np.linalg.norm(offset)),1e-9)
            force=reference.command(world,velocity,0. if exit_yaw is None else exit_yaw,stage)
            world.step(horizontal=force,yaw=reference.heading)
        if world.step_count%5==0 and stop_when is not None and stop_when():return False
    raise PhysicsFailure('cruise path timeout')

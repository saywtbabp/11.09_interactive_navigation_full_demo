"""Compare continuation in contact with a planned, safe withdrawal branch.

Both branches start at the current measured state. No hover-ready prerequisite
is imposed just to calculate a boundary. Execution safety gates remain separate.
"""
import time
import numpy as np
from navigation.fast_cost import CostField,new_cache
from navigation.interactions import panel_geometry
from local_hover import hover_planner,contact_release_clear,retreat_directions
from probe_recovery import monotone_exit

def release_endpoint(field,planner,start,directions,local=None):
    start=np.asarray(start,float)
    if not planner.collides(start)[0] and (local is None or not local.collides(start)[0]):
        try:return start.copy(),field.cost(start)
        except ValueError:pass
    candidates=np.array([start+d*v for d in np.arange(.04,1.25,.04) for v in directions])
    mask=~planner.collides(candidates)
    probes=np.array([[0,0],[.03,0],[-.03,0],[0,.03],[0,-.03]])
    mask &= ~planner.collides((candidates[:,None,:]+probes).reshape(-1,2)).reshape(len(candidates),-1).any(axis=1)
    if local is not None:mask &= ~local.collides(candidates)
    mask &= field.possibly_reachable(candidates)
    for end in candidates[mask]:
        try:cost=field.cost(end)
        except ValueError:continue
        clear=(contact_release_clear(local,start,end,directions[0]) if local is not None else monotone_exit(planner,start,end))
        if not clear:continue
        return end,cost
    raise ValueError('no geometrically feasible release connector for route comparison')

def online_metrics(e):
    started=time.perf_counter();w=e.world;q,v=e.state();p=e.position().copy()
    if not hasattr(e,'online_cost_cache'):e.online_cost_cache=new_cache()
    bypass=e.planner(q,'bypass');future_q=max(e.target,q);future=e.planner(future_q,'target')
    bf=CostField(bypass,e.goal,e.online_cost_cache);cf=CostField(future,e.goal,e.online_cost_cache)
    local=hover_planner(e,q)
    end,Lb=release_endpoint(bf,bypass,p,retreat_directions(e,p,q),local)
    withdrawal=float(np.linalg.norm(end-p));Lb+=withdrawal
    if e.kind=='door':
        pivot=panel_geometry(e.opening,q)[0];t=np.array([np.cos(q),np.sin(q)]);normal=np.array([-np.sin(q),np.cos(q)])
        point=pivot+e.arm*t-.035*normal
        offset=float((p-point)@t)
        contact=point+offset*t-.26*normal
        if e.last_force>.08:contact=p.copy()
        radius=float(np.linalg.norm(contact-pivot));tracking=radius*max(0.,future_q-q)
        angle=future_q-q;c,s=np.cos(angle),np.sin(angle)
        follow=pivot+np.array([[c,-s],[s,c]])@(contact-pivot)
        final_normal=np.array([-np.sin(future_q),np.cos(future_q)])
    else:
        normal=final_normal=np.array([0.,1.]);contact=p.copy()
        gap=e.y+q-.25-p[1]
        if e.last_force<=.08:contact[1]+=max(0.,gap-.26)
        tracking=max(0.,future_q-q);follow=contact+[0.,tracking]
    rejoin=float(np.linalg.norm(contact-p))
    directions=[-final_normal,np.array([0.,-1.])]
    for deg in (15,-15,30,-30,45,-45):
        a=np.radians(deg);directions.append(np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@(-final_normal))
    after,Lc=release_endpoint(cf,future,follow,directions)
    release=float(np.linalg.norm(after-follow));manoeuvre=rejoin+tracking+release
    gain=Lb-Lc-manoeuvre;stroke=max(0.,e.target-q)*e.arm
    economic=gain/(e.args.rate*stroke) if stroke>1e-8 else None
    limit=min(e.cap,max(0.,economic)) if economic is not None else e.cap
    if gain<=0:limit=0.
    return dict(capacity_limit=e.cap,economic_limit=economic,push_limit=limit,gain=gain,stroke=stroke,
        manoeuvre_allowance_m=manoeuvre,expected_delta=0.,alpha=e.args.alpha,bypass_length_m=Lb,candidate_length_m=Lc,
        deferred_exit_m=withdrawal,deferred_exit_path=[p.tolist(),end.tolist()],exit_execution='after_online_avoid',
        continuation_tracking_m=tracking,continuation_release_m=release,recontact_m=rejoin,
        continuation_release_path=[follow.tolist(),after.tolist()],continuation_prediction_q=future_q,
        geometry_q=q,geometry_velocity=v,geometry_xy=p.tolist(),geometry_time=float(w.data.time),
        planning_wall_s=time.perf_counter()-started,cost_field_rebuilds=int(bf.rebuilt)+int(cf.rebuilt),
        cost_source='fresh_native_corner_safe_grid',cost_model='continue_contact_vs_withdraw',
        kinetic_energy_credit=False,grid_resolution_m=e.cfg.grid_resolution,local_goal_xy=e.goal.tolist(),
        planning_scope=e.local_leg['scope'])

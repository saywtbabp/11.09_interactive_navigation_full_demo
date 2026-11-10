"""Geometric completion ends inference; it is independent of pushing utility."""
import copy
import numpy as np
from navigation.fast_cost import CostField,new_cache
from contact_boundary import release_endpoint
from local_hover import hover_planner,retreat_directions


def passage_geometry(e):
    q,velocity=e.state();p=e.position().copy()
    planner=e.planner(q,'target')
    # Cover measured object motion over the crossing lookahead, plus the same
    # public geometric margins used elsewhere. No hidden mass/friction input.
    if e.kind=='door':
        low=max(0.,min(q,q+velocity*.25)-np.radians(e.cfg.DOOR_ANGLE_MARGIN_DEG))
        high=min(e.cfg.DOOR_ANGLE_MAX,max(q,q+velocity*.25)+np.radians(e.cfg.DOOR_ANGLE_MARGIN_DEG))
        angles=np.linspace(low,high,max(2,int(np.ceil(abs(high-low)/np.radians(1.)))+1))
        spacing=abs(high-low)/max(1,len(angles)-1)
        pad=2*e.opening['width_m']*np.sin(spacing/4)
        for angle in angles:
            c,r,h=e.panel(angle);planner.boxes.append((c,r,h+pad+e.envelope.half[:2]@np.abs(r)))
    else:
        future=q+velocity*.25;c,r,h=e.panel((q+future)/2)
        planner.boxes.append((c,r,h+np.array([0.,abs(future-q)/2+e.cfg.DOOR_POSITION_MARGIN])+e.envelope.half[:2]))
    if not hasattr(e,'passage_cost_cache'):e.passage_cost_cache=new_cache()
    field=CostField(planner,e.goal,e.passage_cost_cache)
    local=hover_planner(e,q)
    return q,velocity,p,planner,local,field


def certificate_for(e,geometry,end,route):
    q,velocity,p,planner,local,field=geometry
    return dict(time=float(e.world.data.time),q=q,object_velocity=velocity,
        start_xy=p.tolist(),release_endpoint=end.tolist(),release_path=[p.tolist(),end.tolist()],
        passage_path=route.tolist(),local_goal_xy=e.goal.tolist(),
        target_only=True,motion_margin_checked=True,geometry_basis='current_measured_object_and_full_envelope')


def current_passage(e,geometry=None):
    geometry=passage_geometry(e) if geometry is None else geometry
    q,velocity,p,planner,local,field=geometry
    end,_=release_endpoint(field,planner,p,retreat_directions(e,p,q),local)
    route,_=planner.plan(end,e.goal)
    return certificate_for(e,geometry,end,route)


def try_freeze(e):
    if e.inference_frozen:return True
    if not e.committed or e.online_exit_latched or e.online_choice!='continue':return False
    q,_=e.state()
    # Cheap rejection only; the actual decision is the full route certificate.
    if q<.5*e.target:return False
    try:certificate=current_passage(e)
    except ValueError:return False
    basis=e.snapshots[-1]
    if basis.get('control_choice')!='continue' or not basis.get('decision_push'):return False
    now=float(e.world.data.time)
    frozen=copy.deepcopy(basis)
    frozen.update(label='passage_frozen',time=now,formal_decision=False,weights_changed=False,
        route_preview_status='frozen_after_geometric_completion',boundary_age_s=now-basis['geometry_time'],
        inference_frozen=True,frozen_at=now,frozen_reference_time=basis['time'],
        clearance_geometry_time=now,control_choice='complete',exit_latched=False)
    e.snapshots.append(frozen);e.inference_frozen=True;e.inference_end_time=basis['evidence_time']
    e.freeze_record=dict(time=now,reference_time=basis['time'],evidence_end_time=e.inference_end_time,
        windows=len(e.belief.windows),weight_revision=e.online_revision,p_ok=basis['push_probability'],
        F_star_N=basis['push_limit'],certificate=certificate)
    e.passage_certificate=certificate;e.online_motion_finished=True;e.online_choice='complete'
    e.online_reason='verified_passage_no_further_inference'
    e.event('passage_inference_frozen',freeze_time=now,**{k:v for k,v in e.freeze_record.items() if k!='time'})
    m=e.mission;m.status='通行几何已满足 / 冻结推断，准备穿越';m.action='Cross'
    m.reason='保留完成前最后一次真实权重和边界；传感器、脱离接触及执行净空检查继续'
    m.decisions.append(dict(time_s=now,opening=e.oid,phase='completion',semantic=e.kind,action='Cross',
        p_ok=basis['push_probability'],F_star_N=basis['push_limit'],weight_revision=e.online_revision,
        frozen_reference=True,reference_time=basis['time'],local_leg=e.local_leg,
        reason='actual_target_only_passage_available_inference_frozen'))
    return True

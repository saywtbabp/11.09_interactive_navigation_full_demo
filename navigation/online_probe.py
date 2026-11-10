"""Mission-owned action and read-only inference lifecycle; single Tk root."""
import numpy as np
from .guarded_action import execute, GUARDED
from .door_probe import record_probe
from .interactions import panel_geometry, contact_pose
from .geometry import Planner
from .physics import PhysicsFailure
from live_belief import PrefixBeliefMonitor


def prepare_current(m, w, opening, inst, action):
    q,_=inst.state();t=np.array([np.cos(q),np.sin(q)]);n=np.array([-np.sin(q),np.cos(q)])
    pivot=panel_geometry(opening,q)[0]
    panels=[panel_geometry(opening,q)[2]]
    panels += [panel_geometry(r.geometry,r.angle)[2] for k,r in m.memory.records.items()
               if k!=opening['id'] and r.semantic=='door' and r.angle_time>=0]
    sealed=[k for k,r in m.memory.records.items() if r.traversability=='blocked' or r.bypass_selected]
    planner=Planner(m.arch,m.cfg,panels=panels,sealed=sealed)
    w.stage='DESIGN_PREPARE'
    for distance in (.65,.85,1.05):
        pre=np.r_[pivot+action.arm*t-distance*n,1.25]
        try:path,_=planner.plan(w.observed()[0],pre)
        except ValueError:continue
        m.path=path
        m.memory.event(w.data.time,opening['id'],'guarded_preparation',angle_rad=q,
                       target=pre.tolist(),path=path.tolist(),source='current geometry A*; no reset')
        w.fly(path[1:],yaw=q,speed=.2,stage='DESIGN_PREPARE')
        w.hold(pre,.3,yaw=q,stage='DESIGN_PREPARE')
        return
    raise PhysicsFailure('no safe current-state guarded preparation route')


def tick_monitor(m):
    monitor=m.active_monitor
    if monitor is None:return
    changed=monitor.tick()
    if changed:
        m.weights=monitor.weights.copy();m.labels=monitor.labels.copy()
        row=monitor.updates[-1]
        m.metrics.update(p_D=row['p_push'],p_push=row['p_push'],D=row['D'])
        m.status='Probe前固定标签下的实时证据 · '+monitor.phase
        m.reason='固定初始先验 + 首次接触至当前完整前缀；只显示，不中途改变动作'
        m.weight_history.append(dict(opening=m.belief_owner,mission_time_s=m.world.data.time,
                                    **row,weights=monitor.weights.copy(),labels_source='frozen pre-Probe task labels'))
    # Save actual online views once per measured stage and at a later prefix.
    if monitor.torques:
        key=m.belief_owner+'_live_'+monitor.phase
        if key not in m.saved_keys:m.save_key(key)
        if monitor.elapsed>=.9:m.save_key(m.belief_owner+'_live_prefix_090')


def run_probe(m,rec,instrument,pre,folder):
    w=m.world;oid=rec.stable_id
    monitor=PrefixBeliefMonitor(w,instrument,m.theta,m.weights,pre.labels,m.cfg,m.cfg.live_belief_period)
    monitor.opening_id=oid
    m.probe_monitors[oid]=monitor;m.active_monitor=m.display_monitor=monitor
    # Utility boundary is scene-dependent; hard feasibility remains a separate cap.
    utility=(pre.G_norm*m.cfg.b_ref/(m.cfg.DECISION_LAMBDA*pre.remaining_angle)
             if m.cfg.DECISION_LAMBDA*pre.remaining_angle>0 else float('inf'))
    m.live_boundary=min(m.cfg.tau_push_max,utility)
    previous=w.after_step_observer
    def observe(world):
        if previous:previous(world)
        monitor.after_step(world)
    w.after_step_observer=observe
    try:
        if m.cfg.probe_action=='guarded':
            result=execute(w,rec.geometry,instrument,GUARDED,
                           lambda *args:prepare_current(m,*args))
            contacts=result['contacts'];legacy_z=None
            folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'action_measurements.npz',force=result['force'],
                                raw_requested_commands=result['commands'],
                                requested_commands=np.array([c*min(1.,m.cfg.probe_force_max/max(np.linalg.norm(c),1e-12))
                                                             for c in result['commands']]))
            for a,b in zip(result['trace'].validate(),monitor.trace().validate()):
                np.testing.assert_array_equal(a,b)
        else:
            w.fly([contact_pose(rec.geometry)],yaw=0.,speed=.35)
            for _ in range((25-w.step_count%25)%25):w.step(yaw=0.)
            packet,_=m.sensors.acquire(force=True);m.memory.fuse(packet)
            _,contacts,legacy_z=record_probe(w,rec.geometry,instrument)
        monitor.finish()
        return monitor.trace(),contacts,legacy_z
    except BaseException as exc:
        monitor.error=str(exc);monitor.tick(force=True)
        raise
    finally:
        w.after_step_observer=previous
        m.active_monitor=None
        monitor.save(folder/'live')
        from live_particle_panel import render_panel
        render_panel(monitor,oid+' / '+m.cfg.probe_action,m.live_boundary).save(folder/'live'/'panel_final.png')

"""Small integration layer for dynamics Door only; other Mission paths unchanged."""
from dataclasses import asdict, replace
import numpy as np
from .door_probe import DoorInstrument, record_probe
from .door_dynamics import update_belief, belief_summary
from .door_decision import from_geometry
from .geometry import door_comparison
from .contact_geometry import contact_comparison
from .interactions import contact_pose, merge_pose, recover


def print_decision(result, summary, cfg, *, post=False):
    print('\n========== '+('POST-PROBE DECISION' if post else 'DOOR DECISION')+' ==========', flush=True)
    print(f'Current theta: {result.theta_current:.6f}\nTheta clear: {result.theta_clear:.6f}'
          f'\nremaining angle: {result.remaining_angle:.6f}'
          f'\nL short: {result.L_short:.6f}\nL avoid: {result.L_avoid:.6f}'
          f'\nG: {result.G:.6f}\nG normalized: {result.G_norm:.6f}'
          f'\nB ref: {cfg.b_ref:.6f}\ntau push max: {cfg.tau_push_max:.6f}'
          f'\nmean remaining B: {result.mean_B:.6f}'
          f'\nbelief: mean I: {summary["mean_I"]:.6f}; mean tau_f: {summary["mean_tau_f"]:.6f}; ESS: {summary["ESS"]:.2f}'
          f'\np_push: {result.p_push:.6f}\nD: {result.D:.6f}'
          f'\nmean DeltaJ: {result.mu_deltaJ:.6f}\nstd DeltaJ: {result.sigma_deltaJ:.6f}'
          f'\ndecision: {result.action.upper()}\nreason: {result.reason}', flush=True)


def publish(m, oid, result, geometry, *, post=False):
    m.labels=result.labels.copy()
    m.metrics=dict(G=result.G, p_D=result.p_push, D=result.D, p_push=result.p_push,
                   mu_deltaJ=result.mu_deltaJ, sigma_deltaJ=result.sigma_deltaJ,theta_clear=result.theta_clear)
    m.action, m.reason=result.action, result.reason
    if post:m.status='Probe后正式决策 / '+result.action
    m.door_beliefs[oid]=dict(theta=m.theta.copy(), weights=m.weights.copy(), labels=m.labels.copy(),
                            **m.metrics, probe_used=post, source='door dynamics + analytic effort labels')
    m.decisions.append(dict(time_s=m.world.data.time, opening=oid, **m.metrics,
                           action=result.action, reason=result.reason, N=len(m.theta), geometry=geometry,
                           labels_source='G_norm - lambda * tau_f * remaining_angle / B_ref; hard torque feasibility',
                           probe_used=post, remaining_angle=result.remaining_angle,
                           L_short=result.L_short, L_avoid=result.L_avoid, G_norm=result.G_norm))


def pass_open_door(m, rec, panel):
    # Uses the unchanged physical recovery and ordinary navigation, with no Push.
    w=m.world
    m.action='Pass';m.status='门已打开，净空通行'
    if getattr(m,'clearance_solver',None):
        from .door_crossing import cross_door
        return cross_door(m,rec,panel,measurement='lidar panel fit' if getattr(w,'onboard',None) is not None else 'ideal hinge encoder')
    path=recover(w,rec.geometry,panel)
    rec.traversability='free';rec.verification_state='verified_free';rec.confirmation='confirmed'
    m.memory.event(w.data.time,rec.stable_id,'door_geometry_clear',estimated_angle_rad=panel()[0])
    m.path=path;w.fly(path[1:],yaw=0.);m.save_key(rec.stable_id+'_crossed')


def run_door(m, rec):
    w,o,oid,cfg=m.world,rec.geometry,rec.stable_id,m.cfg
    if getattr(w,'onboard',None) is not None:
        from .door_lidar import SensedDoorInstrument
        instrument=SensedDoorInstrument(w.onboard,w.onboard.tracks[oid])
    else:instrument=DoorInstrument(w,o)
    # Explicit experiment measurement interface, exposing only angle and speed.
    panel=instrument.state
    m.belief_owner=oid;m.status='门动力学决策';m.action='Evaluate'
    # Each new door starts uniformly. A previously probed door keeps its posterior
    # and can never trigger a second Probe on a later visit.
    old=m.door_beliefs.get(oid)
    m.weights=(np.asarray(old['weights']).copy() if old else np.ones(len(m.theta))/len(m.theta))
    used=bool(rec.probe_used)
    geo=door_comparison(m.arch,cfg,o,w.observed()[0],merge_pose(o))
    solver=getattr(m,'clearance_solver',None)
    if solver:
        m.status='计算门洞净空';m.reason='由机体碰撞尺寸、姿态/距离余量与穿门路径求解'
        geo=solver.annotate(o,geo)
        m.memory.event(w.data.time,oid,'computed_door_clearance',**geo['clearance'])
    pre=from_geometry(m.theta,m.weights,panel()[0],geo,cfg,post_probe=used)
    # Explicit CLI baselines only; default adaptive always uses the new scheme.
    if pre.action!='Pass' and (not solver or geo['clearance']['feasible']):
        if m.baseline=='always_avoid':pre=replace(pre,action='Avoid',reason='explicit always_avoid baseline')
        elif m.baseline=='always_push':pre=replace(pre,action='Push',reason='explicit always_push baseline')
        elif m.baseline=='no_probe' and pre.action=='Probe':
            pre=replace(pre,action='Push' if pre.p_push>.5 else 'Avoid',reason='explicit no_probe baseline')
    before=m.weights.copy();summary_before=belief_summary(m.theta,before)
    print_decision(pre,summary_before,cfg)
    publish(m,oid,pre,geo,post=used);m.save_key(oid+'_decision')
    result=pre;trace=None;predictions=None;contacts=[];postgeo=geo
    folder=m.folder/('door_'+oid+'_'+str(len(m.decisions)))
    record=dict(door_method='dynamics',particles=m.theta,weights_before=before,weights_after=before.copy(),
                theta_before=pre.theta_current,theta_after_probe=None,
                L_short=pre.L_short,L_avoid=pre.L_avoid,G_norm=pre.G_norm,
                p_push_before=pre.p_push,D_before=pre.D,mu_deltaJ_before=pre.mu_deltaJ,
                sigma_deltaJ_before=pre.sigma_deltaJ,decision_before=pre.action,
                pre=asdict(pre),probe_time=[],probe_torque=[],theta_real=[],omega_real=[],
                belief_before=summary_before,probe_used_before=used,
                instrument='ideal hinge encoder + real contact wrench; no true inertia/friction in policy')
    if getattr(w,'onboard',None) is not None:
        record['instrument']=instrument.source
        record['measurement_mode']='sensed'
        record['legacy_array_keys_note']='theta_real/omega_real retained as compatibility aliases of estimated angles/speeds; not hinge truth'
    if solver:record['clearance_before']=geo['clearance']
    from experiments.door_analysis import write_experiment, diagnostic_plots
    try:
        if pre.action=='Probe':
            if used:raise RuntimeError('second Probe forbidden')
            rec.probe_used=True;m.probes+=1;m.status='标准主动探测 / 真实门轴力矩'
            from .online_probe import run_probe
            record['probe_action']=cfg.probe_action
            trace,contacts,legacy_z=run_probe(m,rec,instrument,pre,folder)
            # Recompute from the immutable pre-Probe prior, never from live posterior.
            m.weights,error,predictions,_=update_belief(before,m.theta,trace,cfg)
            np.testing.assert_allclose(m.weights,m.probe_monitors[oid].weights,rtol=1e-12,atol=1e-14)
            record.update(weights_after=m.weights.copy(),theta_after_probe=float(trace.theta[-1]),
                          probe_time=trace.time.tolist(),probe_torque=trace.torque.tolist(),
                          theta_real=trace.theta.tolist(),omega_real=trace.theta_dot.tolist(),
                          contacts=contacts,trajectory_error=error,
                          probe_theta_initial=float(trace.theta[0]),probe_omega_initial=float(trace.theta_dot[0]),
                          legacy_peak_angle_diagnostic_only=legacy_z)
            summary_after=belief_summary(m.theta,m.weights)
            print('\n========== PROBE RESULT ==========',flush=True)
            print(f'probe duration: {trace.time[-1]:.6f}\nmax contact torque: {np.max(np.abs(trace.torque)):.6f}'
                  f'\ndoor angle before: {trace.theta[0]:.6f}\ndoor angle after: {trace.theta[-1]:.6f}'
                  f'\nbelief before: {summary_before}\nbelief after: {summary_after}',flush=True)
            print('top candidates:',summary_after['top_candidates'],flush=True)
            # Replan from the actual post-contact position, reusing existing A* connector.
            postgeo=contact_comparison(m.arch,cfg,o,w.observed()[0],merge_pose(o))
            if solver:
                postgeo=solver.annotate(o,postgeo)
                record['clearance_after']=postgeo['clearance']
            result=from_geometry(m.theta,m.weights,panel()[0],postgeo,cfg,post_probe=True)
            print_decision(result,summary_after,cfg,post=True)
            m.weight_history.append(dict(opening=oid,time_s=w.data.time,weights=m.weights.copy(),
                                         source='one full trajectory likelihood from prior; no repeated prefix updates'))
            record['live_evidence_semantics']='Probe前固定标签下的实时证据'
            record['post_decision_semantics']='Probe后真实状态重新规划后的正式决策'
            publish(m,oid,result,postgeo,post=True);m.save_key(oid+'_post_probe')
        summary_after=belief_summary(m.theta,m.weights)
        record.update(weights_after=m.weights.copy(),post=asdict(result),post_geometry=postgeo,
                      estimated_I=summary_after['mean_I'],estimated_tau_f=summary_after['mean_tau_f'],
                      belief_after=summary_after,p_push_after=result.p_push,D_after=result.D,
                      mu_deltaJ_after=result.mu_deltaJ,sigma_deltaJ_after=result.sigma_deltaJ,
                      final_decision=result.action)
        write_experiment(folder,record,w,oid)
        # Plots are diagnostics: filesystem/plot failure cannot change the policy.
        try:diagnostic_plots(folder,m.theta,before,m.weights,pre,result,cfg,trace,predictions)
        except Exception as exc:print('Diagnostic plot warning:',exc,flush=True)
        if result.action=='Pass':pass_open_door(m,rec,panel)
        elif result.action=='Avoid':
            if rec.probe_used:
                m.status='脱离与恢复';path=recover(w,o,lambda:m.state_panel(rec),bypass=True);w.fly(path[1:],yaw=0.)
            rec.bypass_selected=True
            m.memory.event(w.data.time,oid,'avoid_selected',probe_used=rec.probe_used)
        else:m.execute_push(rec)  # unchanged rotor Push, lidar state, recovery and crossing
        record['execution_outcome']='bypass_selected' if rec.bypass_selected else 'crossed'
    except Exception as exc:
        record['execution_error']=str(exc)
        monitor=m.probe_monitors.get(oid)
        if monitor and monitor.torques:
            partial=monitor.trace()
            record.update(probe_time=partial.time.tolist(),probe_torque=partial.torque.tolist(),
                          theta_real=partial.theta.tolist(),omega_real=partial.theta_dot.tolist())
        raise
    finally:
        # Preserve the experiment even on an execution/control failure.
        record.setdefault('post',asdict(result))
        record.setdefault('final_decision',result.action)
        if getattr(w,'onboard',None) is not None:
            record['theta_estimated']=record['theta_real'];record['omega_estimated']=record['omega_real']
            record['door_geometry_measurement']=o.get('sensed_geometry')
        write_experiment(folder,record,w,oid)

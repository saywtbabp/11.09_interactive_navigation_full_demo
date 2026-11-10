"""Audit online causality, window pairing, switching and physical flight."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import runtime
import numpy as np
from belief import Belief
from mission import Mission
from configuration import Config
from navigation.perception import Record

def validate_online(folder,require_door_contrast=False):
    folder=Path(folder);d=json.loads((folder/'replay.json').read_text(encoding='utf-8'));checks=[];stats={}
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    s=d['summary'];check('goal reached',s['success']);check('no wall/frame collision',s['wall_contact_peak_N']<=.35)
    check('goal within 10 cm',np.linalg.norm(np.array(s['final_position'])-d['map']['goal_xyz_m'])<.1)
    check('single monotonically timed world',np.all(np.diff([f['time'] for f in d['frames']])>0))
    check('glass verification preserved',any(v['opening']=='G01' and v['outcome']=='blocked' for v in d['verifications']))
    if s.get('execution_settings',{}).get('verified_exit')=='continuous_reference_through_waypoint':
        exits=[e for e in d['events'] if e['event']=='verified_exit_complete']
        verified=[v for v in d['verifications'] if v['outcome']=='free']
        check('every verified empty aperture completes a smooth exit',
              bool(verified) and [v['opening'] for v in verified]==[e['stable_id'] for e in exits])
        for v,e in zip(verified,exits):
            samples=[r for r in d['telemetry'] if v['end_time_s']<r[0]<=e['time_s']]
            check(v['opening']+' continuous motion through the verified exit',bool(samples)
                  and all(r[1]=='CRUISE_VERIFIED_EXIT' for r in samples)
                  and e['exit_speed_mps']>.10 and e['velocity_slew_mps2']<=.350001)
            check(v['opening']+' no exit contact or forward reversal',
                  all(r[15]==0. and np.dot(r[5:7],v['normal_xy'])>0. for r in samples))
    for oid,r in d['interactions'].items():
        if not r['probed']:continue
        pairs=r['online_pairs'];updates=[x for x in pairs if x['weights_changed']]
        check(oid+' every weight revision has exactly one boundary', [x['revision'] for x in updates]==list(range(1,len(updates)+1)))
        check(oid+' native boundary count matches pairs',r['online_boundary_calls']==len(pairs))
        for p in pairs:
            check(oid+' weight and geometry clocks paired',abs(p['time']-p['geometry_time'])<1e-9 and abs(p['time']-p['evidence_time'])<1e-9)
            check(oid+' current boundary available',p['error'] is None and p['F_star_N'] is not None)
        snapshots=[x for x in r['snapshots'] if x.get('route_preview_status')=='paired_online_geometry']
        for a in snapshots:
            check(oid+' zero-age online geometry',a['formal_decision'] and a['boundary_age_s']==0.)
            if a['push_probability'] is not None and not a['exit_latched'] and a['control_choice']!='complete':
                expect='continue' if a['decision_push'] else 'avoid'
                check(oid+' online action follows current majority',a['control_choice']==expect)
            poly=np.array(a['polygon'])
            if len(poly)>=3:
                for dv,T,Y,ev,ey in a['constraints']:
                    check(oid+' all current halfplanes retained',np.all(poly[:,0]*(dv-ev)+poly[:,1]*T<=Y+ey+1e-7) and np.all(poly[:,0]*(dv+ev)+poly[:,1]*T>=Y-ey-1e-7))
        raw=np.array([x[:5] for x in r['trace']],float);arm=.835 if r['object_kind']=='door' else 1.
        if r.get('inference_frozen'):raw=raw[raw[:,0]<=r['inference_end_time']+1e-8]
        raw[:,1:3]*=arm;raw[:,3]/=arm;offline=Belief();offline.update(raw)
        final=snapshots[-1]
        check(oid+' no duplicate or missing window evidence',len(offline.windows)==len(updates)==final['windows'])
        check(oid+' incremental likelihood equals independent batch reconstruction',np.allclose(offline.precision,final['precision']) and np.allclose(offline.linear,final['linear']))
        end=next(x for x in r['events'] if x['event']=='online_contact_end')
        first=next(x for x in pairs if x['choice'] in ('continue','avoid'))
        check(oid+' choice occurs during contact',first['time']<end['time'] and first['time']-end['first_contact_time']<1.4)
        release=next(x for x in r['events'] if x['event']=='online_release')
        avoid=next((x for x in pairs if x['choice']=='avoid'),None)
        if avoid is not None:check(oid+' avoid command unloads without waiting for probe completion',abs(release['time']-avoid['time'])<=.0021)
        check(oid+' no recovery before contact-mode choice',all(x['time']>=release['time']-1e-8 for x in r['recovery_frames']))
        ready=next(x for x in r['events'] if x['event']=='probe_recovery_ready')
        check(oid+' safety conditions still required',all(ready[k] for k in ('detached','stable','clearance','envelope_ok','exit_path')))
        check(oid+' bypass flight after safety ready',all(x['time']>=ready['time'] for x in r['events'] if x['event']=='bypass_exit_start'))
        check(oid+' no recontact after committed exit',not any(x['from_choice']=='avoid' and x['to_choice']=='continue' for x in r['online_switches']))
        check(oid+' interaction contact guard respected',r['contact_peak_N']<=8.)
        if s.get('execution_settings',{}).get('contact_control')=='debounced_coast_and_reapproach':
            motion=r.get('motion_frames',[])
            losses=[e for e in r['events'] if e['event']=='contact_lost']
            for loss in losses:
                check(oid+' contact loss confirmed with force hysteresis and dwell',loss['contact_N']<.03 and .04-1e-8<=loss['confirmation_s']<=.0421)
                following=next((f for f in motion if f['time']>=loss['time']-1e-8),None)
                check(oid+' loss promptly enters dedicated coast control',following is not None and following['time']-loss['time']<.055 and following['motion_mode']=='COAST_BRAKE')
                check(oid+' coast reference starts at measured velocity',np.allclose(following['initial_reference_velocity_xy'],loss['velocity_xy']))
            for sample in motion:
                check(oid+' continuous velocity reference respects slew limit',sample['max_velocity_slew']<=.350001)
                check(oid+' force reference respects jerk limit',sample['max_force_jerk']<=12.000001)
                check(oid+' yaw reference respects angular acceleration limit',sample['max_yaw_accel']<=1.200001)
            if losses:
                check(oid+' valid coasting data still updates parameters',any(p['weights_changed'] and p['time']>losses[0]['time']+.1 for p in pairs))
            if r.get('inference_frozen') and motion:
                t=r['freeze_record']['time'];before=[x for x in motion if x['time']<t];after=[x for x in motion if x['time']>=t]
                if before and after:check(oid+' geometric completion reuses continuous motion reference',before[-1]['reference_started_time']==after[0]['reference_started_time'])
        if r.get('inference_frozen'):
            freeze=r['freeze_record'];frozen=next(x for x in r['snapshots'] if x.get('inference_frozen'))
            basis=next(x for x in r['snapshots'] if abs(x['time']-freeze['reference_time'])<1e-8)
            check(oid+' freezes last real pushing result',basis['decision_push'] and frozen['push_probability']==basis['push_probability'] and frozen['push_limit']==basis['push_limit'])
            check(oid+' exact frozen density and constraints',frozen['precision']==basis['precision'] and frozen['linear']==basis['linear'] and frozen['constraints']==basis['constraints'] and frozen['polygon']==basis['polygon'])
            check(oid+' no parameter or boundary updates after geometric completion',all(x['time']<freeze['time'] for x in pairs))
            check(oid+' raw sensors continue after inference stops',r['sensor_frames'][-1]['time']>freeze['time']+.1)
            check(oid+' completion uses measured current object and margins',freeze['certificate']['target_only'] and freeze['certificate']['motion_margin_checked'])
            check(oid+' completion unloads immediately',abs(release['time']-freeze['time'])<=.0021)
            if s.get('execution_settings',{}).get('completion_recovery')=='direct_passage_connection':
                recovery=r['recovery_frames'];selected=[e for e in r['events'] if e['event']=='passage_connection_selected']
                check(oid+' unload uses direct passage connection',r['probe_recovery']=='direct_passage_connection' and bool(selected))
                check(oid+' direct handoff does not use a hover anchor',all(x.get('hover_anchor_used') is False and x.get('local_hover') is False for x in recovery))
                check(oid+' every connection is checked with swept body and object motion margin',all(x.get('connection_swept_clear') and x.get('connection_motion_margin_checked') for x in recovery))
                for sample in recovery:
                    selection=next(e for e in reversed(selected) if e['time']<=sample['time']+1e-8)
                    check(oid+' target stays fixed until explicit geometry-triggered selection',sample['connection_target_xy']==selection['target_xy'])
                direction=np.array(r['local_leg']['goal_xy'])-np.array(recovery[0]['position_xy']);direction/=np.linalg.norm(direction)
                for previous,selection in zip(selected,selected[1:]):
                    if selection['reason']=='newly_clear_forward_connection':
                        check(oid+' optional connection upgrade strictly progresses forward',float((np.array(selection['target_xy'])-previous['target_xy'])@direction)>.10)
                check(oid+' no hover recovery fallback or separate crossing connector',not any(e['event'] in ('probe_recovery_fallback','certified_crossing_connector') for e in r['events']) and 'CROSSING_CONNECTOR' not in r['paths'])
                check(oid+' handoff directly from actual current pose',np.linalg.norm(np.diff(np.array(r['final_passage_certificate']['release_path']),axis=0))<1e-6)
                check(oid+' stable gates and continuous confirmation preserved',ready['dwell_s']>=.25 and all(ready[k] for k in ('speed_ok','gyro_ok','tilt_ok','yaw_ok','height_ok')))
                last=recovery[-1]
                check(oid+' measured ready speed and gyro meet original limits',last['speed_mps']<.065 and last['gyro_radps']<.16 and last['contact_N']<.1)
        if r['outcome']=='shortcut':
            cleared=next(x for x in r['events'] if x['event']=='clearance_verified')
            check(oid+' crossing after all safety conditions',cleared['time']>=ready['time'])
            check(oid+' actual opening or current-pose passage certified',cleared['observed_q']>=r['target_q'] or (cleared.get('current_pose_passage_certified') and r.get('final_passage_certificate') is not None))
            check(oid+' actual shortcut path recorded','SHORTCUT' in r['paths'])
            if r.get('inference_frozen'):
                certified=r['final_passage_certificate']['passage_path'];executed=r['paths']['SHORTCUT']
                check(oid+' executes the final margin-certified passage',np.linalg.norm(np.array(certified[0])-executed[0])<1e-8 and np.linalg.norm(np.array(certified[1])-executed[1])<1e-8)
            check(oid+' no alternative-aperture exit executed',not any(x['event']=='bypass_exit_start' for x in r['events']))
            release_clear=next((x for x in r['events'] if x['event']=='clearance_achieved_during_release'),None)
            if release_clear is not None:
                check(oid+' reuse opening without recontact only after no-more-work stop',release_clear['prior_stop_reason']=='no_remaining_navigation_gain' and release_clear['time']>=ready['time'])
                check(oid+' crossing separately recorded as geometry decision',any(x['opening']==oid and x['phase']=='clearance' and x['action']=='Cross' for x in d['decisions']))
        values=[x['wall_s'] for x in pairs]
        stats[oid]=dict(updates=len(updates),first_choice_delay_s=first['time']-end['first_contact_time'],
            first_choice=first['choice'],first_probability=first['p_ok'],median_update_ms=1000*float(np.median(values)),
            p95_update_ms=1000*float(np.percentile(values,95)),switches=r['online_switches'])
    gate=next(x for x in d['decisions'] if x['opening']=='S01' and x['phase']=='precontact')
    check('S01 still rejected before approach',gate['action']=='Avoid' and gate['gate_location']=='before_approach')
    local=s.get('execution_settings',{}).get('navigation_scope')=='rolling_local_horizon'
    if local:
        check('local goal used for S01',gate['local_leg']['scope']=='current_wall_to_common_local_merge' and abs(gate['local_leg']['row_y']-5.6)<.01)
        check('no cost-comparison braking stage',not any(row[1]=='COST_GATE_BRAKE' for row in d['telemetry']))
        moving=[x for x in d['events'] if x['event']=='moving_cost_gate' and x['stable_id']=='P01']
        check('P01 compared during actual cruise',bool(moving) and moving[0]['speed_mps']>.20 and not moving[0]['brake_inserted'])
        for row in d['decisions']:
            if row['phase']=='precontact':
                leg=row['local_leg'];check(row['opening']+' both branches stop at common local goal',leg['goal_xy'][1]>leg['row_y'] and leg['goal_xy'][1]<=leg['row_y']+s['execution_settings']['local_merge_distance_m']+.01 and leg['goal_xy']!=d['map']['goal_xyz_m'][:2])
        after_f01=next(x['time'] for x in json.loads((folder/'keyframes.json').read_text(encoding='utf-8')) if x['label']=='F01_verified_free')
        second_row=[f for f in d['frames'] if f['time']>=after_f01 and f['position'][1]<=5.6 and f['position'][1]>=3.28]
        check('F01 to F02 does not lean toward S01',bool(second_row) and max(f['position'][0] for f in second_row)<-3.85)
        centred=[row for row in d['telemetry'] if row[1]=='CRUISE_TO_OBSERVATION' and row[0]>=after_f01 and 3.3<row[3]<4.6]
        check('empty F02 observation approach is centred',bool(centred) and max(abs(row[2]+4.2) for row in centred)<.04)
    if require_door_contrast:
        check('P01 online interaction ends in actual shortcut',d['interactions']['P01']['outcome']=='shortcut')
        check('P01 initial evidence chooses continue',stats['P01']['first_choice']=='continue')
        check('A01 still chooses bypass',d['interactions']['A01']['outcome']=='probe_then_bypass')
        check('A01 last contact choice is avoid',stats['A01']['switches'][-1]['to_choice']=='avoid')
        check('B01 still crosses after interaction',d['interactions']['B01']['outcome']=='shortcut')
        door=next(o for o in d['map']['openings'] if o['id']=='P01');y=door['center_xy_m'][1]
        crossing=[a['position'][0]+(b['position'][0]-a['position'][0])*(y-a['position'][1])/(b['position'][1]-a['position'][1])
            for a,b in zip(d['frames'],d['frames'][1:]) if a['position'][1]<y<=b['position'][1]]
        check('physical flight crosses P01 aperture rather than F03',any(abs(x-door['center_xy_m'][0])<door['width_m']/2-.09 for x in crossing))
    # An early-stopped box must remain in the cognitive geometry at its observed
    # translated position. Do not test against hidden physical builder contents.
    m=Mission(Config(),folder/'map_check')
    try:
        saved=json.loads((folder/'online_map.json').read_text(encoding='utf-8'))
        m.memory.records={k:Record(**v) for k,v in saved['records'].items()}
        for oid,r in m.memory.records.items():
            if r.semantic=='box' and r.probe_used:
                x,y=r.geometry['center_xy_m']
                check('moved '+oid+' retained after stop choice',m.planner().collides([x,y+r.linear_position])[0])
    finally:m.close()
    check('self-contained report exists',(folder/'report.html').is_file())
    out=dict(passed=True,count=len(checks),door_contrast_required=require_door_contrast,summary=s,online_statistics=stats,checks=checks)
    (folder/'online_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=len(checks),statistics=stats),ensure_ascii=False));return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--require-door-contrast',action='store_true')
    a=p.parse_args();validate_online(a.folder,a.require_door_contrast)

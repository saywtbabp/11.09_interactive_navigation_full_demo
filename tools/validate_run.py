"""Audit measured behavior and inference from a completed continuous mission."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import runtime
import numpy as np
from decision_rule import evaluate

def validate(folder,require_boundary=True):
    d=json.loads((folder/'replay.json').read_text(encoding='utf-8'));checks=[]
    if d['summary'].get('navigation_revision')=='paired_window_online_control':
        from tools.validate_online import validate_online
        return validate_online(folder)
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    s=d['summary'];check('mission reached goal',s['success'])
    p=np.array(s['final_position']);goal=np.array(d['map']['goal_xyz_m'])
    check('final position within 10 cm',np.linalg.norm(p-goal)<.1)
    check('wall/jamb guard respected',s['wall_contact_peak_N']<=.35)
    check('time increases, no world reset',np.all(np.diff([f['time'] for f in d['frames']])>0))
    first=d['frames'][0]['map'];check('initial semantic map unknown',all(r['semantic']=='unknown' for r in first.values()))
    check('glass blocked by original verification',any(v['opening']=='G01' and v['outcome']=='blocked' and v['valid_force_samples']>=2 for v in d['verifications']))
    expected={'S01':'heuristic_skip','B01':'shortcut'}
    if require_boundary:expected.update(P01='shortcut',A01='probe_then_bypass')
    for oid,outcome in expected.items():check(oid+' outcome '+outcome,d['interactions'][oid]['outcome']==outcome)
    for row in d['decisions']:
        if row['phase']=='precontact':
            admitted=row['G0_m']>row['approach_extra_m']+row['probe_budget_m']
            check(row['opening']+' precontact gate agrees with costs',(row['action']=='Probe')==admitted and row['p_ok'] is None)
    probabilities={}
    for oid,result in d['interactions'].items():
        snaps=result['snapshots'];check(oid+' independent unbounded initial belief',snaps[0]['windows']==0 and snaps[0]['bounds'] is None)
        formal=next((x for x in snaps if x['label']=='replan'),None)
        if not formal:continue
        p=formal['push_probability'];probabilities[oid]=p
        rule=evaluate(p,formal['integration_converged'],'majority')
        choice=next(e for e in result['events'] if e['event']=='route_choice')
        check(oid+' decision follows measured majority',rule['decision_push']==(choice['route']=='push'))
        ready=next(e for e in result['events'] if e['event']=='probe_recovery_ready')
        check(oid+' formal decision after state ready',choice['time']>=ready['time'])
        check(oid+' recovery states all satisfied',all(ready[k] for k in ('detached','stable','envelope_ok','clearance','exit_path')))
        exits=[e for e in result['events'] if e['event']=='bypass_exit_start']
        check(oid+' bypass connector deferred',all(e['time']>=choice['time'] for e in exits) and (not exits if choice['route']=='push' else True))
        for snap in snaps:
            poly=np.array(snap['polygon']);constraints=snap.get('constraints',[])
            if len(poly)<3:continue
            for dv,T,Y,ev,ey in constraints:
                check(oid+' polygon obeys both window halfplanes',np.all(poly[:,0]*(dv-ev)+poly[:,1]*T<=Y+ey+1e-7) and np.all(poly[:,0]*(dv+ev)+poly[:,1]*T>=Y-ey-1e-7))
        check(oid+' recovery boundary sampled',any(x['label']=='retreat_route_preview' or x['label'].startswith(('live_PROBE_UNLOAD','live_PROBE_HOVER')) for x in snaps))
        if 'planning_stats' in result:
            stats=result['planning_stats']
            check(oid+' formal geometry is current',formal['geometry_time']==formal['time'] and formal['route_preview_status']=='fresh_formal_geometry')
            check(oid+' exactly one fresh formal comparison',stats['formal_recomputations']==1)
            check(oid+' empty mechanical evidence never invokes preview',not any(x['windows']==0 and x['label'].startswith('live_') for x in snaps))
            stages={x[-1] for x in result['frames']}
            check(oid+' fast and near contact phases recorded','PROBE_APPROACH_FAST' in stages and 'PROBE_APPROACH_NEAR' in stages)
            requests=result.get('preview_requests',[])
            for a,b in zip(requests,requests[1:]):
                changed=(b['map_changed'] or np.linalg.norm(np.array(b['position_xy'])-a['position_xy'])>=.04-1e-9 or abs(b['object_s_m']-a['object_s_m'])>=.01-1e-9)
                check(oid+' preview needs changed geometry',changed and b['time']-a['time']>=.35-1e-9)
            for sample in result['recovery_frames']:
                check(oid+' recovery reason channels recorded',all(k in sample for k in ('speed_ok','gyro_ok','tilt_ok','yaw_ok','height_ok','exit_path_checked')))
    if 'planning_stats' in d['interactions']['S01']:
        gate=next(x for x in d['decisions'] if x['opening']=='S01' and x['phase']=='precontact')
        check('S01 rejected before approach',gate['gate_location']=='before_approach' and gate['position_xy'][1]<4.)
        check('no wasted S01 observation approach',not any(f['current']=='S01' and f['stage']=='TO_OBSERVATION' for f in d['frames']))
    if require_boundary:
        check('push-door majority near boundary',.5<probabilities['P01']<.8)
        check('avoid-door majority near boundary',.2<probabilities['A01']<.5)
    detections=[a for frame in d['camera_frames'] for a in frame['detections'] if abs(a['time_s']-frame['time'])<1e-5]
    check('camera overlays have matching exposure',len(detections)>20)
    check('all three classes inferred from RGB',{'door','box','suspected_glass'}<={a['class_name'] for a in detections})
    check('HTML generated',(folder/'report.html').is_file())
    out=dict(passed=True,count=len(checks),summary=s,probabilities=probabilities,checks=checks)
    (folder/'validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=len(checks),probabilities=probabilities),ensure_ascii=False));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--allow-nonboundary',action='store_true');a=p.parse_args()
    validate(a.folder,not a.allow_nonboundary)

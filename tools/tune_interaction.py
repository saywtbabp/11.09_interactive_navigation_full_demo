"""Offline scene design ONLY. Restored states are never used by main.py.

Replay a physical checkpoint with different friction to find majority-boundary
examples, keeping the controller, lambda, observations and decision rule fixed.
"""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import runtime
import mujoco,numpy as np
from configuration import Config
from mission import Mission
from navigation.perception import Record
from session import InteractionSession

def trial(checkpoints,oid,friction,output,inertia=None):
    spec=json.loads((ROOT/'config/scene.json').read_text(encoding='utf-8'))
    spec['door_parameters'][oid][1]=friction
    if inertia is not None:spec['door_parameters'][oid][0]=inertia
    output.mkdir(parents=True,exist_ok=True);scene=output/'scene.json'
    scene.write_text(json.dumps(spec),encoding='utf-8')
    m=Mission(Config(),output,scene_file=scene);w=m.world
    saved=np.load(checkpoints/(oid+'.npz'))
    mujoco.mj_setState(w.model,w.data,saved['state'],int(saved['state_spec']))
    w.target=saved['target'].copy();w.ctrl=saved['ctrl'].copy()
    w.yaw_reference=float(saved['yaw_reference']);w.yaw_target=float(saved['yaw_target']);w.step_count=int(saved['step_count'])
    mujoco.mj_forward(w.model,w.data)
    meta=json.loads((checkpoints/(oid+'.json')).read_text(encoding='utf-8'))
    m.memory.records={k:Record(**v) for k,v in meta['records'].items()};m.current=oid
    s=InteractionSession(m,m.memory.records[oid]);m.active_session=m.display_session=s
    # Rendering and perception are irrelevant to this offline dynamics check.
    # The checkpoint already contains independently acquired class + geometry.
    def tick(world):
        s.observe_state();s.capture(world)
    w.callback=tick;m.save_key=lambda label:None
    try:s.execute()
    finally:m.close()
    snapshot=s.snapshots[-1] if s.snapshots else {}
    first=next((x for x in s.online_pairs if x['choice'] in ('continue','avoid')),None)
    return dict(object=oid,inertia=spec['door_parameters'][oid][0],friction=friction,outcome=s.outcome,
        first_online_choice=first,contact_peak_N=s.peak,wall_peak_N=w.wall_peak,p=snapshot.get('push_probability'),
        Fstar=snapshot.get('push_limit'),gain=snapshot.get('gain'),windows=snapshot.get('windows'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('object');p.add_argument('friction',type=float,nargs='+')
    p.add_argument('--checkpoints',type=Path,default=ROOT/'results/release/checkpoints')
    p.add_argument('--inertia',type=float,help='offline scene-design inertia, kg m^2; omit to preserve the scene value')
    a=p.parse_args();rows=[]
    for value in a.friction:
        suffix='' if a.inertia is None else f'_I{a.inertia:.6f}'
        row=trial(a.checkpoints,a.object,value,ROOT/'results/tuning'/f'{a.object}_{value:.6f}{suffix}',a.inertia)
        rows.append(row);print('OFFLINE_RESULT',json.dumps(row),flush=True)
    (ROOT/'results/tuning'/f'{a.object}_latest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

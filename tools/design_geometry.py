"""Offline map-cost study from acquired post-probe evidence (no flight)."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import runtime,mujoco,numpy as np
from mission import Mission
from configuration import Config
from session import InteractionSession
from navigation.perception import Record
from belief import Belief

p=argparse.ArgumentParser();p.add_argument('object');p.add_argument('friction',type=float);p.add_argument('gap');p.add_argument('positions',type=float,nargs='+');a=p.parse_args()
folder=ROOT/'results/tuning'/f'{a.object}_{a.friction:.6f}'
result=json.loads((folder/'interactions'/a.object/'interaction.json').read_text(encoding='utf-8'))
snap=next(s for s in result['snapshots'] if s['label']=='replan')
frame=min(result['frames'],key=lambda f:abs(f[0]-snap['time']))
m=Mission(Config(),ROOT/'results/geometry_study');w=m.world
meta=json.loads((ROOT/'results/release/checkpoints'/(a.object+'.json')).read_text(encoding='utf-8'))
m.memory.records={k:Record(**v) for k,v in meta['records'].items()}
s=InteractionSession(m,m.memory.records[a.object]);jid=s.joint
w.data.qpos[w.model.jnt_qposadr[jid]]=frame[4];mujoco.mj_forward(w.model,w.data)
s.belief.windows=snap['constraints'];s.belief.precision=np.array(snap['precision']);s.belief.linear=np.array(snap['linear']);s.belief.rebuild()
for value in a.positions:
    o=next(o for o in m.arch['openings'] if o['id']==a.gap);o['center_xy_m'][0]=value
    m.memory.records[a.gap].geometry=o
    _,_,metrics=s.route_metrics(np.array(frame[1:3]));prob,converged=s.belief.probability(metrics['push_limit'])
    print(json.dumps(dict(gap=a.gap,x=value,p=prob,converged=converged,limit=metrics['push_limit'],gain=metrics['gain'])),flush=True)
m.close()

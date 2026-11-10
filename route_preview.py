"""Geometry-only planning snapshots; the worker never accesses MuJoCo state."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import numpy as np
from local_hover import deferred_egress,hover_planner

def length(path):return float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())

class PlanningView:
    def __init__(self,e):
        self.time=float(e.world.data.time);self.q=e.state()[0];self.start=e.position().copy()
        self.target=e.target;self.staging=e.staging.copy();self.kind=e.kind
        self.arm=e.arm;self.cap=e.cap;self.rate=e.args.rate
        self.args=SimpleNamespace(probe_recovery=e.args.probe_recovery,alpha=e.args.alpha)
        self.goal=e.goal.copy()
        self.current=e.planner(self.q,'target');self.bypass=e.planner(self.q,'bypass')
        self.future=e.planner(self.target,'target')
        needs_exit=self.current.collides(self.start)[0] or self.bypass.collides(self.start)[0]
        self.local=hover_planner(e,self.q) if needs_exit else None

    def planner(self,q,route):
        if route=='bypass':return self.bypass
        return self.future if q==self.target else self.current

    def plan(self,start,q,route):
        path,_=self.planner(q,route).plan(start,self.goal);return path,length(path)

def calculate(view):
    started=time.perf_counter();q=view.q;start=view.start;prefix=None;comparison=start
    if view.current.collides(start)[0] or view.bypass.collides(start)[0]:
        prefix=deferred_egress(view,start,q,view.local);comparison=prefix[-1]
    bypass,Lb=view.plan(comparison,q,'bypass');candidate,Lc=view.plan(comparison,view.target,'target')
    if prefix is not None:
        extra=length(prefix);Lb+=extra;Lc+=extra
        bypass=np.vstack((prefix,bypass[1:]));candidate=np.vstack((prefix,candidate[1:]))
    stroke=max(view.target-q,0.)*view.arm;manoeuvre=2*stroke+.56 if stroke>0 else 0.
    gain=Lb-Lc-manoeuvre;economic=gain/(view.rate*stroke) if view.rate*stroke>0 else None
    limit=min(view.cap,economic) if economic is not None else view.cap
    if gain<=0:limit=0.
    metrics=dict(capacity_limit=view.cap,economic_limit=economic,push_limit=limit,gain=gain,stroke=stroke,
        expected_delta=0.,manoeuvre_allowance_m=manoeuvre,alpha=view.args.alpha,bypass_length_m=Lb,
        candidate_length_m=Lc,deferred_exit_m=0. if prefix is None else length(prefix),
        deferred_exit_path=[] if prefix is None else prefix.tolist(),exit_execution='after_bypass_choice',
        geometry_q=q,geometry_xy=start.tolist(),geometry_time=view.time,planning_wall_s=time.perf_counter()-started)
    return bypass,candidate,metrics

class PreviewWorker:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='geometry-preview')
        self.future=None;self.closed=False
        self.sample_time=None

    def submit(self,view):
        if self.closed or self.future is not None:return False
        self.sample_time=view.time
        self.future=self.pool.submit(calculate,view);return True

    def poll(self):
        if self.future is None or not self.future.done():return None
        future=self.future;self.future=None
        try:return dict(metrics=future.result()[2])
        except ValueError as exc:return dict(error=str(exc),geometry_time=self.sample_time)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.future is not None:self.future.cancel()
        self.pool.shutdown(wait=True)

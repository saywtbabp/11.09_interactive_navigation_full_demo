"""Geometric comparison at a measured post-contact state.

The conservative square planning envelope can overlap an inflated jamb by a few
millimetres while the real flexible vehicle is physically in contact elsewhere.
Ordinary A* must remain strict. An explicit, common recovery connector is allowed
only if it monotonically exits the INITIAL inflation overlap and never enters a
new obstacle. After that connector every edge retains the full .34 m margin.

This module computes geometric value only. It neither moves the physical drone
nor changes candidate physical rollout states, labels or stored predictions.
"""
import numpy as np
from .geometry import Planner,door_comparison
from .interactions import approach

def contact_comparison(arch,cfg,opening,start,merge):
    try:return door_comparison(arch,cfg,opening,start,merge)
    except ValueError as original:
        start=np.asarray(start)[:2];x,y=opening['center_xy_m']
        others=[o['id'] for o in arch['openings'] if o['id']!=opening['id'] and abs(o['center_xy_m'][1]-y)<.01]
        planner=Planner(arch,cfg,sealed=others,row=y)
        overlaps=[]
        for center,rotation,extent in planner.boxes:
            depth=float(np.min(extent-np.abs((start-center)@rotation)))
            overlaps.append(max(0.,depth))
        initial=np.array(overlaps)
        if not (0<float(initial.max())<=.04):raise original
        mid=np.rint((start-planner.origin)/planner.res).astype(int);candidates=[]
        for dx in range(-2,3):
            for dy in range(-2,3):
                cell=mid+[dx,dy]
                if not all(0<=cell[k]<planner.shape[k] for k in (0,1)):continue
                if planner.blocked[tuple(cell)]:continue
                end=planner.origin+cell*planner.res;points=np.linspace(start,end,25)
                valid=True
                for j,(center,rotation,extent) in enumerate(planner.boxes):
                    depth=np.maximum(0.,np.min(extent-np.abs((points-center)@rotation),axis=1))
                    if initial[j]>0:
                        if np.any(np.diff(depth)>1e-10):valid=False;break
                    elif depth.max()>1e-10:valid=False;break
                if valid:candidates.append((float(np.linalg.norm(end-start)),end))
        if not candidates:raise original
        connector,end=min(candidates,key=lambda item:item[0])
        retreat,length=planner.plan(end,approach(opening))
        prefix=np.vstack((start,retreat));prefix_length=connector+length
        result=door_comparison(arch,cfg,opening,approach(opening),merge)
        result['J_door_geo']+=prefix_length;result['J_bypass_geo']+=prefix_length
        result['G']=result['J_bypass_geo']-result['J_door_geo']
        result['door_path']=np.vstack((prefix,result['door_path'][1:]))
        result['bypass_path']=np.vstack((prefix,result['bypass_path'][1:]))
        result['common_start']=np.asarray(start)
        result['common_geometric_retreat_m']=prefix_length
        result['initial_inflation_overlap_m']=float(initial.max())
        result['monotone_contact_egress_connector_m']=connector
        result['geometry_assumption']='identical explicit post-contact retreat in both geometric branches; initial inflation overlap monotonically exited; full margin thereafter; physical continuation states unchanged'
        return result

"""Current-geometry, corner-safe metric cost fields for every online window.

Native code only accelerates rasterization and shortest paths. Inputs are the
same public geometry / own footprint used by Planner; no parameter truth enters.
"""
import ctypes,hashlib
from pathlib import Path
from collections import OrderedDict
import numpy as np

_path=Path(__file__).with_name('native')/'fast_grid.dll'
_lib=ctypes.CDLL(str(_path))
_double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
_byte=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
_lib.raster.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,_double,_byte]
_lib.distances.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double,_byte,ctypes.c_int,_double]
_lib.visible.argtypes=[ctypes.c_int,ctypes.c_int,_double,_double,_double,_byte]

def box_array(planner):
    return np.ascontiguousarray([[*c,*r.ravel(),*h] for c,r,h in planner.boxes],dtype=np.float64)

class CostField:
    def __init__(self,planner,goal,cache):
        self.planner=planner;self.boxes=box_array(planner);self.goal=np.asarray(goal,dtype=float)[:2]
        self.shape=planner.shape;self.origin=planner.origin;self.step=planner.res
        self.blocked=np.empty(self.shape,dtype=np.uint8)
        code=_lib.raster(*self.shape,*self.origin,self.step,len(self.boxes),self.boxes,self.blocked)
        if code:raise RuntimeError('native raster error '+str(code))
        cells=self.attach(self.goal);points=self.origin+cells*self.step
        goal_index=int(np.argmin(np.linalg.norm(points-self.goal,axis=1)))
        cell=cells[goal_index];self.goal_offset=float(np.linalg.norm(points[goal_index]-self.goal))
        key=(self.shape,self.step,*self.origin,*cell,hashlib.sha256(self.blocked.tobytes()).digest())
        if key in cache:
            self.dist=cache.pop(key);cache[key]=self.dist;self.rebuilt=False
        else:
            self.dist=np.empty(self.shape,dtype=np.float64)
            code=_lib.distances(*self.shape,self.step,self.blocked,int(cell[0]*self.shape[1]+cell[1]),self.dist)
            if code:raise RuntimeError('native shortest-path error '+str(code))
            cache[key]=self.dist;self.rebuilt=True
            while len(cache)>8:cache.popitem(last=False)

    def visible(self,starts,ends):
        a=np.ascontiguousarray(np.atleast_2d(starts),dtype=np.float64);b=np.ascontiguousarray(np.atleast_2d(ends),dtype=np.float64)
        a,b=np.broadcast_arrays(a,b);a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
        result=np.empty(len(a),dtype=np.uint8)
        code=_lib.visible(len(a),len(self.boxes),self.boxes,a,b,result)
        if code:raise RuntimeError('native visibility error '+str(code))
        return result.astype(bool)

    def attach(self,point):
        point=np.asarray(point,dtype=float)[:2]
        mid=np.rint((point-self.origin)/self.step).astype(int)
        cells=mid+np.array([(x,y) for x in range(-2,3) for y in range(-2,3)])
        inside=np.all((cells>=0)&(cells<np.array(self.shape)),axis=1);cells=cells[inside]
        cells=cells[~self.blocked[cells[:,0],cells[:,1]].astype(bool)]
        if not len(cells):raise ValueError('endpoint has no free cost-field cell')
        points=self.origin+cells*self.step
        if np.any(point<self.planner.origin) or np.any(point>self.planner.upper):raise ValueError('endpoint outside public map')
        # Endpoint visibility also rejects continuous obstacle overlap.
        cells=cells[self.visible(point,points)]
        if not len(cells):raise ValueError('endpoint lacks continuous envelope clearance')
        return cells

    def possibly_reachable(self,points):
        """Cheap optimistic filter, with the SAME +/-2 attachment cells.

        False certifies unreachability; True still needs exact visibility and
        connector checks. This avoids expensive release sweeps below a cut.
        """
        points=np.atleast_2d(points);mid=np.rint((points-self.origin)/self.step).astype(int)
        result=np.zeros(len(points),dtype=bool)
        for dx in range(-2,3):
            for dy in range(-2,3):
                cells=mid+[dx,dy];valid=np.all((cells>=0)&(cells<self.shape),axis=1)
                idx=np.flatnonzero(valid);x,y=cells[valid].T
                result[idx] |= (~self.blocked[x,y].astype(bool)) & np.isfinite(self.dist[x,y])
        return result

    def cost(self,point):
        point=np.asarray(point,dtype=float)[:2];cells=self.attach(point)
        values=self.dist[cells[:,0],cells[:,1]]+np.linalg.norm(self.origin+cells*self.step-point,axis=1)+self.goal_offset
        value=float(values.min())
        if not np.isfinite(value):raise ValueError('no cost-field route to goal')
        return value

def new_cache():return OrderedDict()

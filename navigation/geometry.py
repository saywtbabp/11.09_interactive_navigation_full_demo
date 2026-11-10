"""Metric map and corner-safe 8-neighbour A*. Only public geometry and memory enter."""
import heapq
import math
import numpy as np

def wall_boxes(arch):
    b=arch['bounds_wall_centerlines_m']; t=arch['wall_thickness_m']/2
    x0,x1,y0,y1=b['xmin'],b['xmax'],b['ymin'],b['ymax']
    boxes=[]
    def box(name,a,b,c,d):
        boxes.append((name,np.array([(a+b)/2,(c+d)/2]),np.array([(b-a)/2,(d-c)/2])))
    box('south',x0-t,x1+t,y0-t,y0+t); box('north',x0-t,x1+t,y1-t,y1+t)
    box('west',x0-t,x0+t,y0,y1); box('east',x1-t,x1+t,y0,y1)
    lookup={o['id']:o for o in arch['openings']}
    for row in arch['internal_walls']:
        left=x0; y=row['y_m']
        for oid in row['opening_ids']:
            o=lookup[oid]; x=o['center_xy_m'][0]; w=o['width_m']
            box(row['id']+'_'+oid,left,x-w/2,y-t,y+t); left=x+w/2
        box(row['id']+'_end',left,x1,y-t,y+t)
    return boxes

def rectangle(center,half,angle=0.):
    c,s=math.cos(angle),math.sin(angle)
    return (np.asarray(center),np.array([[c,-s],[s,c]]),np.asarray(half))

class Planner:
    def __init__(self,arch,cfg,panels=(),sealed=(),row=None,footprint=None,region=None):
        self.arch,self.cfg=arch,cfg; self.res=cfg.grid_resolution
        self.footprint=np.full(2,cfg.clearance) if footprint is None else np.asarray(footprint,float)
        if self.footprint.shape!=(2,) or not np.isfinite(self.footprint).all() or np.any(self.footprint<=0):
            raise ValueError('invalid vehicle footprint')
        b=arch['bounds_wall_centerlines_m']
        self.origin=np.array([b['xmin'],b['ymin']]); self.upper=np.array([b['xmax'],b['ymax']])
        if region is not None:
            xmin,xmax,ymin,ymax=region
            self.origin=np.maximum(self.origin,[xmin,ymin]);self.upper=np.minimum(self.upper,[xmax,ymax])
            # Grid bounds are rounded outward within the public map.
            self.origin=np.floor(self.origin/self.res+1e-9)*self.res
            self.upper=np.ceil(self.upper/self.res-1e-9)*self.res
        self.shape=tuple(np.rint((self.upper-self.origin)/self.res).astype(int)+1)
        self.boxes=[rectangle(c,h+self.footprint) for name,c,h in wall_boxes(arch)
                    if row is None or name in ('east','west') or abs(c[1]-row)<.01]
        for o in arch['openings']:
            x,y=o['center_xy_m']
            if row is not None and abs(y-row)>.01:continue
            for sign in (-1,1):
                self.boxes.append(rectangle([x+sign*(o['width_m']/2-.045),y],np.array([.045,.045])+self.footprint))
        for o in arch['openings']:
            if o['id'] in sealed:
                self.boxes.append(rectangle(o['center_xy_m'],np.array([o['width_m']/2,.06])+self.footprint))
        for c,r,h in panels:
            # Square envelope support along each oriented panel axis.
            self.boxes.append((np.array(c),np.array(r),np.array(h)+self.footprint@np.abs(r)))
        # Passage guards only query segments. Build the raster only if A* needs it.
        self._blocked=None

    @property
    def blocked(self):
        if self._blocked is None:
            cells=np.indices(self.shape).reshape(2,-1).T
            self._blocked=self.collides(self.origin+cells*self.res).reshape(self.shape)
        return self._blocked

    def collides(self,points):
        p=np.atleast_2d(points)[...,:2]
        hit=np.any((p<self.origin)|(p>self.upper),axis=1)
        for c,r,h in self.boxes:
            hit |= np.all(np.abs((p-c)@r)<=h+1e-9,axis=1)
        return hit

    def clear(self,a,b):
        a,b=np.asarray(a)[:2],np.asarray(b)[:2]
        n=max(2,int(np.linalg.norm(b-a)/(self.res*.4))+2)
        return not self.collides(np.linspace(a,b,n)).any()

    def connect(self,p):
        if self.collides(p)[0]: raise ValueError('endpoint lacks vehicle clearance: '+str(p))
        mid=np.rint((p-self.origin)/self.res).astype(int); candidates=[]
        for dx in range(-2,3):
            for dy in range(-2,3):
                cell=tuple(mid+[dx,dy])
                if all(0<=cell[k]<self.shape[k] for k in (0,1)) and not self.blocked[cell]:
                    q=self.origin+np.array(cell)*self.res
                    if self.clear(p,q): candidates.append((float(np.linalg.norm(p-q)),cell))
        if not candidates: raise ValueError('cannot attach endpoint to A*')
        return min(candidates)[1]

    def plan(self,start,goal):
        start,goal=np.asarray(start)[:2],np.asarray(goal)[:2]
        # A completely blocked grid row/column separates the endpoints. Reject
        # these impossible searches before expanding thousands of A* cells.
        # connect() may choose a cell within +/-2, so exclude that full allowance.
        mid=np.rint((np.array([start,goal])-self.origin)/self.res).astype(int)
        for axis in (0,1):
            lo=max(0,int(mid[:,axis].min())+2)
            hi=min(self.shape[axis]-1,int(mid[:,axis].max())-2)
            if lo<=hi and self.blocked.all(axis=1-axis)[lo:hi+1].any():
                raise ValueError('no A* route: blocked grid cut')
        a,b=self.connect(start),self.connect(goal)
        queue=[(0.,a)]; cost={a:0.}; parent={}; closed=set()
        while queue:
            _,p=heapq.heappop(queue)
            if p in closed: continue
            closed.add(p)
            if p==b: break
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                q=(p[0]+dx,p[1]+dy)
                if not all(0<=q[k]<self.shape[k] for k in (0,1)): continue
                if self.blocked[q] or self.blocked[p[0]+dx,p[1]] or self.blocked[p[0],p[1]+dy]: continue
                g=cost[p]+math.hypot(dx,dy)*self.res
                if g<cost.get(q,float('inf')):
                    cost[q]=g;parent[q]=p;heapq.heappush(queue,(g+math.dist(q,b)*self.res,q))
        if b not in closed: raise ValueError('no A* route')
        chain=[b]
        while chain[-1]!=a: chain.append(parent[chain[-1]])
        raw=np.array([start]+[self.origin+np.array(c)*self.res for c in reversed(chain)]+[goal])
        length=float(np.linalg.norm(np.diff(raw,axis=0),axis=1).sum())
        route=[start];i=0
        while i<len(raw)-1:
            j=len(raw)-1
            while j>i+1 and not self.clear(raw[i],raw[j]):j-=1
            route.append(raw[j]);i=j
        return np.array(route),length

def door_comparison(arch,cfg,opening,start,merge):
    y=opening['center_xy_m'][1]
    others=[o['id'] for o in arch['openings'] if o['id']!=opening['id'] and abs(o['center_xy_m'][1]-y)<.01]
    door_planner=Planner(arch,cfg,sealed=others,row=y)
    bypass_planner=Planner(arch,cfg,sealed=[opening['id']],row=y)
    prefix=np.empty((0,2));prefix_length=0.;comparison_start=np.asarray(start)
    if bypass_planner.collides(start)[0]:
        # A post-contact vehicle can occupy the hypothetical closed-aperture
        # exclusion zone. Use an explicit identical A* retreat prefix in BOTH
        # geometric branches, rather than silently reuse the old G. Physical
        # continuation rollouts still start at the actual observed post-state.
        x,y=opening['center_xy_m'];comparison_start=np.array([x+.24375,y-.95])
        prefix,prefix_length=door_planner.plan(start,comparison_start)
    direct,jd=door_planner.plan(comparison_start,merge)
    bypass,jb=bypass_planner.plan(comparison_start,merge)
    jd+=prefix_length;jb+=prefix_length
    if len(prefix):direct=np.vstack((prefix,direct[1:]));bypass=np.vstack((prefix,bypass[1:]))
    return dict(G=jb-jd,J_door_geo=jd,J_bypass_geo=jb,door_path=direct,bypass_path=bypass,
                common_start=np.asarray(start),common_merge=np.asarray(merge),units='m',
                common_geometric_retreat_m=prefix_length,
                geometry_assumption='door hypothetically passable; unknown others optimistic; explicit common retreat if post-state lies in virtual closed-aperture exclusion')

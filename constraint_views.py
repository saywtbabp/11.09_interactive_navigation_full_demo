"""Display each original window band without discarding its excluded context."""
import runtime
import numpy as np
from belief import clip

PALETTE = ['#70b6ff','#b69aff','#66d6ec','#d29ee8','#91b8e9','#88a0ff',
           '#63c4e3','#b188f3','#8bcaef','#d0b6ff','#6b93d6','#aab5ee','#ae85cf','#86b3d0']

def view_bounds(bounds, limit, constraints):
    # View limits only, never supplied to the estimator or probability integral.
    xmax=max(1.5,float(bounds[0][1])*1.20) if bounds is not None else 4.
    ymax=max(.1,float(bounds[1][1])*1.2) if bounds is not None else 2.
    if limit is not None: ymax=max(ymax,max(limit,0.)*1.15)
    return 0.,xmax,0.,ymax

def band_geometry(window, extent):
    dv,T,Y,ev,ey=window
    x0,x1,y0,y1=extent
    box=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)
    band=clip(box,np.array([dv-ev,T]),Y+ey)
    if len(band):band=clip(band,-np.array([dv+ev,T]),-(Y-ey))
    segments=[]
    for slope,offset in ((dv-ev,Y+ey),(dv+ev,Y-ey)):
        points=[]
        for x in (x0,x1):
            y=(offset-slope*x)/T
            if y0-1e-9<=y<=y1+1e-9:points.append((x,y))
        if abs(slope)>1e-12:
            for y in (y0,y1):
                x=(offset-T*y)/slope
                if x0-1e-9<=x<=x1+1e-9:points.append((x,y))
        points=list(dict.fromkeys(points))
        if len(points)>=2:segments.append([points[0],points[-1]])
    return band,segments

def excluded_candidates(constraints, polygon, extent, nx=28, ny=26):
    """Display-only grid: satisfies some windows but fails their intersection.

    These pale crosses carry ZERO effective weight and are never integrated.
    """
    if not constraints:return np.empty((0,2))
    x0,x1,y0,y1=extent
    xx,yy=np.meshgrid(x0+(np.arange(nx)+.5)*(x1-x0)/nx,
                      y0+(np.arange(ny)+.5)*(y1-y0)/ny)
    p=np.column_stack((xx.ravel(),yy.ravel()))
    some=np.zeros(len(p),bool);all_pass=np.ones(len(p),bool)
    for dv,T,Y,ev,ey in constraints:
        passed=((dv-ev)*p[:,0]+T*p[:,1]<=Y+ey+1e-10)&((dv+ev)*p[:,0]+T*p[:,1]>=Y-ey-1e-10)
        some|=passed;all_pass&=passed
    # An empty polygon can mean UNBOUNDED, not excluded. Test the actual
    # halfplanes so early online windows do not turn all feasible strips gray.
    return p[some&~all_pass]

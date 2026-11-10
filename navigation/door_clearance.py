"""Own collision geometry -> conservative envelope -> target-door route angle.

Reads only own geometry and public/observed architecture. No door I/friction,
world clone, state editing, physics step, or contact-solver forward call.
Angles are searched in geometry only. Unknown other apertures cannot supply a
fake crossing: every other aperture in the target wall is sealed for this test.
"""
from dataclasses import dataclass, asdict
from itertools import product
import math
import numpy as np
import mujoco
from .geometry import Planner
from .interactions import panel_geometry, approach, merge_pose


def collision_bounds(world,body_frame=False):
    """Conservative exact support AABBs for supported primitive geoms."""
    m,d=world.model,world.data
    origin=world.observed()[0]
    basis=world.observed()[2].T if body_frame else np.eye(3)
    bounds=[]
    for g in sorted(world.robot_geoms):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        if getattr(world,'onboard',None) is not None:
            pos,rotation=world.onboard.geom_pose(g)
        else:pos,rotation=d.geom_xpos[g],d.geom_xmat[g].reshape(3,3)
        R=basis@rotation;c=basis@(pos-origin)
        size=m.geom_size[g];kind=m.geom_type[g];axis=R[:,2]
        if kind==mujoco.mjtGeom.mjGEOM_BOX:h=np.abs(R)@size
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:h=np.full(3,size[0])
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:h=size[0]+size[1]*np.abs(axis)
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            h=size[0]*np.sqrt(np.maximum(0.,1-axis*axis))+size[1]*np.abs(axis)
        elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:h=np.sqrt((R*size)**2@np.ones(3))
        else:raise ValueError('Unsupported own collision geom: '+str(kind))
        bounds.append((c-h,c+h))
    if not bounds:raise ValueError('No own collision geometry')
    return bounds


def rotated_support(component,norm,angle):
    """Maximum absolute projection under ANY rotation of at most angle."""
    component=np.minimum(np.abs(component),norm)
    other=np.sqrt(np.maximum(0.,norm*norm-component*component))
    return np.where(np.arctan2(other,component)<=angle,norm,
                    component*np.cos(angle)+other*np.sin(angle))


@dataclass(frozen=True)
class VehicleEnvelope:
    nominal_half: np.ndarray
    half: np.ndarray
    geom_count: int
    position_margin: float
    flex_margin: float
    tilt_deg: float
    yaw_deg: float

    @classmethod
    def from_world(cls,w):
        cfg=w.cfg
        if (min(cfg.DOOR_POSITION_MARGIN,cfg.DOOR_FLEX_MARGIN,cfg.DOOR_ANGLE_MARGIN_DEG)<0
                or not 0<cfg.DOOR_CLEARANCE_STEP_DEG<=5
                or not 0<=cfg.DOOR_PASS_TILT_DEG<45 or not 0<=cfg.DOOR_PASS_YAW_DEG<45):
            raise ValueError('invalid door clearance margins')
        bounds=collision_bounds(w,body_frame=True)
        vertices=np.array([np.where(s,hi,lo) for lo,hi in bounds for s in product((False,True),repeat=3)])
        nominal=np.max(np.abs(vertices),axis=0)
        norm=np.linalg.norm(vertices,axis=1);radial=np.linalg.norm(vertices[:,:2],axis=1)
        yaw=np.radians(cfg.DOOR_PASS_YAW_DEG);tilt=np.radians(cfg.DOOR_PASS_TILT_DEG)
        half=np.empty(3)
        for axis in (0,1):
            yaw_support=rotated_support(vertices[:,axis],radial,yaw)
            half[axis]=np.max(rotated_support(yaw_support,norm,tilt))
        half[2]=np.max(rotated_support(vertices[:,2],norm,tilt))
        half+=cfg.DOOR_POSITION_MARGIN+cfg.DOOR_FLEX_MARGIN
        return cls(nominal,half,len(bounds),cfg.DOOR_POSITION_MARGIN,cfg.DOOR_FLEX_MARGIN,
                   cfg.DOOR_PASS_TILT_DEG,cfg.DOOR_PASS_YAW_DEG)

    def record(self):return asdict(self)


class DoorClearance:
    def __init__(self,world):
        self.world=world;self.cfg=world.cfg;self.arch=world.arch
        self.envelope=VehicleEnvelope.from_world(world)
        self.solutions={}
        self.progress=None

    def planner(self,opening,angle,*,uncertainty=True,tracking_allowance=True,extra_angle_margin=0.):
        # Full public walls, target row's alternative apertures sealed.
        y=opening['center_xy_m'][1]
        sealed=[o['id'] for o in self.arch['openings']
                if o['id']!=opening['id'] and abs(o['center_xy_m'][1]-y)<.01]
        margin=(np.radians(self.cfg.DOOR_ANGLE_MARGIN_DEG) if uncertainty else 0.)+extra_angle_margin
        low=max(self.cfg.DOOR_ANGLE_MIN,angle-margin);high=min(self.cfg.DOOR_ANGLE_MAX,angle+margin)
        count=max(1,int(np.ceil((high-low)/np.radians(1.)))+1) if margin else 1
        angles=np.linspace(low,high,count)
        # Cover the continuous sweep BETWEEN samples as well as their endpoints.
        spacing=(high-low)/(count-1) if count>1 else 0.
        sweep_pad=2*(opening['width_m']-.18+.06)*np.sin(spacing/4)
        panels=[]
        for a in angles:
            center,rotation,extent=panel_geometry(opening,a)[2]
            panels.append((center,rotation,extent+sweep_pad))
        half=self.envelope.half[:2]-(0. if tracking_allowance else self.cfg.DOOR_POSITION_MARGIN)
        b=self.arch['bounds_wall_centerlines_m'];padding=self.envelope.half[1]+2*self.cfg.grid_resolution
        # Limit search to this wall's approach/exit band; no whole-building detour.
        region=(b['xmin'],b['xmax'],approach(opening)[1]-padding,merge_pose(opening)[1]+padding)
        return Planner(self.arch,self.cfg,panels=panels,sealed=sealed,footprint=half,region=region)

    def solve(self,opening):
        key=opening['id']
        sensed=opening.get('sensed_geometry',{})
        signature=np.round(np.r_[sensed.get('hinge_xy_m',[0.,0.]),sensed.get('leaf_center_offset_m',0.)]/.02).astype(int).tolist()
        if key in self.solutions and self.solutions[key].get('geometry_signature')==signature:return self.solutions[key]
        cfg=self.cfg;height=self.arch['flight_height_m'];e=self.envelope
        record=dict(feasible=False,theta_clear=None,theta_clear_deg=None,envelope=e.record(),
                    geometry_signature=signature,geometry_measurement=sensed.copy(),
                    angle_margin_deg=cfg.DOOR_ANGLE_MARGIN_DEG,
                    search_resolution_deg=cfg.DOOR_CLEARANCE_STEP_DEG,
                    source='own primitive collision geometry + tilt/yaw/flex/position margins + target-only A*')
        if 2*e.half[0]>=opening['width_m']-.18:
            record['reason']='vehicle envelope exceeds clear jamb width'
        elif height-e.half[2]<=0 or height+e.half[2]>=cfg.DOOR_CLEAR_HEIGHT:
            record['reason']='vehicle envelope exceeds floor/lintel height'
        else:
            step=np.radians(cfg.DOOR_CLEARANCE_STEP_DEG)
            angles=np.unique(np.r_[np.arange(cfg.DOOR_ANGLE_MIN,cfg.DOOR_ANGLE_MAX,step),cfg.DOOR_ANGLE_MAX])
            for angle in angles:
                if self.progress:self.progress()
                planner=self.planner(opening,float(angle))
                try:path,length=planner.plan(approach(opening),merge_pose(opening))
                except ValueError:continue
                record.update(feasible=True,theta_clear=float(angle),theta_clear_deg=float(np.degrees(angle)),
                              route=path,route_length_m=length,reason='first feasible searched angle with full margins')
                break
            else:record['reason']='no target-door route within public hinge limits'
        self.solutions[key]=record
        return record

    def annotate(self,opening,geometry):
        return dict(geometry,clearance=self.solve(opening))

    def path(self,opening,angle,start):
        return self.planner(opening,angle).plan(start,merge_pose(opening))[0]

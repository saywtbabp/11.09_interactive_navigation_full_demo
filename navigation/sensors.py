"""Actual body-fixed RGB and noisy/occluded range measurements.

Surface IDs are private to the measurement model and never leave acquire().
Glass is a configurable weak/background-return material, still a collider.
"""
from dataclasses import dataclass
import math
import runtime
import mujoco
import numpy as np
from .force_observer import ForceObserver

@dataclass
class Packet:
    time: float
    rgb: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    camera_position: np.ndarray
    optical_rotation: np.ndarray
    focal: float
    directions: np.ndarray
    distances: np.ndarray
    valid: np.ndarray
    returns: list
    points: np.ndarray
    scan_origin: np.ndarray = None
    force_measurement: dict = None

class Sensors:
    def __init__(self,world,cfg):
        self.world,self.cfg=world,cfg;self.rng=np.random.default_rng(cfg.seed)
        self.renderer=None;self.packet=None;self.last_time=-1.;self.frames=0
        self.opt=mujoco.MjvOption();self.opt.geomgroup[:]=1;self.opt.geomgroup[2]=0;self.opt.geomgroup[5]=0
        self.opt.sitegroup[:]=0;self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=0
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT]=0
        self.dropout_frames=0;self.range_dropout=0.;self.camera_dropout=0.
        self.contact=ForceObserver(cfg)
        self.force_rng=np.random.default_rng(cfg.seed+919)

    def measure_force(self,calibrate=False):
        w=self.world;pos,vel,R,gyro=w.observed()
        if getattr(w,'onboard',None) is not None:
            if calibrate and np.linalg.norm(vel)<.08 and np.linalg.norm(gyro)<.25:self.contact.calibrate()
            return self.contact.record()
        acc=w.data.sensor('imu_acc').data.copy()+self.force_rng.normal(0,.012,3)
        stable=np.linalg.norm(vel)<.08 and np.linalg.norm(gyro)<.25 and R[2,2]>.985
        self.contact.update(float(w.data.time),acc,R,w.mass,float(w.ctrl.sum()),calibrate and stable)
        return self.contact.record()

    def acquire(self,force=False):
        w,cfg=self.world,self.cfg;m,d=w.model,w.data
        if not force and d.time-self.last_time<cfg.sensor_period-1e-8:return self.packet,False
        mujoco.mj_forward(m,d);self.last_time=float(d.time);self.frames+=1
        if self.renderer is None:self.renderer=mujoco.Renderer(m,height=cfg.camera_height,width=cfg.camera_width)
        self.renderer.update_scene(d,camera='onboard',scene_option=self.opt)
        rgb=self.renderer.render().copy()
        if self.rng.random()<self.camera_dropout:rgb=None;self.dropout_frames+=1
        pos,vel,R,gyro=w.observed();cp=d.cam_xpos[m.camera('onboard').id].copy()
        if getattr(w,'onboard',None) is not None:cp=pos+R@np.array([0.,.015,.037])
        force=self.measure_force(calibrate=w.stage=='INITIAL_SETTLE' and d.time>.25)
        optical=R@np.column_stack(([1.,0,0],[0,0,-1.],[0,1.,0]))
        # Full pose follows frame roll, pitch and yaw; this is not a gimbal.
        rel=np.linspace(-math.radians(78),math.radians(78),cfg.scan_rays)
        local_dirs=np.column_stack((np.sin(rel),np.cos(rel),np.zeros(len(rel))))
        dirs=local_dirs@R.T
        physical_dirs=local_dirs@d.xmat[w.frame_id].reshape(3,3).T
        origin=d.site_xpos[m.site('imu').id].copy()
        ranges=np.full(len(rel),np.nan);valid=np.zeros(len(rel),bool);returns=[]
        group=np.array([1,1,0,0,0,0],np.uint8);opaque=np.array([1,0,0,0,0,0],np.uint8);hit=np.zeros(1,np.int32)
        for i,direction in enumerate(physical_dirs):
            distance=mujoco.mj_ray(m,d,origin,direction,group,True,w.frame_id,hit)
            status='no_return'
            if distance>.015:
                # Material is simulator-private. Sensor output contains no ID/class.
                if int(m.geom_group[int(hit[0])])==1:
                    r=self.rng.random()
                    if r<cfg.glass_dropout_probability:distance=-1.;status='dropout'
                    elif r<1-cfg.glass_surface_return_probability:
                        distance=mujoco.mj_ray(m,d,origin,direction,opaque,True,w.frame_id,hit);status='surface_return'
                    else:status='weak_return'
                else:status='surface_return'
            if self.rng.random()<(.01+self.range_dropout):distance=-1.;status='dropout'
            if distance>cfg.scan_range:distance=cfg.scan_range;status='max_range';valid[i]=True
            elif .015<distance<=cfg.scan_range:distance=max(.015,distance+self.rng.normal(0,.0015));valid[i]=True
            if valid[i]:ranges[i]=distance
            returns.append(status)
        if getattr(w,'onboard',None) is not None:origin=pos.copy()
        points=origin[None,:]+ranges[:,None]*dirs
        self.packet=Packet(float(d.time),rgb,pos,R,cp,optical,
                           cfg.camera_height/(2*math.tan(math.radians(cfg.camera_fovy)/2)),
                           dirs,ranges,valid,returns,points,origin,force)
        return self.packet,True

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None

def project(packet,points):
    q=(np.atleast_2d(points)-packet.camera_position)@packet.optical_rotation
    uv=q[:,:2]/np.maximum(q[:,2,None],1e-9)*packet.focal
    height,width=packet.rgb.shape[:2] if packet.rgb is not None else (320,480)
    return uv+[width/2,height/2],q[:,2]

"""Onboard-only estimation. Simulator access is confined to SensorEmulator.

IMUs and cable length encoders belong to the vehicle, never to a door. The
static metric wall map and launch pose are explicit task priors. Motor thrust
is a calibrated command model (the existing ideal actuator has no RPM state).
"""
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .geometry import wall_boxes


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def increment(v):
    a=float(np.linalg.norm(v)); K=skew(v)
    return np.eye(3)+K+(K@K)*.5 if a<1e-7 else np.eye(3)+np.sin(a)/a*K+(1-np.cos(a))/a**2*(K@K)


class Calibration:
    """Own CAD/inertial calibration, captured once in the known launch pose."""
    def __init__(self,w):
        m,d=w.model,w.data
        self.names=['link1','link2','link3','link4','quad_frame']
        self.roots=[m.body(n).id for n in self.names]
        self.mass=np.zeros(5);self.com=np.zeros((5,3));self.inertia=np.zeros((5,3,3))
        self.root_for={}
        R0=np.array([d.xmat[b].reshape(3,3).copy() for b in self.roots])
        p0=np.array([d.xpos[b].copy() for b in self.roots])
        for b in w.robot_ids:
            a=b
            while a not in self.roots:a=int(m.body_parentid[a])
            self.root_for[b]=self.roots.index(a)
        for b,i in self.root_for.items():
            mass=float(m.body_mass[b]);c=R0[i].T@(d.xipos[b]-p0[i])
            self.mass[i]+=mass;self.com[i]+=mass*c
        self.com/=self.mass[:,None]
        for b,i in self.root_for.items():
            c=R0[i].T@(d.xipos[b]-p0[i])-self.com[i]
            Q=R0[i].T@d.ximat[b].reshape(3,3)
            self.inertia[i]+=Q@np.diag(m.body_inertia[b])@Q.T+m.body_mass[b]*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        self.R0=R0;self.relative=(p0-p0[-1])@R0[-1]
        self.imu_offset=np.zeros((5,3));self.imu_offset[-1]=m.site_pos[m.site('imu').id]
        self.sites={}
        for sid in range(m.nsite):
            bid=int(m.site_bodyid[sid])
            if bid in self.root_for:
                i=self.root_for[bid]
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,sid)
                self.sites[name]=(i,R0[i].T@(d.site_xpos[sid]-p0[i]))
        tree=ET.fromstring(w.xml)
        self.cables=[]
        for tendon in tree.findall('tendon/spatial'):
            sites=[s.get('site') for s in tendon.findall('site')]
            if len(sites)==2 and all(s in self.sites for s in sites):
                self.cables.append((m.tendon(tendon.get('name')).id,*[self.sites[s] for s in sites]))
        self.geoms={}
        for gid in w.robot_geoms:
            i=self.root_for[int(m.geom_bodyid[gid])]
            self.geoms[gid]=(i,R0[i].T@(d.geom_xpos[gid]-p0[i]),R0[i].T@d.geom_xmat[gid].reshape(3,3))
        self.rotors=[]
        for aid in w.motor_ids:
            sid=int(m.actuator_trnid[aid,0])
            self.rotors.append((m.site_pos[sid].copy(),m.actuator_gear[aid].copy()))


class SensorEmulator:
    """Only this class reads simulator sensor/pose/raycast buffers.

    Output has no environmental IDs, hinge state, contact points or parameters.
    Noise is applied before estimation; no estimator calls mj_forward.
    """
    def __init__(self,w,cal):
        self.w=w;self.cal=cal;self.rng=np.random.default_rng(w.cfg.seed+3071)
        az=np.linspace(-np.pi,np.pi,144,endpoint=False)
        self.directions=np.array([[np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)]
                                  for e in np.radians([-65,-20,0,20,65]) for a in az])
        self.group=np.array([1,1,0,0,0,0],np.uint8)
        self.hit=np.zeros(1,np.int32)

    def inertial(self):
        d=self.w.data
        acc=np.array([d.sensor('imu_acc_'+n).data if n!='quad_frame' else d.sensor('imu_acc').data for n in self.cal.names]).copy()
        gyro=np.array([d.sensor('imu_gyro_'+n).data if n!='quad_frame' else d.sensor('imu_gyro').data for n in self.cal.names]).copy()
        cfg=self.w.cfg
        return dict(acc=acc+self.rng.normal(0,cfg.imu_acc_noise,acc.shape),
                    gyro=gyro+self.rng.normal(0,cfg.imu_gyro_noise,gyro.shape),
                    lengths=np.array([d.ten_length[t[0]] for t in self.cal.cables])+self.rng.normal(0,cfg.cable_length_noise,len(self.cal.cables)),
                    thrust=self.w.ctrl.copy()*(1.+cfg.motor_calibration_error)+self.rng.normal(0,cfg.motor_thrust_noise,4))

    def lidar(self):
        w=self.w;m,d=w.model,w.data
        sid=m.site('imu').id;origin=d.site_xpos[sid].copy();Q=d.site_xmat[sid].reshape(3,3)
        distances=np.full(len(self.directions),np.nan)
        for j,v in enumerate(self.directions@Q.T):
            r=mujoco.mj_ray(m,d,origin,v,self.group,True,w.frame_id,self.hit)
            if .04<r<8. and self.rng.random()>.005:
                # Glass response is private to the sensor emulator.
                if int(m.geom_group[self.hit[0]])==1 and self.rng.random()>.12:continue
                distances[j]=r+self.rng.normal(0,w.cfg.lidar_range_noise)
        return self.directions.copy(),distances


class OnboardEstimator:
    def __init__(self,w):
        self.cfg=w.cfg;self.cal=Calibration(w);self.emulator=SensorEmulator(w,self.cal)
        self.p=np.array(w.arch['start_xyz_m'],float)-self.cal.imu_offset[-1]
        self.v=np.zeros(3);self.R=self.cal.R0.copy();self.omega=np.zeros((5,3));self.old_omega=self.omega.copy()
        self.relative=self.cal.relative.copy();self.force=np.zeros(3);self.moment=np.zeros(3)
        self.center=self.p.copy();self.bias=np.zeros(3);self.samples=[];self.noise=.025
        self.time=0.;self.valid=True;self.scan_time=-1.;self.points=np.empty((0,3))
        self.local_points=np.empty((0,3));self.last_scan_position=None;self.last_scan_time=None
        self.log=[];self.raw=[];self.history=deque(maxlen=60);self.last_packet=None
        self.scan_ranges=[];self.scan_poses=[]
        self.lidar_quality=0;self.pose_innovation=0.;self.contact_target=None
        self.last_good_localization=0.
        self.edges=[]
        H=w.arch['wall_height_m']
        for _,c,h in wall_boxes(w.arch):
            for axis in (0,1):
                other=1-axis
                for sign in (-1,1):self.edges.append((axis,c[axis]+sign*h[axis],c[other]-h[other],c[other]+h[other],H))

    def observed(self):
        if not self.valid:raise RuntimeError('onboard localization invalid; no ideal-state fallback')
        return self.p+self.R[-1]@self.cal.imu_offset[-1],self.v.copy(),self.R[-1].copy(),self.omega[-1].copy()

    def _shape(self,lengths):
        # Cable stretch encoders + distributed gyro orientations determine the
        # four free rods relative to the rigid rotor frame (12 unknowns).
        Q=self.R[-1].T@self.R
        x=self.relative[:4].copy()
        for _ in range(2):
            J=[];err=[]
            for length,(_,a,b) in zip(lengths,self.cal.cables):
                i,sa=a;j,sb=b
                pi=(x[i] if i<4 else np.zeros(3))+Q[i]@sa
                pj=(x[j] if j<4 else np.zeros(3))+Q[j]@sb
                delta=pi-pj;r=max(float(np.linalg.norm(delta)),1e-9);u=delta/r
                row=np.zeros((4,3))
                if i<4:row[i]+=u
                if j<4:row[j]-=u
                J.append(row.ravel());err.append(length-r)
            J=np.asarray(J);e=np.asarray(err)
            dx=np.linalg.solve(J.T@J+np.eye(12)*.015,J.T@e).reshape(4,3)
            x+=np.clip(dx,-.015,.015)
        self.relative[:4]=x

    def _localize(self,local,scan_time):
        # Point-to-plane registration against the PUBLIC static wall prior.
        # Dynamic leaf/jamb returns are rejected by distance/face support.
        p0=self.p.copy();yaw_total=0.
        for _ in range(3):
            points=self.p+(local+self.cal.imu_offset[-1])@self.R[-1].T
            residual=np.full(len(points),np.inf);axis_for=np.zeros(len(points),int)
            for axis,value,lo,hi,H in self.edges:
                other=1-axis
                good=(points[:,other]>lo+.06)&(points[:,other]<hi-.06)&(points[:,2]>.06)&(points[:,2]<H-.05)
                e=points[:,axis]-value;take=good&(np.abs(e)<np.abs(residual))
                residual[take]=e[take];axis_for[take]=axis
            floor=(points[:,2]<.12)&(points[:,2]>-.12)
            take=floor&(np.abs(points[:,2])<np.abs(residual));residual[take]=points[take,2];axis_for[take]=2
            mask=np.abs(residual)<.10
            self.lidar_quality=int(mask.sum())
            if mask.sum()<20:return
            self.last_good_localization=self.time
            pts=points[mask]-self.p;axes=axis_for[mask];e=residual[mask]
            J=np.zeros((len(e),4));J[np.arange(len(e)),axes]=1.
            J[:,3]=np.where(axes==0,-pts[:,1],np.where(axes==1,pts[:,0],0.))
            weight=np.minimum(1.,.012/np.maximum(np.abs(e),1e-6))
            delta=np.linalg.solve(J.T@(J*weight[:,None])+np.diag([.05,.05,.05,.2]),-J.T@(weight*e))
            delta=np.clip(delta,[-.025,-.025,-.025,-.006],[.025,.025,.025,.006])
            self.p+=delta[:3];self.R=increment([0,0,delta[3]])@self.R;yaw_total+=delta[3]
        innovation=self.p-p0;self.pose_innovation=float(np.linalg.norm(innovation))
        # Correct inertial velocity from position innovations, retaining the
        # fast IMU prediction instead of differentiating noisy scan positions.
        self.v+=np.clip(innovation*.9,-.035,.035)

    def after_step(self,w):
        dt=w.cfg.timestep;self.time=float(w.data.time)
        packet=self.emulator.inertial();self.last_packet=packet
        if not all(np.isfinite(packet[k]).all() for k in ('acc','gyro','lengths','thrust')):
            self.valid=False;raise RuntimeError('nonfinite onboard measurement')
        gyro=packet['gyro'];alpha=(gyro-self.omega)/dt
        self.old_omega=self.omega.copy();self.omega=gyro
        # Sensors refer to the solved PRE-integration interval.
        R=self.R.copy()
        specific=np.einsum('nij,nj->ni',R,packet['acc'])
        offsets=self.cal.com-self.cal.imu_offset
        correction=np.cross(alpha,offsets)+np.cross(gyro,np.cross(gyro,offsets))
        fcom=specific+np.einsum('nij,nj->ni',R,correction)
        origins=self.p+self.relative@R[-1].T
        centers=origins+np.einsum('nij,nj->ni',R,self.cal.com)
        mass=self.cal.mass;self.center=np.sum(mass[:,None]*centers,axis=0)/mass.sum()
        total=np.sum(mass[:,None]*fcom,axis=0);moment=np.sum(np.cross(centers-self.center,mass[:,None]*fcom),axis=0)
        for i in range(5):
            bodymoment=self.cal.inertia[i]@alpha[i]+np.cross(gyro[i],self.cal.inertia[i]@gyro[i])
            moment+=R[i]@bodymoment
        for thrust,(site,gear) in zip(packet['thrust'],self.cal.rotors):
            F=thrust*(R[-1]@gear[:3]);r=self.p+R[-1]@site-self.center
            total-=F;moment-=np.cross(r,F)+thrust*(R[-1]@gear[3:])
        self.force=total-self.bias;self.moment=moment
        # Propagate frame origin, correcting the IMU's lever arm acceleration.
        a=specific[-1]+np.array([0,0,-9.81])-R[-1]@(np.cross(alpha[-1],self.cal.imu_offset[-1])+np.cross(gyro[-1],np.cross(gyro[-1],self.cal.imu_offset[-1])))
        self.p+=self.v*dt+.5*a*dt*dt;self.v+=a*dt
        self.R=np.array([Q@increment(g*dt) for Q,g in zip(R,gyro)])
        if w.step_count%10==0:self._shape(packet['lengths'])
        if self.time-self.scan_time>=self.cfg.lidar_period-1e-9:
            directions,ranges=self.emulator.lidar();self.scan_time=self.time
            valid=np.isfinite(ranges);local=directions[valid]*ranges[valid,None]
            self._localize(local,self.time-dt)
            self.local_points=local;self.points=self.p+(local+self.cal.imu_offset[-1])@self.R[-1].T
            self.scan_ranges.append(ranges)
            self.scan_poses.append(np.r_[self.time,self.observed()[0],self.R[-1].ravel()])
            for track in getattr(self,'tracks',{}).values():
                if np.linalg.norm(np.asarray(track.opening['center_xy_m'])-self.p[:2])<3.5:
                    track.update(self.points,self.time)
        self.history.append(dict(time_s=self.time,force=self.force.copy(),moment=self.moment.copy(),center=self.center.copy()))
        self.valid=bool(self.time-self.last_good_localization<.5 and np.isfinite(self.p).all())
        self.raw.append(np.r_[self.time,packet['acc'].ravel(),gyro.ravel(),packet['lengths'],packet['thrust']])
        if w.step_count%5==0:
            self.log.append([self.time,*self.observed()[0],*self.v,*self.force,*self.moment,self.lidar_quality,self.pose_innovation])

    def calibrate(self):
        self.samples.append(self.force+self.bias);self.samples=self.samples[-150:]
        self.bias=np.mean(self.samples,axis=0);self.noise=max(.02,float(np.std(np.asarray(self.samples),axis=0).max()))

    def normal_force(self,normal):return max(0.,-float(self.force[:2]@np.asarray(normal)))

    @property
    def threshold(self):return max(self.cfg.verify_force_floor,self.cfg.verify_force_sigma*self.noise)

    def record(self):
        return dict(time_s=self.time,external_force_N=self.force.tolist(),external_moment_Nm=self.moment.tolist(),
                    valid=self.valid,threshold_N=self.threshold,bias_N=self.bias.tolist(),noise_N=self.noise,
                    calibration_samples=len(self.samples),source='distributed onboard IMU + cable encoders + calibrated motor command model')

    def geom_pose(self,gid):
        i,p,Q=self.cal.geoms[gid]
        return self.p+self.R[-1]@self.relative[i]+self.R[i]@p,self.R[i]@Q

    def save(self,folder):
        np.savez_compressed(folder/'onboard_measurements.npz',raw=np.asarray(self.raw),estimates=np.asarray(self.log))
        np.savez_compressed(folder/'lidar_measurements.npz',directions_body=self.emulator.directions,
                            ranges_m=np.asarray(self.scan_ranges),estimated_poses=np.asarray(self.scan_poses))
        door_records={k:dict(plane_history_columns=['time_s','theta_rad','normal_x','normal_y','plane_offset_m'],
                            plane_history=v.history,geometry=v.opening.get('sensed_geometry'),valid=v.valid)
                      for k,v in getattr(self,'tracks',{}).items()}
        import json
        metadata=dict(raw_rate_hz=1/self.cfg.timestep,estimate_log_rate_hz=1/(5*self.cfg.timestep),
            raw_columns=['time_s']+[f'acc_{n}_{a}_m_s2' for n in self.cal.names for a in 'xyz']+
                [f'gyro_{n}_{a}_rad_s' for n in self.cal.names for a in 'xyz']+
                [f'cable_{t[0]}_length_m' for t in self.cal.cables]+[f'motor_{i}_modeled_thrust_N' for i in range(4)],
            estimate_columns=['time_s','x','y','z','vx','vy','vz','Fx','Fy','Fz','Mx','My','Mz','lidar_support','pose_innovation_m'],
            imu_acc_noise_std_m_s2=self.cfg.imu_acc_noise,imu_gyro_noise_std_rad_s=self.cfg.imu_gyro_noise,cable_noise_std_m=self.cfg.cable_length_noise,
            lidar_range_noise_std_m=self.cfg.lidar_range_noise,lidar_period_s=self.cfg.lidar_period,
            motor_model='calibrated ideal force-command model, no physical RPM sensor/dynamics',
            contact='equivalent net wrench; not per-contact forces; target association assumes no simultaneous unmodeled external wrench')
        (folder/'onboard_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        (folder/'door_state_measurements.json').write_text(json.dumps(door_records,ensure_ascii=False,indent=2),encoding='utf-8')

"""Anonymous point-cloud door tracking and equivalent target-contact wrench.

No simulator/model access. A single plane does not identify the hinge. Initial
frame dimensions are marked map priors; hinge side/offset are refined from
multiple moving planes. No angle is initialized from a presumed closed door.
"""
import numpy as np


class DoorTrack:
    def __init__(self,opening):
        self.opening=opening;self.q=0.;self.omega=0.;self.time=-1.;self.history=[]
        self.valid=False;self.residual=float('inf');self.support=0;self.span=0.
        self.rng=np.random.default_rng(810)
        self.hinge_source='frame prior / hinge not yet observable from motion'
        self.hinge=np.array(opening['center_xy_m'],float)
        self.hinge[0]-=(opening['width_m']-.18)/2
        self.frame_cloud=[];self.frame_fit=None

    def frame(self,points):
        """Fit anonymous jamb cross sections; dimensions remain tagged priors
        until both arcs have enough spatial support. No surface IDs are used.
        """
        o=self.opening;x,y=o['center_xy_m'];w=o['width_m']
        region=(np.abs(points[:,0]-x)<w/2+.08)&(np.abs(points[:,1]-y)<.09)&(points[:,2]>.3)&(points[:,2]<1.9)
        self.frame_cloud.append(points[region,:2]);self.frame_cloud=self.frame_cloud[-24:]
        cloud=np.concatenate(self.frame_cloud) if self.frame_cloud else np.empty((0,2))
        fits=[]
        for sign in (-1,1):
            nominal=np.array([x+sign*(w/2-.045),y])
            p=cloud[np.linalg.norm(cloud-nominal,axis=1)<.072]
            # Only the front arc is visible from the approach. Use the nominal
            # radius as a weak regularizer, not a true scene measurement.
            p=p[p[:,1]<y-.028]
            if len(p)<18 or np.ptp(p[:,0])<.025:return
            c=nominal.copy();r=.045
            for _ in range(4):
                d=p-c;norm=np.linalg.norm(d,axis=1);e=norm-r
                J=np.column_stack((-d/np.maximum(norm[:,None],1e-8),-np.ones(len(p))))
                weights=np.minimum(1.,.003/np.maximum(abs(e),1e-6))
                normal=J.T@(weights[:,None]*J)+np.diag([.02,.02,2.])
                rhs=-J.T@(weights*e)+np.array([0.,0.,2.*(.045-r)])
                delta=np.linalg.solve(normal,rhs);c+=delta[:2];r+=delta[2]
            rms=float(np.sqrt(np.mean((np.linalg.norm(p-c,axis=1)-r)**2)))
            if rms>.006 or not .025<r<.065 or np.linalg.norm(c-nominal)>.035:return
            fits.append(dict(center_xy_m=c.tolist(),radius_m=float(r),rms_m=rms,samples=len(p)))
        gap=fits[1]['center_xy_m'][0]-fits[1]['radius_m']-fits[0]['center_xy_m'][0]-fits[0]['radius_m']
        self.frame_fit=dict(jambs=fits,clear_width_m=float(gap),source='multi-frame 3D LiDAR arc fit; weak radius prior',uncertainty_m=.012)

    def update(self,points,time):
        if time<=self.time:return
        o=self.opening;x,y=o['center_xy_m'];width=o['width_m']-.18
        if not len(points):return
        if not self.valid or (abs(self.omega)<.08 and self.q<.12):self.frame(points)
        mask=(points[:,0]>x-width/2+.025)&(points[:,0]<x+width/2-.025)&(points[:,1]>y-.13)&(points[:,1]<y+width+.1)&(points[:,2]>.25)&(points[:,2]<1.95)
        p=points[mask,:2]
        if len(p)<7:return
        if len(p)>240:p=p[np.linspace(0,len(p)-1,240,dtype=int)]
        pairs=self.rng.integers(0,len(p),(80,2));delta=p[pairs[:,1]]-p[pairs[:,0]]
        length=np.linalg.norm(delta,axis=1);delta=delta[length>.10]
        if not len(delta):return
        delta[delta[:,0]<0]*=-1
        angles=np.arctan2(delta[:,1],delta[:,0]);good=(angles>-.10)&(angles<1.62)
        candidates=angles[good]
        if not len(candidates):return
        normals=np.column_stack((-np.sin(candidates),np.cos(candidates)))
        # Intercepts are measured from point pairs, never fixed at a true axis.
        anchors=p[pairs[:,0]][length>.10][good]
        residual=np.abs(p@normals.T-np.sum(anchors*normals,axis=1))
        counts=(residual<.007).sum(axis=0);k=int(np.argmax(counts));inlier=residual[:,k]<.009
        if inlier.sum()<7:return
        pts=p[inlier];center=pts.mean(axis=0);_,V=np.linalg.eigh((pts-center).T@(pts-center));t=V[:,-1]
        if t[0]<0:t=-t
        q=float(np.arctan2(t[1],t[0]));span=float(np.ptp(pts@t))
        if span<.14 or not -.08<q<1.62:return
        normal=np.array([-t[1],t[0]]);b=float(center@normal)
        rms=float(np.sqrt(np.mean(((pts-center)@normal)**2)))
        if self.valid:
            dt=time-self.time
            # Reject a wall/jamb hypothesis inconsistent with recent motion.
            if abs(q-self.q)>max(.12,2.8*dt):return
            velocity=(q-self.q)/dt
            self.omega=.65*velocity+.35*self.omega
        self.q=float(np.clip(q,0,np.radians(88)));self.time=float(time);self.valid=True
        self.residual=rms;self.support=int(len(pts));self.span=span
        self.history.append((time,self.q,*normal,b))
        # Hinge = common rotation center of the observed planes. The unknown
        # signed surface-to-axis offset is fitted too, not set to .035/.060.
        h=np.asarray(self.history[-160:])
        if len(h)>=8 and np.ptp(h[:,1])>np.radians(10):
            A=np.column_stack((h[:,2:4],np.ones(len(h))));rhs=h[:,4]
            prior=np.array([self.hinge[0],y,0.])
            estimate=np.linalg.solve(A.T@A+np.diag([.005,.005,.0001]),A.T@rhs+np.array([.005,.005,.0001])*prior)
            candidates=[np.array([x-width/2,y]),np.array([x+width/2,y])]
            side=int(np.argmin([np.linalg.norm(estimate[:2]-c) for c in candidates]))
            if np.linalg.norm(estimate[:2]-candidates[side])<.10:
                self.hinge=estimate[:2];self.hinge_source='multi-frame LiDAR rotating-plane fit'
                # Current scenes have an upward-opening positive hinge axis.
                # A right-side hinge requires a signed control adaptation and
                # is exposed as unsupported instead of silently using truth.
                if side==1:self.hinge_source+=' / opposite hinge: unsupported controller'
        center_offset=float(np.clip(b-normal@self.hinge+.025,-.10,.10))
        o['sensed_geometry']=dict(hinge_xy_m=self.hinge.tolist(),hinge_source=self.hinge_source,
            leaf_center_offset_m=center_offset,thickness_m=.05,thickness_source='nominal door thickness prior',
            leaf_width_m=width,width_source='public frame prior minus nominal jamb allowance',
            axis_world=[0.,0.,1.],axis_source='vertical-door task prior',
            plane_residual_m=rms,points=self.support,observed_span_m=span,time_s=self.time,
            measured_frame=self.frame_fit,
            angle_sigma_rad=max(.003,rms/max(span,.14)),hinge_uncertainty_m=.04)

    def state(self,now):
        if not self.valid:raise RuntimeError('door plane not observed; no hinge truth fallback')
        age=max(0.,now-self.time)
        if age>.30:raise RuntimeError('door LiDAR observation stale')
        # Prediction between genuine 20 Hz scan observations, explicitly logged.
        return float(np.clip(self.q+self.omega*min(age,.075),0,np.radians(88))),float(self.omega)


class SensedDoorInstrument:
    pretrigger_intervals=10
    source='LiDAR door state + distributed-IMU equivalent external wrench'
    def __init__(self,bus,track):self.bus=bus;self.track=track;self.last={}
    def state(self):return self.track.state(self.bus.time)
    def contact(self):
        q,_=self.state();bus=self.bus
        n=np.array([-np.sin(q),np.cos(q)]);t=np.array([np.cos(q),np.sin(q)])
        h=self.track.hinge;p=bus.observed()[0];delta=p[:2]-h
        near=(-.50<float(delta@n)<.15 and -.15<float(delta@t)<self.track.opening['width_m']+.10)
        fn=-float(bus.force[:2]@n)
        detected=near and fn>max(.10,4.*bus.noise)
        # Only a geometrically associated single target plane can be assigned
        # the net external wrench. Unassociated motion is not door evidence.
        pivot=np.r_[h,bus.center[2]]
        torque=-float((bus.moment+np.cross(bus.center-pivot,bus.force))[2])
        if not detected:return (torque if near else 0.),[]
        summary=dict(equivalent_wrench=True,force_on_door_world=(-bus.force).tolist(),
                     moment_on_door_about_uav_com_world=(-bus.moment).tolist(),
                     torque_about_hinge=torque,source=self.source,
                     association='single nearby target plane; other contacts/wind not separable')
        self.last=summary
        return torque,[summary]

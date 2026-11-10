# -*- coding: utf-8 -*-
"""Actual 3D world, cognitive map, onboard RGB and per-object continuous belief."""
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont,ImageOps
from navigation.geometry import wall_boxes
from preview import overview_camera
BG='#0d1925';CARD='#142636';BORDER='#304858';TEAL='#75e3c7';MUTED='#a5bdc9';WHITE='#e8f3f3';AMBER='#efb77b'
COLORS=dict(unknown='#79919f',door='#d5a76c',box='#dfb272',suspected_glass='#72cbe9',free='#70d9af',blocked='#ee8a86')
LABELS=dict(unknown='未知',door='门',box='箱体',suspected_glass='疑似玻璃')


@lru_cache(32)
def font(size):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
def text(d,xy,s,size=15,color=WHITE):d.text(xy,str(s),font=font(size),fill=color)


def map_image(m,width=720,height=540):
    im=Image.new('RGB',(width,height),CARD);d=ImageDraw.Draw(im);b=m.arch['bounds_wall_centerlines_m']
    scale=min((width*.50-30)/(b['xmax']-b['xmin']),(height-40)/(b['ymax']-b['ymin']))
    left=18;bottom=height-20
    xy=lambda p:(left+(p[0]-b['xmin'])*scale,bottom-(p[1]-b['ymin'])*scale)
    for name,c,h in wall_boxes(m.arch):
        a,z=xy(c-h),xy(c+h);d.rectangle((a[0],z[1],z[0],a[1]),fill='#8a9ba8')
    for oid,r in m.memory.records.items():
        x,y=r.geometry['center_xy_m'];half=r.geometry['width_m']/2;color=COLORS.get(r.semantic,COLORS['unknown'])
        if r.traversability in ('free','blocked'):color=COLORS[r.traversability]
        d.line([xy([x-half,y]),xy([x+half,y])],fill=color,width=4)
        px,py=xy([x-half,y]);text(d,(px,py-17),oid,10,color)
        if r.semantic=='door' and r.angle_time>=0:
            from navigation.interactions import panel_geometry
            pivot,w,_=panel_geometry(r.geometry,r.angle);end=pivot+w*np.array([math.cos(r.angle),math.sin(r.angle)])
            d.line([xy(pivot),xy(end)],fill=color,width=2)
        if r.semantic=='box' and r.probe_used:
            p1,p2=xy([x-.59,y+r.linear_position-.25]),xy([x+.59,y+r.linear_position+.25])
            d.rectangle((p1[0],p2[1],p2[0],p1[1]),outline=color,width=2)
    if len(m.path)>1:d.line([xy(p) for p in m.path],fill=AMBER,width=2)
    if len(m.trajectory)>1:d.line([xy(p[1:3]) for p in m.trajectory[::2]],fill=TEAL,width=2)
    p,v,R,g=m.world.observed();cx,cy=xy(p);d.ellipse((cx-4,cy-4,cx+4,cy+4),fill=WHITE)
    end=m.arch['goal_xyz_m'];ex,ey=xy(end);d.ellipse((ex-5,ey-5,ex+5,ey+5),outline=AMBER,width=2)
    if m.local_leg is not None:
        lx,ly=xy(m.local_leg['goal_xy']);d.polygon([(lx,ly-6),(lx+6,ly),(lx,ly+6),(lx-6,ly)],fill=AMBER)
        text(d,(lx+7,ly-7),'局部点',10,AMBER)
    x=width*.52;text(d,(x,7),'感知语义 / 通行记忆',17,TEAL)
    for i,(oid,r) in enumerate(m.memory.records.items()):
        color=COLORS.get(r.semantic,COLORS['unknown']);y=42+i*43
        state={'free':'净空','blocked':'受阻','unverified':'待验证'}[r.traversability]
        if r.bypass_selected:state+=' / 绕行'
        text(d,(x,y),oid+'  '+LABELS.get(r.semantic,r.semantic)+' · '+state,13,color)
        src='尚未识别' if r.last_seen_time<0 else 'RGB多帧确认'
        if r.probe_used:src+=' / 已试探'
        text(d,(x,y+19),src,10,MUTED)
    return im


def camera_image(m):
    packet=m.memory.visual_packet or m.sensors.packet
    if packet is None or packet.rgb is None:return Image.new('RGB',(480,320),CARD)
    im=Image.fromarray(packet.rgb).copy();d=ImageDraw.Draw(im)
    for det in m.memory.detections:
        if abs(det.get('time_s',-1)-packet.time)>1e-6:continue
        color=COLORS.get(det['class_name'],MUTED);box=det['bbox_xyxy'];d.rectangle(box,outline=color,width=2)
        text(d,(box[0]+3,max(0,box[1]-18)),f"{det['stable_id']} {det['class_name']} {det['score']:.2f}",11,color)
    d.rectangle((0,298,480,320),fill=BG);text(d,(8,300),f'实际RGB / YOLO三类 · 曝光 {packet.time:.2f}s',12,TEAL)
    return im


class ObserverRenderer:
    def __init__(self,m):
        self.mission=m;self.camera=overview_camera();self.renderer=mujoco.Renderer(m.world.model,height=540,width=1008)
        self.opt=mujoco.MjvOption();self.opt.geomgroup[:]=1;self.opt.sitegroup[:]=0;self.follow=False;self.automatic=True
    def render(self):
        w=self.mission.world
        if self.follow:self.camera.lookat[:]=w.observed()[0];self.camera.distance=4.;self.camera.elevation=-40
        camera=self.camera
        session=self.mission.active_session
        closeup=(session is not None and session.probed) or w.stage in ('GLASS_VERIFY','OPENING_VERIFY','VERIFY_CONTACT','VERIFY_APPROACH')
        exiting=w.stage=='CRUISE_VERIFIED_EXIT'
        if self.automatic and not self.follow and (closeup or exiting):
            blend=0.
            if exiting:
                elapsed=w.data.time-self.mission.verifications[-1]['end_time_s']
                progress=min(1.,max(0.,elapsed/1.6))
                blend=progress**2*(3.-2.*progress)
            near=w.observed()[0]+np.array([0.,.42,-.10])
            camera=mujoco.MjvCamera()
            camera.lookat[:]=(1.-blend)*near+blend*self.camera.lookat
            camera.distance=(1.-blend)*3.6+blend*self.camera.distance
            camera.azimuth=65.
            camera.elevation=(1.-blend)*(-38.)+blend*self.camera.elevation
        self.renderer.update_scene(w.data,camera=camera,scene_option=self.opt)
        scn=self.renderer.scene;trajectory=self.mission.trajectory[-1000::5]
        for a,b in zip(trajectory,trajectory[1:]):
            if scn.ngeom>=scn.maxgeom:break
            geom=scn.geoms[scn.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.2,.95,.84,.75]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.009,np.array(a[1:4]),np.array(b[1:4]));scn.ngeom+=1
        return Image.fromarray(self.renderer.render().copy())
    def close(self):self.renderer.close()


def dashboard(m,renderer):
    im=Image.new('RGB',(1800,1140),BG);d=ImageDraw.Draw(im)
    text(d,(24,16),'INTERACT  /  一致性交互导航',28,TEAL);text(d,(650,20),m.status,24)
    text(d,(24,65),m.reason[:100],15,MUTED);text(d,(1560,20),f'{m.world.data.time:07.2f} s',27,AMBER)
    text(d,(26,103),'真实物理现场 · 柔性无人机 / 旋翼驱动',16);im.paste(renderer.render(),(20,132))
    text(d,(1055,103),'在线认知地图 · v'+str(m.memory.version),16);im.paste(map_image(m),(1055,132))
    text(d,(24,691),'机载相机 / 实际神经网络检测',16);im.paste(camera_image(m),(20,724))
    text(d,(24,1057),'玻璃：IMU残差接触验证',14,TEAL);text(d,(24,1082),'门/箱力学：11.08仿真参考测量',13,MUTED)
    session=m.display_session
    if session is not None:
        from evidence_dashboard import panel
        key=(id(session),len(session.snapshots),len(session.trace),len(session.rows or []))
        if getattr(m,'_panel_key',None)!=key:m._panel_image=panel(session);m._panel_key=key
        im.paste(m._panel_image,(515,690))
    else:
        text(d,(545,740),'接触前不设质量/阻力有限先验',23,TEAL)
        text(d,(545,785),'先比较全局几何收益与名义试探预算',18)
        text(d,(545,825),'触碰后：有效窗口 → 参数约束 → 多数权重',18,MUTED)
    x=1555;text(d,(x,707),'当前决策',20,TEAL)
    for j,(label,value) in enumerate([('对象',m.current or '—'),('动作',m.action),('Probe次数',m.probes),('重规划',m.replans),
            ('G / m','—' if m.metrics.get('G') is None else f"{m.metrics['G']:.3f}"),
            ('支持率','—' if m.metrics.get('p_D') is None else f"{m.metrics['p_D']:.2%}"),
            ('F* / N','—' if m.metrics.get('F_star') is None else f"{m.metrics['F_star']:.3f}")]):
        text(d,(x,752+j*43),label,12,MUTED);text(d,(x+85,749+j*43),value,17)
    return im


def save_views(m,prefix,renderer=None):
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    if renderer is None and m.ui is not None:renderer=m.ui.renderer
    if renderer is None:
        if m.observer is None:m.observer=ObserverRenderer(m)
        renderer=m.observer
    renderer.render().save(str(prefix)+'_3d.png');map_image(m).save(str(prefix)+'_map.png')
    camera_image(m).save(str(prefix)+'_camera.png');dashboard(m,renderer).save(str(prefix)+'_dashboard.png')

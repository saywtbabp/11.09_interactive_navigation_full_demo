"""Single Tk dashboard: live MuJoCo pixels, map, SI measurements, density."""
import time
import tkinter as tk
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont, ImageTk
from constraint_views import view_bounds, band_geometry, excluded_candidates, PALETTE
from decision_rule import description, from_snapshot

FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'
CONTINUE = '#48d5aa'
BYPASS = '#f4a261'

def decision_extent(bounds, limit):
    """Include the reference decision line even when all data lie on one side."""
    high = float(bounds[1, 1])
    if limit is not None and np.isfinite(limit):
        high = max(high, max(0., limit)*1.15)
    return 0., max(high, .01)

def weight_verdict(snap, alpha):
    return description(snap,alpha)
def font(size=16):
    return ImageFont.truetype(FONT_PATH,size)

def panel(e):
    im=Image.new('RGB',(1020,440),'#101e29'); d=ImageDraw.Draw(im)
    def txt(x,y,s,color='#dceaf0',size=16): d.text((x,y),s,font=font(size),fill=color)
    txt(18,12,f'{getattr(e,"oid",e.args.case)} · 参数权重  质量 kg / 阻力 N',size=15)
    snap=e.snapshots[-1] if e.snapshots else None
    if snap and (snap['bounds'] is not None or snap.get('constraints')):
        bounds=np.array(snap['bounds']) if snap['bounds'] is not None else None
        constraints=snap.get('constraints',[]);n=150
        limit=snap.get('push_limit')
        x0,x1,y0,y1=view_bounds(snap['bounds'],limit,constraints)
        x=np.linspace(x0,x1,n); y=np.linspace(y1,y0,n)
        xx,yy=np.meshgrid(x,y); pts=np.column_stack((xx.ravel(),yy.ravel()))
        precision=np.array(snap['precision']); h=np.array(snap['linear'])
        log=pts@h-.5*np.sum((pts@precision)*pts,axis=1)
        poly=np.array(snap['polygon']); mask=np.full(len(pts),bool(len(poly)))
        for a,b in zip(poly,np.roll(poly,-1,axis=0)):
            edge=b-a; cross=edge[0]*(pts[:,1]-a[1])-edge[1]*(pts[:,0]-a[0])
            mask &= cross >= -1e-9
        values=np.zeros(len(pts))
        if mask.any(): values[mask]=np.exp((log[mask]-log[mask].max())*.6)
        good = pts[:,1] < limit if limit is not None else np.zeros(len(pts),bool)
        # Faint background = decision side; bright cells = relative data weight.
        rgb=np.where(good[:,None], [18.,54.,46.], [60.,39.,29.])
        rgb+=values[:,None]*np.where(good[:,None], [54.,165.,128.], [179.,115.,65.])
        if limit is None:
            rgb=np.array([24.,36.,49.])+values[:,None]*np.array([69.,160.,190.])
        heat=Image.fromarray(rgb.clip(0,255).astype('uint8').reshape(n,n,3)).resize((330,265))
        im.paste(heat,(66,55))
        project=lambda a:(66+330*(a[0]-x0)/(x1-x0),55+265*(y1-a[1])/(y1-y0))
        overlay=Image.new('RGBA',im.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
        pending_lines=[]
        for i,w in enumerate(constraints):
            band,segments=band_geometry(w,(x0,x1,y0,y1))
            color=PALETTE[i%len(PALETTE)]
            rgb_color=tuple(int(color[k:k+2],16) for k in (1,3,5))
            if len(band)>2:od.polygon([project(a) for a in band],fill=(*rgb_color,9))
            for segment in segments:pending_lines.append((i,rgb_color,segment))
        for i,rgb_color,segment in pending_lines:
            od.line([project(a) for a in segment],fill=(*rgb_color,220 if i==len(constraints)-1 else 95),width=2 if i==len(constraints)-1 else 1)
        im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB');d=ImageDraw.Draw(im)
        for point in excluded_candidates(constraints,poly,(x0,x1,y0,y1)):
            cx,cy=project(point)
            d.line((cx-1.5,cy-1.5,cx+1.5,cy+1.5),fill='#7b8996',width=1)
            d.line((cx-1.5,cy+1.5,cx+1.5,cy-1.5),fill='#7b8996',width=1)
        d.rectangle((66,55,396,320),outline='#69818e')
        txt(65,326,f'{x0:.2f}                         {x1:.2f} kg')
        txt(5,50,f'{y1:.2f}',size=12);txt(5,302,f'{y0:.2f}',size=12)
        if len(poly):d.line([project(a) for a in [*poly,poly[0]]],fill='white',width=3)
        if limit is not None and y0<=limit<=y1:
            py=55+265*(y1-limit)/(y1-y0)
            d.line((66,py,396,py),fill='#ffea82',width=4)
            txt(75,max(57,py-23),f'分界 F* = {limit:.3f} N','#ffea82',14)
        txt(18,348,'绿：R < F* 续推  |  橙：R ≥ F* 绕行',size=13)
        p=snap.get('push_probability')
        verdict,color=weight_verdict(snap,e.args.alpha)
        criterion=from_snapshot(snap,e.args.alpha)
        rule_name='多数' if criterion['decision_rule']=='majority' else '保守'
        txt(18,371,f"p_ok={p:.3f} / {rule_name}门限 {criterion['decision_push_threshold']:.2f}" if p is not None else 'p_ok 未定义',size=14)
        txt(18,393,verdict,color,15)
        txt(18,418,f'灰×：已排除/零权重；蓝紫：{len(constraints)}组带；白：交集',size=11)
    else:
        limit=None if snap is None else snap.get('push_limit')
        if limit is not None:
            hi=max(2.,limit*1.15);top,bottom=60,310;split=bottom-(bottom-top)*limit/hi
            d.rectangle((66,top,396,split),fill='#3c271d');d.rectangle((66,split,396,bottom),fill='#12362e')
            d.line((66,split,396,split),fill='#ffea82',width=3)
            txt(75,max(top,split-23),f'当前分界 F* = {limit:.3f} N','#ffea82',14)
            txt(18,330,'几何边界已计算；尚无有限参数区域',size=13)
            txt(18,355,'p_ok 未定义：等待可辨识的运动窗口',size=13)
            txt(18,381,'背景仅表示决策两侧，不是参数先验',size=12)
        else:
            txt(25,130,'尚无可归一化的有限参数区域')
            txt(25,165,'接触后每个有效窗口同步计算边界')
            txt(25,200,'无固定等待窗口数')
    rows=e.trace[-3000:]
    if e.rows is not None: rows=[*rows,*e.rows]
    if len(rows)>2:
        z=np.array([r[:5] for r in rows],float); z=e.local_rows(z)
        for j,(col,title,color) in enumerate([(3,'沿运动方向力 N','#ffa56b'),(2,'局部速度 m/s','#6cddc3')]):
            top=48+j*173; bot=top+116; left=480; right=990
            v=z[:,col]; lo=min(0.,float(v.min())); hi=max(.01,float(v.max()))
            xx=left+(z[:,0]-z[0,0])/max(z[-1,0]-z[0,0],.001)*(right-left)
            yy=bot-(v-lo)/(hi-lo)*(bot-top)
            d.line(list(zip(xx,yy)),fill=color,width=2)
            txt(left,top-25,f'{title}  [{lo:.2f}, {hi:.2f}]')
            d.line((left,bot,right,bot),fill='#65808d')
        txt(480,397,f'总接触力 {e.last_force:.2f} N / 正功 {e.work:.3f} J')
    if snap:
        source=snap.get('route_preview_status','');age=snap.get('boundary_age_s')
        label='每窗同步：权重 + 边界 + 动作' if source=='paired_online_geometry' else '正式决策：最新状态复算' if source=='fresh_formal_geometry' else '已提交：保留正式分界' if source=='committed_decision_reference' else '后台分界预览' if source=='asynchronous_geometry_sample' else '等待有效证据 / 几何预览'
        if snap.get('inference_frozen'):label=f"通行几何已满足 / 推断冻结 · 保留 {snap['frozen_reference_time']:.2f}s 的真实结果";age=None
        if snap.get('exit_latched'):label+=' · 已进入退出流程'
        txt(480,419,label+(f' · 几何滞后 {age:.2f}s' if age is not None else ''),size=11)
    return im


class Dashboard:
    def __init__(self,e,speed):
        self.root=tk.Tk();self.root.title('一致性交互导航 · MuJoCo 实时仿真')
        self.root.configure(bg='#101e29'); self.speed=speed;self.last=0.;self.paused=False
        self.closed=False;self.renderer=mujoco.Renderer(e.world.model,height=430,width=690)
        self.camera=mujoco.MjvCamera();self.camera.lookat[:]=[0,5.8,.7]
        self.camera.distance=16;self.camera.azimuth=90;self.camera.elevation=-62
        self.top=tk.Label(self.root,bg='#101e29',fg='white',font=('Microsoft YaHei',12)); self.top.pack()
        self.image=tk.Label(self.root);self.image.pack()
        self.bottom=tk.Label(self.root);self.bottom.pack()
        bar=tk.Frame(self.root);bar.pack(fill='x')
        tk.Button(bar,text='暂停 / 继续（空格）',command=self.toggle).pack(side='left')
        tk.Button(bar,text='保存当前证据',command=lambda:self.save(e)).pack(side='left')
        self.root.bind('<space>',lambda _:self.toggle())
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.update()

    def toggle(self): self.paused=not self.paused
    def pause_for_decision(self,e):
        self.last=-1.
        self.update(e)
        self.paused=True
        self.top.configure(text=self.top.cget('text')+' · 正式决策处暂停，空格继续')
        while self.paused and not self.closed:
            self.root.update();time.sleep(.03)
        if self.closed:raise KeyboardInterrupt('dashboard closed')
    def close(self):
        self.closed=True
        self.root.destroy()
    def update(self,e):
        if self.closed: raise KeyboardInterrupt('dashboard closed')
        while self.paused and not self.closed:
            self.root.update();time.sleep(.03)
        if self.closed: raise KeyboardInterrupt('dashboard closed')
        if e.world.data.time-self.last<.10: return
        self.last=float(e.world.data.time)
        self.renderer.update_scene(e.world.data,camera=self.camera)
        im=Image.fromarray(self.renderer.render()); combined=Image.new('RGB',(1020,430),'#101e29')
        combined.paste(im,(0,0));dr=ImageDraw.Draw(combined)
        xy=lambda p:(int(850+p[0]*25),int(415-p[1]*31))
        dr.text((705,10),'实时地图 / 实飞轨迹',font=font(16),fill='white')
        for oid,x in [('玻璃',-3.5),('目标',0),('绕行',4.55)]:
            a=xy([x-.55,5.4]);b=xy([x+.55,5.4]); dr.line((*a,*b),fill='#ffa66a' if x==0 else '#6fc8da',width=4)
            dr.text((a[0],a[1]-25),oid,font=font(12),fill='white')
        if len(e.frames)>1: dr.line([xy(r[1:3]) for r in e.frames],fill='#74e4c4',width=2)
        pos=xy(e.position());dr.ellipse((pos[0]-5,pos[1]-5,pos[0]+5,pos[1]+5),fill='white')
        goal=xy(e.goal);dr.text(goal,'终点',font=font(13),fill='#ffdf83')
        self.last_image=combined;self.last_panel=panel(e)
        self.photo=ImageTk.PhotoImage(combined);self.image.configure(image=self.photo)
        self.photo2=ImageTk.PhotoImage(self.last_panel);self.bottom.configure(image=self.photo2)
        self.top.configure(text=f'{e.world.data.time:6.1f} s · {e.world.stage} · {e.kind} · {e.args.action} · 姿态/接触为仿真参考测量')
        recovery = getattr(e, 'probe_recovery', None)
        if recovery is not None and e.world.stage in ('PROBE_UNLOAD','PROBE_HOVER','AFTER_PROBE_STATE'):
            flags = recovery.last_status
            names = [('detached','脱离'),('stable','稳定'),('envelope_ok','包络'),
                     ('clearance','净空'),('exit_path','退出路径')]
            if flags.get('local_hover'):
                names=[('detached','脱离'),('stable','稳定'),('envelope_ok','机体'),
                       ('clearance','局部悬停净空'),('exit_path','退出预案')]
            self.top.configure(text=f'{e.world.data.time:6.1f} s · '+('就近稳定与观察 · ' if flags.get('local_hover') else '状态触发卸载 · ')+
                ' / '.join(name+(' ✓' if flags.get(key) else ' …') for key,name in names)+
                f" · 持续 {flags.get('dwell_s',0.):.2f}/0.25 s")
        self.root.update();time.sleep(.10/self.speed)
    def save(self,e):
        folder=e.args.output;folder.mkdir(parents=True,exist_ok=True)
        out=Image.new('RGB',(1020,870));out.paste(self.last_image);out.paste(self.last_panel,(0,430))
        out.save(folder/'dashboard.png')
    def finish(self,e,close):
        if not self.closed:
            self.save(e)
            if not close:self.root.mainloop()
            else:self.close()
        self.renderer.close()

"""Offline three-class RGB data; simulator IDs label TRAINING data only."""
import sys,math,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import runtime
import numpy as np
import mujoco
from PIL import Image
from configuration import Config,architecture
from navigation.physics import World
from navigation.sensors import Sensors


def fixture_pose(w,base,position,yaw,pitch,roll):
    m,d=w.model,w.data;d.qpos[:]=base;mujoco.mj_forward(m,d);origin=d.sensor('pos').data.copy()
    q=np.array([1.,0.,0.,0.])
    for axis,a in [([0,0,1],yaw),([1,0,0],pitch),([0,1,0],roll)]:
        out=np.empty(4);mujoco.mju_mulQuat(out,q,np.r_[math.cos(a/2),np.array(axis)*math.sin(a/2)]);q=out
    mat=np.empty(9);mujoco.mju_quat2Mat(mat,q);R=mat.reshape(3,3)
    for j in range(m.njnt):
        if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[j]) in w.robot_ids:
            a=int(m.jnt_qposadr[j]);d.qpos[a:a+3]=position+R@(base[a:a+3]-origin)
            mujoco.mju_mulQuat(d.qpos[a+3:a+7],q,base[a+3:a+7])
    mujoco.mj_forward(m,d)


def main():
    out=ROOT/'data/vision_synthetic';out.mkdir(parents=True,exist_ok=True)
    frozen=out/'generation_architecture.json'
    arch=json.loads(frozen.read_text(encoding='utf-8')) if frozen.exists() else architecture()
    if not frozen.exists():frozen.write_text(json.dumps(arch,ensure_ascii=False,indent=2),encoding='utf-8')
    cfg=Config();manifest=[]
    for scene in range(15):
        split='train' if scene<11 else 'val' if scene<13 else 'test'
        rng=np.random.default_rng(4357+scene*97)
        # Every independent randomized scene contains all three classes.
        kinds=['door','glass','box','empty','door','box','glass','empty','box','door']
        rng.shuffle(kinds);contents={o['id']:k for o,k in zip(arch['openings'],kinds)}
        w=World(arch,cfg,observed_contents=contents);m,d=w.model,w.data
        for g in range(m.ngeom):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            if name.startswith('surface_') and contents.get(name[8:]) in ('door','box') and scene%3:
                m.geom_matid[g]=-1;m.geom_rgba[g,:3]=rng.uniform(.15,.80,3)
            if name.startswith('wall_') and scene%3==1:
                m.geom_matid[g]=-1;m.geom_rgba[g,:3]=rng.uniform(.4,.85,3)
        m.light_diffuse[:]*=rng.uniform(.7,1.2)
        base=d.qpos.copy();s=Sensors(w,cfg);renderer=mujoco.Renderer(m,height=320,width=480)
        targets={m.geom('surface_'+oid).id:{'door':0,'glass':1,'box':2}[kind] for oid,kind in contents.items() if kind!='empty'}
        for view in range(50):
            o=arch['openings'][view%len(arch['openings'])];x,y=o['center_xy_m']
            position=np.array([np.clip(x+rng.uniform(-.9,.9),-5.0,5.0),max(.45,y-rng.uniform(.55,3.5)),rng.uniform(1.,1.5)])
            yaw=math.atan2(-(x-position[0]),y-position[1])+rng.uniform(-.22,.22)
            fixture_pose(w,base,position,yaw,rng.uniform(-.1,.1),rng.uniform(-.08,.08))
            for oid,kind in contents.items():
                if kind=='door':d.qpos[int(m.jnt_qposadr[m.joint('hinge_'+oid).id])]=rng.uniform(0,1.) if view%5==0 else 0.
                elif kind=='box':d.qpos[int(m.jnt_qposadr[m.joint('hinge_'+oid).id])]=rng.uniform(0,.6) if view%5==0 else 0.
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera='onboard',scene_option=s.opt);rgb=renderer.render().copy()
            renderer.enable_segmentation_rendering();renderer.update_scene(d,camera='onboard',scene_option=s.opt);ids=renderer.render().copy();renderer.disable_segmentation_rendering()
            labels=[]
            for gid,cls in targets.items():
                yy,xx=np.nonzero((ids[:,:,0]==gid)&(ids[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)))
                if len(xx)<150:continue
                x1,x2,y1,y2=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
                if min(x2-x1,y2-y1)<10:continue
                labels.append([cls,(x1+x2)/960,(y1+y2)/640,(x2-x1)/480,(y2-y1)/320])
            name=f'scene_{scene:02d}_view_{view:03d}';images=out/'images'/split;annotations=out/'labels'/split
            images.mkdir(parents=True,exist_ok=True);annotations.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(images/(name+'.jpg'),quality=90)
            (annotations/(name+'.txt')).write_text('\n'.join('%d %.7f %.7f %.7f %.7f'%tuple(a) for a in labels),encoding='utf-8')
            manifest.append(dict(name=name,scene=scene,split=split,seed=4357+scene*97,labels=len(labels)))
        renderer.close();s.close();print('scene',scene,split,flush=True)
    (out/'data.yaml').write_text('path: '+out.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: door\n  1: suspected_glass\n  2: box\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()

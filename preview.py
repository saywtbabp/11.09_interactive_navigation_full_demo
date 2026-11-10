"""Render actual MuJoCo state, never a concept image."""
import runtime
import mujoco
from PIL import Image
from configuration import ROOT,Config,architecture
from navigation.physics import World

def overview_camera():
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=[0,9.0,.25];cam.distance=27.;cam.azimuth=65;cam.elevation=-62
    return cam

def run():
    import json
    from functools import partial
    from scene.build import build_xml
    spec=json.loads((ROOT/'config/scene.json').read_text(encoding='utf-8'))
    w=World(architecture(),Config(),door_parameters=spec['door_parameters'],observed_contents=spec['contents'],
            scene_builder=partial(build_xml,box_parameters=spec['box_parameters']));w.hold(w.target,.8)
    renderer=mujoco.Renderer(w.model,height=1080,width=1440)
    opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.geomgroup[:]=1
    renderer.update_scene(w.data,camera=overview_camera(),scene_option=opt)
    Image.fromarray(renderer.render()).save(ROOT/'results'/'scene_preview.png')
    opt.geomgroup[2]=0;opt.geomgroup[5]=0;opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=0
    renderer.update_scene(w.data,camera='onboard',scene_option=opt)
    Image.fromarray(renderer.render()).save(ROOT/'results'/'onboard_preview.png')
    renderer.close();print(ROOT/'results'/'scene_preview.png')

if __name__=='__main__':run()

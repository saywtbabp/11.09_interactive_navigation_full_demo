"""One physical building from the public metric architecture. Truth stays here."""
import xml.etree.ElementTree as ET
import math
import runtime
import numpy as np
from PIL import Image, ImageFilter
from configuration import ROOT
from navigation.geometry import wall_boxes

def textures():
    folder=ROOT/'assets'/'textures'; folder.mkdir(exist_ok=True)
    if all((folder/(n+'.png')).exists() for n in ('concrete','plaster','walnut')): return
    rng=np.random.default_rng(19); n=768
    def noise(sz,scale):
        im=Image.fromarray(rng.integers(0,256,(sz,sz),dtype=np.uint8))
        return (np.asarray(im.resize((n,n),Image.Resampling.BICUBIC),float)-128)*scale
    field=noise(12,.065)+noise(60,.055)+noise(n,.025)
    for name,base in [('concrete',[139,146,149]),('plaster',[219,216,202])]:
        pixels=np.array(base)[None,None,:]+field[:,:,None]
        if name=='concrete':
            yy,xx=np.indices((n,n)); seam=(xx<2)|(yy<2)
            pixels[seam]-=18
        Image.fromarray(np.uint8(np.clip(pixels,0,255))).save(folder/(name+'.png'))
    yy,xx=np.indices((n,n)); warp=6*np.sin(yy/90)+3*np.sin(yy/23)
    grain=7*np.sin((xx+warp)/3.3)+4*np.sin((xx+warp)/1.2)+noise(16,.12)+noise(n,.025)
    wood=np.array([112,68,38])[None,None,:]+grain[:,:,None]*np.array([1,.76,.5])
    Image.fromarray(np.uint8(np.clip(wood,0,255))).save(folder/'walnut.png')

def build_xml(arch,cfg,variant='standard',door_parameters=None,visual=True,observed_contents=None,box_parameters=None):
    textures()
    root=ET.parse(ROOT/'assets'/'models'/'tensegrity_real_model.xml').getroot()
    root.set('model','INTERACT | Full glass-door-box mission 11 x 18.4 m')
    option=root.find('option'); option.set('timestep',str(cfg.timestep));option.set('iterations','80')
    ET.SubElement(root,'compiler',angle='degree',autolimits='true')
    wb=root.find('worldbody')
    for light in list(wb.findall('light')):wb.remove(light)
    # Original robot physics is untouched; recolouring and massless skins are display only.
    for geom in wb.findall('.//geom'):
        geom.set('group','2')
        name=geom.get('name','')
        if name.startswith('link'): geom.set('rgba','0.13 0.29 0.35 1')
        elif name.startswith('capsule'):geom.set('rgba','0.86 0.45 0.14 1')
        else:geom.set('rgba','0.17 0.21 0.25 1')
    for site in wb.findall('.//site'):site.set('group','4')
    for tendon in root.findall('tendon/spatial'):tendon.set('rgba','0.45 0.52 0.57 1')
    frame=wb.find("body[@name='quad_frame']")
    ET.SubElement(frame,'camera',name='onboard',pos='0 0.015 0.055',xyaxes='1 0 0 0 0 1',fovy=str(cfg.camera_fovy))
    visual_node=ET.SubElement(root,'visual')
    ET.SubElement(visual_node,'global',offwidth='1920',offheight='1200',azimuth='90',elevation='-63')
    ET.SubElement(visual_node,'quality',shadowsize='4096',offsamples='4')
    ET.SubElement(visual_node,'headlight',ambient='.30 .32 .36',diffuse='.32 .34 .38',specular='.15 .15 .15')
    ET.SubElement(visual_node,'rgba',haze='.04 .065 .10 1')
    ET.SubElement(visual_node,'map',znear='.005',zfar='80',fogstart='30',fogend='70',shadowclip='1.5',shadowscale='.7')
    ET.SubElement(root,'statistic',center='0 6.5 0',extent='16')
    asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.11 .16 .23',rgb2='.025 .04 .065',width='512',height='3072')
    for name,repeat,spec,shine in [('concrete','1 1','.20','.16'),('plaster','1 1','.12','.08'),('walnut','1 1','.32','.3')]:
        ET.SubElement(asset,'texture',name=name,type='2d',file=(ROOT/'assets'/'textures'/(name+'.png')).as_posix())
        attrs=dict(name=name,texrepeat=repeat,texuniform='true' if name!='walnut' else 'false',specular=spec,shininess=shine)
        if name=='plaster':attrs['rgba']='.71 .69 .62 1'
        else:attrs.update(texture=name,rgba='.60 .65 .69 1' if name=='concrete' else '.85 .85 .85 1')
        ET.SubElement(asset,'material',**attrs)
    for name,rgba,spec,shine in [('metal','.13 .17 .20 1','.7','.7'),('brass','.53 .39 .20 1','.55','.5'),
                                  ('glass','.39 .66 .71 .23','.9','.9'),('edge','.25 .49 .52 .7','.8','.6')]:
        ET.SubElement(asset,'material',name=name,rgba=rgba,specular=spec,shininess=shine)
    ET.SubElement(asset,'material',name='cyan',rgba='.11 .78 .86 1',emission='.65')
    def box(parent,name,pos,size,material=None,rgba=None,collide=True,group='0',**kw):
        args=dict(name=name,type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),
                  group=group,friction='.3 .005 .0001',solref='.012 1',solimp='.95 .99 .001')
        if material:args['material']=material
        if rgba:args['rgba']=rgba
        if not collide:args.update(contype='0',conaffinity='0',mass='0')
        args.update(kw);return ET.SubElement(parent,'geom',**args)
    ET.SubElement(wb,'light',name='sun',pos='-4 -4 12',dir='.30 .40 -1',directional='true',diffuse='.83 .77 .64',specular='.40 .36 .30',castshadow='true')
    ET.SubElement(wb,'light',name='fill',pos='6 7 10',dir='-.5 0 -1',directional='true',diffuse='.40 .49 .58',castshadow='false')
    ET.SubElement(wb,'light',name='rim',pos='-6 12 8',dir='.5 -.3 -1',diffuse='.35 .39 .43',castshadow='false')
    bounds=arch['bounds_wall_centerlines_m'];middle=(bounds['ymin']+bounds['ymax'])/2;half_y=(bounds['ymax']-bounds['ymin'])/2
    box(wb,'ground',[0,middle,-.035],[5.56,half_y+.06,.035],'concrete')
    box(wb,'plinth',[0,middle,-.22],[5.70,half_y+.20,.17],'metal',collide=False,group='5')
    box(wb,'foundation',[0,middle,-.45],[5.82,half_y+.32,.07],rgba='.08 .11 .15 1',collide=False,group='5')
    H=arch['wall_height_m']
    for name,c,h in wall_boxes(arch):
        box(wb,'wall_'+name,[*c,H/2],[*h,H/2],'plaster')
        if visual:
            box(wb,'cap_'+name,[*c,H+.018],[h[0]+.012,h[1]+.012,.018],'metal',collide=False,group='3')
            box(wb,'base_'+name,[*c,.055],[h[0]+.006,h[1]+.006,.055],'metal',collide=False,group='3')
            # Wall cap light is an observer layer; no class or dynamics encoded.
            box(wb,'light_'+name,[*c,H+.039],[max(.005,h[0]-.02),.009,.004],'cyan',collide=False,group='5')
    # Hidden assignments are supplied exclusively to the physical simulator.
    contents={'O01':'door','O03':'glass','O05':'door'}
    if variant=='glass_removed':contents.pop('O03')
    if variant=='swapped':contents={'O01':'glass','O03':'door','O05':'door'}
    if variant=='all_empty':contents={}
    if observed_contents is not None:contents=dict(observed_contents)
    dynamics=door_parameters or {'O01':(.071,.035),'O05':(.437,2.473)}
    truth={};width=1.1825; thick=.05
    for opening in arch['openings']:
        oid=opening['id'];x,y=opening['center_xy_m'];w=opening['width_m']; kind=contents.get(oid,'empty')
        truth[oid]=dict(kind=kind)
        # Identical neutral jambs at all openings. No labels enter the sensor view.
        for sign in (-1,1):
            # Rounded metal jambs retain the nominal .09 m width and centreline
            # net clearance, with real hinge corner relief rather than overlap.
            ET.SubElement(wb,'geom',name='jamb_'+oid+str(sign),type='cylinder',
                          pos=f'{x+sign*(w/2-.045)} {y} 1.04',size='.042 1.04',
                          material='metal',friction='.3 .005 .0001',solref='.012 1',group='0')
            if visual:
                ET.SubElement(wb,'geom',name='jamb_skin_'+oid+str(sign),type='cylinder',
                              pos=f'{x+sign*(w/2-.045)} {y} 1.04',size='.045 1.04',material='metal',
                              contype='0',conaffinity='0',mass='0',group='3')
        box(wb,'lintel_'+oid,[x,y,2.08],[w/2,.065,.04],'metal')
        if kind=='door':
            I,tau=dynamics.get(oid,(.071,.035));mass=I/(width**2/3+thick**2/12+.035**2)
            # Rear-offset real hinge axis clears the leaf thickness at the jamb.
            # A centre-plane hinge with zero side gap would lock against its own frame.
            hinge=[x-width/2,y+.035,0]
            body=ET.SubElement(wb,'body',name='panel_'+oid,pos=' '.join(map(str,hinge)))
            ET.SubElement(body,'joint',name='hinge_'+oid,type='hinge',axis='0 0 1',range='0 88',damping=str(cfg.hinge_damping),frictionloss=str(tau))
            inertia=[mass*(thick**2+2**2)/12,mass*(width**2+2**2)/12,mass*(width**2+thick**2)/12]
            ET.SubElement(body,'inertial',pos=f'{width/2} -.035 1.02',mass=str(mass),diaginertia=' '.join(map(str,inertia)))
            box(body,'surface_'+oid,[width/2,-.035,1.02],[width/2,thick/2,1],'walnut')
            if visual:
                for z in (.15,1.85):
                    box(body,'inlay_'+oid+str(z),[width/2,-.061,z],[width*.38,.002,.004],'brass',collide=False,group='3')
                for z in (.3,1.,1.7):
                    ET.SubElement(body,'geom',type='cylinder',pos=f'0 0 {z}',size='.026 .06',material='brass',mass='0',contype='0',conaffinity='0',group='3')
                box(body,'handle_'+oid,[width-.15,-.100,1],[.075,.02,.016],'brass',collide=False,group='3')
            truth[oid].update(inertia=I,friction=tau,width=width)
        elif kind=='box':
            spec=box_parameters or {}
            mass,friction=spec.get(oid,(1.1,.62))
            body=ET.SubElement(wb,'body',name='panel_'+oid,pos=f'{x} {y} 0')
            ET.SubElement(body,'joint',name='hinge_'+oid,type='slide',axis='0 1 0',range='0 2',damping='.05',frictionloss=str(friction))
            box(body,'surface_'+oid,[0,0,.9],[width/2,.25,.9],'walnut',mass=str(mass))
            if visual:
                box(body,'box_tape_'+oid,[0,-.252,.9],[.035,.002,.88],rgba='.68 .59 .38 1',collide=False,group='3')
                box(body,'box_band_'+oid,[0,-.253,1.4],[width/2,.002,.028],rgba='.66 .56 .35 1',collide=False,group='3')
            truth[oid].update(mass=mass,friction=friction,width=width,depth=.5,height=1.8)
        elif kind=='glass':
            box(wb,'surface_'+oid,[x,y,1.01],[w/2-.09,.018,.99],'glass',group='1')
            # Thin edge and quiet reflection strips give actual visual evidence.
            for offset in (-w/2+.10,w/2-.10):
                box(wb,'glass_edge_'+oid+str(offset),[x+offset,y-.020,1.01],[.009,.002,.97],'edge',collide=False,group='3')
            for z in (.07,1.96):box(wb,'rail_'+oid+str(z),[x,y,z],[w/2-.09,.03,.018],'metal')
            if visual:
                for k,offset in enumerate((-.30,.32)):
                    box(wb,'reflection_'+oid+str(k),[x+offset,y-.019,1.12],[.026,.001,.62],rgba='.72 .89 .91 .19',collide=False,group='3')
    # Start/goal halos are absent from onboard RGB and all scans.
    for label,point,color in [('launch',arch['start_xyz_m'],'.15 .90 .72 .8'),('target',arch['goal_xyz_m'],'.95 .46 .20 .9')]:
        ET.SubElement(wb,'geom',name=label,type='cylinder',pos=f'{point[0]} {point[1]} .004',size='.28 .003',rgba=color,contype='0',conaffinity='0',mass='0',group='5')
    return ET.tostring(root,encoding='unicode'),truth

"""Build a self-contained release zip without pilot runs or browser profiles."""
import argparse,json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def include(p):
    rel=p.relative_to(ROOT);parts=rel.parts
    if any(x in ('__pycache__','.git','.idea','.matplotlib','.ultralytics') for x in parts):return False
    if p.suffix in ('.pyc','.pyo'):return False
    if len(parts)==1:return p.suffix in ('.py','.md','.txt','.cmd','.json')
    if parts[0] in ('assets','config','navigation','scene','ui','models','.vendor'):return True
    if parts[0]=='tools':return p.name!='camera_layout.py'
    if parts[0]=='data':return p.name in ('manifest.json','data.yaml','generation_architecture.json')
    if parts[0]=='results':
        if len(parts)>1 and parts[1] in ('release','navigation_update_verified','online_verified','online_door_pass','local_navigation_final'):
            if 'browser_qa' in parts:return p.suffix in ('.png','.json') and not any(x.startswith('profile_') for x in parts)
            return True
        return rel.as_posix() in ('results/vision_training/test_metrics.json',
            'results/vision_training/three_classes/results.csv','results/vision_training/three_classes/args.yaml')
    return False

def package(path):
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='DELIVERY_MANIFEST.json' and include(p))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest=dict(project=ROOT.name,files=hashes,exclusions='pilot runs, synthetic training images, browser profiles, Python caches')
    (ROOT/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if ROOT/'DELIVERY_MANIFEST.json' not in files:files.append(ROOT/'DELIVERY_MANIFEST.json')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.zip.sha256').write_text(digest+'  '+path.name+'\n',encoding='ascii')
    print(json.dumps(dict(zip=str(path),files=len(files),bytes=path.stat().st_size,sha256=digest)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('zip',type=Path);package(p.parse_args().zip)

"""Optional offline 3-class training; main.py only needs the exported ONNX."""
import argparse,os,sys,json,hashlib,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--deps-root',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--export-only',action='store_true')
    a=p.parse_args()
    sys.path[:0]=[str(a.deps_root/('.training_deps' if a.export_only else '.training_deps38')),
                  str(a.deps_root/('.training_deps38' if a.export_only else '.training_deps'))]
    os.environ.update(YOLO_CONFIG_DIR=str(ROOT/'.ultralytics'),MPLCONFIGDIR=str(ROOT/'.matplotlib'),
        YOLO_AUTOINSTALL='false',MKL_THREADING_LAYER='SEQUENTIAL',OMP_NUM_THREADS='4',
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',WANDB_MODE='disabled')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    best=ROOT/'results/vision_training/three_classes/weights/best.pt'
    if not a.export_only:
        model=YOLO(str(ROOT/'models/vision/door_glass_yolov8n.pt'))
        model.train(data=str(ROOT/'data/vision_synthetic/data.yaml'),epochs=a.epochs,imgsz=320,batch=24,
            device=0 if torch.cuda.is_available() else 'cpu',workers=0,seed=1953,pretrained=True,
            project=str(ROOT/'results/vision_training'),name='three_classes',exist_ok=True,
            patience=8,amp=False,plots=False,mosaic=.5,close_mosaic=5,hsv_h=.08,hsv_s=.6,hsv_v=.35)
        trained=YOLO(str(best));metrics=trained.val(data=str(ROOT/'data/vision_synthetic/data.yaml'),
            split='test',imgsz=320,device=0 if torch.cuda.is_available() else 'cpu',workers=0,plots=False)
        (ROOT/'results/vision_training/test_metrics.json').write_text(json.dumps({k:float(v) for k,v in metrics.results_dict.items()},indent=2),encoding='utf-8')
        subprocess.run(['G:/anaconda3/envs/py39/python.exe',__file__,'--deps-root',str(a.deps_root),
                        '--epochs',str(a.epochs),'--export-only'],check=True)
    else:
        import onnx
        exported=YOLO(str(best)).export(format='onnx',imgsz=320,opset=12,simplify=False,dynamic=False,device='cpu')
        dest=ROOT/'models/vision/door_glass_box_yolov8n.onnx'
        shutil.copy2(exported,dest);shutil.copy2(best,dest.with_suffix('.pt'))
        onnx.checker.check_model(onnx.load(str(dest)))
        metadata=dict(architecture='YOLOv8n',classes={'0':'door','1':'suspected_glass','2':'box'},epochs=a.epochs,
            onnx_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),input_size=320,
            test_metrics=json.loads((ROOT/'results/vision_training/test_metrics.json').read_text()),
            scope='synthetic randomized scenes; not real-world recognition accuracy',training_manifest='data/vision_synthetic/manifest.json',
            license='Ultralytics AGPL-3.0; see https://ultralytics.com/license')
        dest.with_name('metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(metadata,flush=True)

if __name__=='__main__':main()

"""Real learned YOLOv8 ONNX inference. Only RGB and exposure metadata enter.

No map crop, color classifier, simulator IDs or semantic fallback. Output boxes
are in the original image. The single worker bounds the inference queue to one.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from PIL import Image
from configuration import ROOT


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple
    class_name: str
    score: float

    def record(self):
        return dict(bbox_xyxy=list(self.bbox_xyxy), class_name=self.class_name,
                    score=float(self.score), confidence=float(self.score),
                    source='YOLOv8n learned ONNX detector', calibrated_probability=False)


@dataclass(frozen=True)
class VisionFrame:
    packet: object
    detections: tuple
    inference_ms: float
    model_sha256: str


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = float(np.maximum(hi-lo, 0).prod())
    return inter/max(1e-9, float((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod())-inter)


class YoloDetector:
    def __init__(self, cfg):
        import onnxruntime as ort
        self.cfg = cfg
        path = ROOT/cfg.yolo_model
        if not path.is_file():
            raise FileNotFoundError('缺少已训练 YOLO 权重：%s；请保留 models/vision，或运行 tools/train_vision.py。没有颜色规则回退。' % path)
        self.sha = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = path.with_name('metadata.json')
        self.metadata = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {}
        if self.metadata.get('onnx_sha256', self.sha) != self.sha:
            raise RuntimeError('YOLO 权重与 metadata.json 的 SHA-256 不一致')
        opt = ort.SessionOptions(); opt.intra_op_num_threads = 2; opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(self.session.get_inputs()[0].shape[-1])
        if self.size != cfg.yolo_input_size:raise ValueError('YOLO input size differs from configuration')
        self.names = {0: 'door', 1: 'suspected_glass', 2: 'box'}

    def infer(self, packet):
        start = time.perf_counter(); rgb = packet.rgb
        if rgb is None: return VisionFrame(packet, (), 0., self.sha)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('YOLO expects HxWx3 uint8 RGB')
        h, w = rgb.shape[:2]; scale = min(self.size/w, self.size/h)
        nw, nh = int(round(w*scale)), int(round(h*scale))
        left, top = (self.size-nw)//2, (self.size-nh)//2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top+nh, left:left+nw] = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR))
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)/255.
        raw = np.asarray(self.session.run(None, {self.input_name: tensor})[0])[0]
        channels=7 if len(self.session.get_outputs()[0].shape)>1 and self.session.get_outputs()[0].shape[1]==7 else 6
        if raw.shape[0] == channels: raw = raw.T
        if raw.ndim != 2 or raw.shape[1] not in (6,7):
            raise ValueError('Expected YOLOv8 detect output Nx(4+classes), got %r' % (raw.shape,))
        candidates = []
        for row in raw:
            cls = int(np.argmax(row[4:])); score = float(row[4+cls])
            if not np.isfinite(row).all() or score < self.cfg.yolo_confidence: continue
            cx, cy, bw, bh = row[:4]
            box = [(cx-bw/2-left)/scale, (cy-bh/2-top)/scale,
                   (cx+bw/2-left)/scale, (cy+bh/2-top)/scale]
            box = np.clip(box, [0, 0, 0, 0], [w, h, w, h]).tolist()
            if min(box[2]-box[0], box[3]-box[1]) < 5: continue
            candidates.append(Detection(tuple(box), self.names[cls], score))
        candidates.sort(key=lambda d: d.score, reverse=True)
        kept = []
        for det in candidates:
            if any(k.class_name == det.class_name and iou(k.bbox_xyxy, det.bbox_xyxy) > self.cfg.yolo_iou for k in kept): continue
            kept.append(det)
            if len(kept) >= 30: break
        return VisionFrame(packet, tuple(kept), (time.perf_counter()-start)*1000, self.sha)


class VisionWorker:
    def __init__(self, cfg, detector=None):
        self.cfg, self.detector = cfg, detector or YoloDetector(cfg)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='yolo')
        self.future = None; self.submitted = self.delivered = self.dropped = 0
        self.inference_ms = []; self.model_sha256 = getattr(self.detector, 'sha', 'test-detector')

    def process(self, packet):
        result = None
        if self.future is not None and self.future.done():
            result = self.future.result(); self.future = None
        if self.future is None and packet.rgb is not None:
            self.future = self.pool.submit(self.detector.infer, packet); self.submitted += 1
        elif packet.rgb is not None: self.dropped += 1
        if result is None and self.future is not None:
            try:
                result = self.future.result(timeout=self.cfg.vision_wait_s); self.future = None
            except TimeoutError: pass
        if result is not None:
            self.delivered += 1; self.inference_ms.append(result.inference_ms)
        return result

    def record(self):
        return dict(model_sha256=self.model_sha256, submitted=self.submitted, delivered=self.delivered,
                    busy_frames_skipped=self.dropped, queue_capacity=1, providers=['CPUExecutionProvider'],
                    inference_ms_mean=float(np.mean(self.inference_ms)) if self.inference_ms else None,
                    inference_ms_p95=float(np.percentile(self.inference_ms, 95)) if self.inference_ms else None)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

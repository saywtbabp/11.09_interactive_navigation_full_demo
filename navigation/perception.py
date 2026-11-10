"""Learned semantics, accumulated range geometry and verified passage.

No simulator objects, scene assignments, materials, IDs or hinge encoders.
"""
from dataclasses import dataclass, field, asdict
import math
import numpy as np
from .sensors import project
from .ambiguity import RangeMemory
from .vision import VisionWorker


@dataclass
class Record:
    stable_id: str
    geometry: dict
    semantic: str = 'unknown'
    traversability: str = 'unverified'
    confirmation: str = 'candidate'
    geometry_state: str = 'candidate'
    verification_state: str = 'unverified'
    visual_evidence: dict = field(default_factory=dict)
    range_evidence: dict = field(default_factory=dict)
    ambiguity_evidence: dict = field(default_factory=dict)
    contact_history: list = field(default_factory=list)
    last_seen_time: float = -1.
    last_range_time: float = -1.
    last_update_time: float = -1.
    evidence_kind: str = ''
    evidence_count: int = 0
    range_count: int = 0
    contradict_count: int = 0
    angle: float = 0.
    angle_time: float = -1.
    angular_velocity: float = 0.
    linear_position: float = 0.
    linear_velocity: float = 0.
    probe_used: bool = False
    bypass_selected: bool = False


class Memory:
    def __init__(self, arch, cfg, detector=None):
        self.arch, self.cfg = arch, cfg
        self.records = {o['id']: Record(o['id'], dict(o)) for o in arch['openings']}
        self.version = 0; self.events = []; self.last_frame = -1.
        self.last_visual_frame = -1.; self.detections = []; self.packet = None
        self.ranges = RangeMemory(cfg); self.vision = VisionWorker(cfg, detector)
        self.vision_log = []; self.last_association = {}
        self.visual_packet = None

    def event(self, time, oid, kind, **info):
        self.version += 1
        self.events.append(dict(time_s=float(time), stable_id=oid, event=kind, map_version=self.version, **info))

    def fuse(self, p):
        if p is None or not np.isfinite(p.time) or p.time <= self.last_frame: return False
        self.last_frame = p.time; self.packet = p; self.ranges.add(p)
        for oid, rec in self.records.items():
            if rec.semantic == 'door': self.track_panel(rec, p)
            e = self.ranges.evidence(rec.geometry, p)
            rec.range_evidence = e
            if not e['quality_valid']: continue
            if p.time-rec.last_range_time>self.cfg.vision_max_gap:rec.range_count=0
            rec.last_range_time = p.time
            if e['side_support_confirmed']: rec.geometry_state = 'confirmed'
            candidate = e['ambiguous_gap'] or (e['panel_support'] > .55 and e['side_support_confirmed'])
            rec.range_count = rec.range_count+1 if candidate else 0
            if rec.range_count >= self.cfg.confirmation_frames:
                rec.ambiguity_evidence = dict(e, support_scans=rec.range_count)
                if rec.traversability == 'unverified' and rec.verification_state == 'unverified':
                    rec.verification_state = 'pending'; rec.confirmation = 'pending_verification'
                    self.event(p.time, oid, 'opening_requires_verification', semantic=rec.semantic,
                               source='bilateral accumulated range geometry', panel_support=e['panel_support'])
            # Positive obstacle returns can revoke free passage. Missing rays cannot.
            if rec.traversability == 'free' and e['panel_support'] > .55:
                rec.contradict_count += 1
                if rec.contradict_count == self.cfg.confirmation_frames:
                    rec.traversability = 'unverified'; rec.verification_state = 'pending'
                    rec.confirmation = 'pending_verification'
                    self.event(p.time, oid, 'contradiction_requires_reverification')
            elif rec.traversability == 'free': rec.contradict_count = 0
        frame = self.vision.process(p)
        if frame is not None: self.fuse_visual(frame, p.time)
        return True

    def fuse_visual(self, frame, now):
        p = frame.packet
        if p.time <= self.last_visual_frame: return False
        if not 0 <= now-p.time <= self.cfg.vision_max_age:
            self.vision_log.append(dict(time_s=now, image_time_s=p.time, rejected='expired_or_future_exposure'))
            return False
        self.last_visual_frame = p.time; self.detections = []
        self.visual_packet = p
        projections, pairs = {}, []
        for oid, rec in self.records.items():
            x, y = rec.geometry['center_xy_m']; width = rec.geometry['width_m']
            if np.linalg.norm(np.array([x, y])-p.camera_position[:2]) > 4.: continue
            pts = [[x-width*.45, y, z] for z in (.05, 2.02)]+[[x+width*.45, y, z] for z in (.05, 2.02)]
            uv, depth = project(p, pts)
            if np.any(depth < .25): continue
            h, w = p.rgb.shape[:2]
            box = np.array([max(0, uv[:, 0].min()), max(0, uv[:, 1].min()),
                            min(w, uv[:, 0].max()), min(h, uv[:, 1].max())])
            if min(box[2:]-box[:2]) < 10: continue
            # Match at exposure pose, never at inference arrival pose.
            e = self.ranges.evidence(rec.geometry, p)
            box_range_candidate=e['valid_ratio']>=.5 and e['ray_count']>=5
            if not e['quality_valid'] and not box_range_candidate: continue
            projections[oid] = box.tolist()
            for i, det in enumerate(frame.detections):
                if not e['quality_valid'] and det.class_name!='box':continue
                if det.class_name in ('door','box'):
                    # A farther object seen through an empty opening is not an
                    # object at THIS opening. Require near-plane range support.
                    points=p.points[p.valid]
                    local=(np.abs(points[:,0]-x)<width*.46)&(points[:,1]>y-.42)&(points[:,1]<y+.16)
                    if int(local.sum())<3:continue
                    if det.class_name=='door' and e['panel_support']<.25:continue
                b = np.array(det.bbox_xyxy)
                inter = np.maximum(np.minimum(box[2:], b[2:])-np.maximum(box[:2], b[:2]), 0).prod()
                cover = float(inter/max(1., min(np.prod(box[2:]-box[:2]), np.prod(b[2:]-b[:2]))))
                dx = abs((b[0]+b[2]-box[0]-box[2])/2)/max(1., box[2]-box[0])
                if cover >= .55 and dx < .45: pairs.append((oid, i, cover))
        matches = {}
        for oid, i, _ in pairs:
            if sum(a == oid for a, _, _ in pairs) == 1 and sum(b == i for _, b, _ in pairs) == 1:
                matches[oid] = i
        for i, det in enumerate(frame.detections):
            ids = [oid for oid, j in matches.items() if j == i]
            self.detections.append(dict(det.record(), stable_id=ids[0] if ids else '-', time_s=p.time,
                                        association='unique_geometry_match' if ids else 'unassociated_or_ambiguous'))
        for oid, rec in self.records.items():
            if oid not in matches:
                # Leaving the FOV or missing a detection supplies no contrary
                # class evidence. Keep valid observations of this stable ID.
                continue
            det = frame.detections[matches[oid]]; kind = det.class_name
            rec.last_seen_time = p.time
            rec.visual_evidence = dict(det.record(), time_s=p.time, inference_ms=frame.inference_ms,
                                       model_sha256=frame.model_sha256, association='unique_geometry_match')
            if rec.evidence_kind == kind: rec.evidence_count += 1
            else: rec.evidence_kind = kind; rec.evidence_count = 1
            if rec.evidence_count < self.cfg.confirmation_frames: continue
            if rec.semantic != kind:
                protected=rec.semantic!='unknown' and (rec.probe_used or rec.contact_history or rec.traversability=='free')
                if protected:
                    if rec.evidence_count==self.cfg.confirmation_frames:
                        rec.confirmation='pending_verification'
                        self.event(p.time,oid,'semantic_conflict_requires_reverification',candidate=kind,
                                   preserved=rec.semantic,contact_history_preserved=True)
                    continue
                previous = rec.semantic; rec.semantic = kind
                self.event(p.time, oid, 'yolo_multi_frame_confirmed', semantic=kind, previous=previous,
                           score=det.score, contact_history_preserved=bool(rec.contact_history))
            rec.last_update_time = p.time
            if rec.traversability == 'unverified':
                rec.confirmation = 'confirmed' if kind in ('door','box') else 'pending_verification'
                if kind == 'suspected_glass': rec.verification_state = 'pending'
        self.last_association = dict(image_time_s=p.time, arrival_sim_time_s=now, projections=projections,
                                     matches=matches, eligible_pairs=pairs)
        self.vision_log.append(dict(time_s=now, image_time_s=p.time, inference_ms=frame.inference_ms,
                                   model_sha256=frame.model_sha256, detections=list(self.detections),
                                   association=self.last_association))
        return True

    def track_panel(self, rec, p):
        """Existing standard-frame western-hinge range fit for door control."""
        if getattr(self,'onboard',None) is not None:
            track=self.onboard.tracks[rec.stable_id]
            if track.valid:
                rec.angle,rec.angular_velocity=track.q,track.omega
                rec.angle_time=track.time
            return
        o = rec.geometry; x, y = o['center_xy_m']; width = o['width_m']-.18
        hinge = np.array([x-width/2, y+.035]); points = p.points[p.valid, :2]
        q = points-hinge; radius = np.linalg.norm(q, axis=1)
        q = q[(radius > .13) & (radius < width-.06) & (q[:, 0] > -.08) & (q[:, 1] > -.07)]
        if len(q) < 3: return
        angles = np.linspace(0, math.radians(88), 177); normals = np.column_stack((-np.sin(angles), np.cos(angles)))
        residual = abs(q@normals.T+.035); counts = (residual < .014).sum(axis=0)
        k = int(np.argmax(counts)); mask = residual[:, k] < .018
        if mask.sum() < 3: return
        pts = q[mask]
        if np.linalg.norm(np.ptp(pts, axis=0)) < .11: return
        centered = pts-pts.mean(axis=0); _, vec = np.linalg.eigh(centered.T@centered); v = vec[:, -1]
        if v[0] < 0: v = -v
        angle = float(np.clip(math.atan2(v[1], v[0]), 0, math.radians(88)))
        if rec.angle_time >= 0 and p.time > rec.angle_time: rec.angular_velocity = (angle-rec.angle)/(p.time-rec.angle_time)
        rec.angle = angle; rec.angle_time = p.time

    def verified(self, oid, time, outcome, force, **kw):
        if outcome not in ('free', 'blocked', 'inconclusive'): raise ValueError('invalid verification outcome')
        r = self.records[oid]
        r.traversability = outcome if outcome != 'inconclusive' else 'unverified'
        r.verification_state = 'verified_'+outcome if outcome != 'inconclusive' else 'inconclusive'
        r.confirmation = 'confirmed' if outcome != 'inconclusive' else 'pending_verification'
        r.last_update_time = time
        source = kw.pop('force_source', 'IMU/AHRS/motor residual estimate')
        r.contact_history.append(dict(time_s=time, outcome=outcome, force_peak_N=force, force_source=source, **kw))
        self.event(time, oid, 'contact_verification', outcome=outcome, force_peak_N=force, force_source=source, **kw)

    def snapshot(self):
        return dict(version=self.version, records={k: asdict(v) for k, v in self.records.items()}, vision=self.vision.record())

    def close(self): self.vision.close()

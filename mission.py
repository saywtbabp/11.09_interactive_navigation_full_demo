"""Continuous RGB/range perception, cognitive map, NAMO and physical flight."""
import runtime
import json,math,time,traceback
from pathlib import Path
from functools import partial
from dataclasses import asdict
import numpy as np
from PIL import Image
from configuration import ROOT,architecture
from scene.build import build_xml
from navigation.physics import World,PhysicsFailure,forward_yaw
from navigation.sensors import Sensors
from navigation.perception import Memory
from navigation.geometry import Planner,rectangle
from navigation.door_clearance import VehicleEnvelope
from navigation.interactions import approach,panel_geometry
from navigation.verification import verify_opening
from session import InteractionSession
from local_navigation import next_leg,cruise
from motion_reference import SmoothFlightReference


def safe(x):
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,Path):return str(x)
    return x


class Mission:
    def __init__(self,cfg,folder,ui=None,rate=None,scene_file=None):
        self.cfg=cfg;self.arch=architecture();self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        spec=json.loads(Path(scene_file or ROOT/'config/scene.json').read_text(encoding='utf-8'))
        self.rate=spec['lambda_m_per_J'] if rate is None else rate;self.force_cap=spec['force_cap_N']
        # Hidden physical assignments are consumed here and never queried by policy.
        self.world=World(self.arch,cfg,door_parameters=spec['door_parameters'],observed_contents=spec['contents'],
            scene_builder=partial(build_xml,box_parameters=spec['box_parameters']))
        (self.folder/'scene_truth_for_evaluation.json').write_text(json.dumps(self.world.truth,indent=2),encoding='utf-8')
        self.envelope=VehicleEnvelope.from_world(self.world)
        self.sensors=Sensors(self.world,cfg);self.memory=Memory(self.arch,cfg);self.ui=ui
        self.current=None;self.active_session=None;self.display_session=None;self.interaction_records={}
        self.status='初始化';self.reason='RGB / 激光 / IMU外力启动';self.action='Observe'
        self.metrics=dict(G=None,p_D=None,D=None);self.trajectory=[];self.telemetry=[];self.frames=[];self.camera_frames=[]
        self.path=np.empty((0,2));self.decisions=[];self.verifications=[];self.map_history=[]
        self.replans=0;self.probes=0;self.success=False;self.failure='';self.saved_keys=set()
        self.keyframes=[]
        self.wait_status={}
        self.local_leg=None;self.prepared_sessions={}
        self._exit_reference=None
        self.prediction=False;self.progress=(0,0);self.observer=None;self.wall_started=time.perf_counter()
        self._last_image=-10.;self._last_frame=-1.;self._last_map_version=-1
        self._sensor_log=(self.folder/'sensors.jsonl').open('w',encoding='utf-8')
        self.world.callback=self.on_step
        if ui:ui.attach(self)

    def write_json(self,path,data):
        Path(path).write_text(json.dumps(safe(data),ensure_ascii=False,allow_nan=False),encoding='utf-8')

    def on_step(self,w):
        session=self.active_session
        if session is not None:
            if w.stage=='PUSH_UNLOAD':
                self.status='续推后卸载 / 准备检查穿越净空';self.action='Observe'
                self.reason='正在卸载和调整姿态；随后验证物体位置、速度及完整机体通行净空'
            session.observe_state();session.capture(w)
        p,v,R,g=w.observed();now=float(w.data.time)
        self.trajectory.append([now,*p.tolist()]);self.telemetry.append([now,w.stage,*p.tolist(),*v.tolist(),*g.tolist(),*w.ctrl.tolist(),float(w.force),float(w.wall_force)])
        packet,fresh=self.sensors.acquire()
        if fresh:
            self.memory.fuse(packet)
            # Perception range-fit cannot overwrite current higher-rate reference
            # measurements used by 11.08; the source is explicit in the log/UI.
            if session is not None:session.observe_state()
            self._sensor_log.write(json.dumps(safe(dict(time_s=packet.time,position=packet.position,
                distances_m=packet.distances,valid=packet.valid,detections=self.memory.detections,
                force_observer=packet.force_measurement,association=self.memory.last_association)),ensure_ascii=False)+'\n')
            period=.4 if session is not None or w.stage=='GLASS_VERIFY' else 1.
            exposure=self.memory.visual_packet or packet
            if exposure.rgb is not None and exposure.time-self._last_image>=period:
                folder=self.folder/'camera';folder.mkdir(exist_ok=True)
                name=f'{exposure.time:09.3f}.jpg';Image.fromarray(exposure.rgb).save(folder/name,quality=80)
                self.camera_frames.append(dict(time=float(exposure.time),file='camera/'+name,detections=safe(self.memory.detections)))
                self._last_image=float(exposure.time)
        if self.memory.version!=self._last_map_version:
            self.map_history.append(dict(time=now,version=self.memory.version,records=self.map_state()))
            self._last_map_version=self.memory.version
        if now-self._last_frame>=.099:
            self.frames.append(dict(time=now,position=p.tolist(),yaw=float(math.atan2(R[1,0],R[0,0])),
                stage=w.stage,current=self.current,action=self.action,status=self.status,reason=self.reason,
                metrics=safe(self.metrics),map=self.map_state(),map_version=self.memory.version,
                contact_N=float(w.force),force_estimate=self.sensors.contact.record(),
                speed_mps=float(np.linalg.norm(v)),
                wait_status=self.wait_status if w.stage in ('PROBE_UNLOAD','PROBE_HOVER','AFTER_PROBE_STATE','ONLINE_UNLOAD','ONLINE_COMPLETE_UNLOAD','PASSAGE_RELEASE','PASSAGE_CONNECT') else None,
                interaction_snapshot=len(session.snapshots)-1 if session is not None else None))
            self.frames[-1]['local_leg']=self.local_leg
            self._last_frame=now
        if w.wall_peak>.35:raise PhysicsFailure(f'unintended wall/jamb contact: {w.wall_peak:.3f} N')
        if self.active_session is None and (w.stage in ('FLIGHT','TO_OBSERVATION') or w.stage.startswith('CRUISE')) and w.force>.35:
            raise PhysicsFailure(f'unexpected object contact during navigation: {w.force:.3f} N')
        if self.ui:self.ui.tick()

    def map_state(self):
        return {oid:dict(semantic=r.semantic,traversability=r.traversability,confirmation=r.confirmation,
                    bypass_selected=r.bypass_selected,probe_used=r.probe_used,angle=r.angle,
                    linear_position=r.linear_position,last_seen_time=r.last_seen_time)
                for oid,r in self.memory.records.items()}

    def planner(self,extra_sealed=(),*,verified_opening=None):
        sealed=list(extra_sealed);panels=[]
        for oid,r in self.memory.records.items():
            if r.traversability=='blocked' or r.bypass_selected:sealed.append(oid)
            # The just verified passage supersedes a tentative semantic panel
            # at this aperture, even if vision still labels it as a box.
            if oid==verified_opening and r.traversability=='free' and not r.probe_used:
                continue
            if (r.traversability=='free' or r.probe_used or r.bypass_selected) and r.semantic=='door':panels.append(panel_geometry(r.geometry,r.angle)[2])
            elif (r.traversability=='free' or r.probe_used or r.bypass_selected) and r.semantic=='box':
                x,y=r.geometry['center_xy_m'];panels.append(rectangle([x,y+r.linear_position],[.59125,.25]))
        return Planner(self.arch,self.cfg,panels=panels,sealed=sealed,footprint=self.envelope.half[:2],
            region=None if self.local_leg is None else self.local_leg['region'])

    def plan(self,goal):
        path,_=self.planner().plan(self.world.observed()[0],goal);self.path=path;self.replans+=1
        self.memory.event(self.world.data.time,self.current or '', 'replan',count=self.replans)
        return path

    def next_opening(self,path):
        position=self.world.observed()[0]
        for a,b in zip(path[:-1],path[1:]):
            if abs(b[1]-a[1])<1e-9:continue
            candidates=[]
            for oid,r in self.memory.records.items():
                x,y=r.geometry['center_xy_m']
                if y<position[1]-.55:continue
                fraction=(y-a[1])/(b[1]-a[1])
                if 0<=fraction<=1 and abs(a[0]+fraction*(b[0]-a[0])-x)<r.geometry['width_m']/2:
                    candidates.append((fraction,oid))
            if candidates:return min(candidates)[1]
        return None

    def observe(self,rec):
        w=self.world;o=rec.geometry
        if self.reliable_interaction(rec):return True
        w.align_yaw(forward_yaw(o['normal_xy']),stage='ALIGN_OPENING')
        self.status='观察开口 / 融合视觉与测距';self.action='Observe'
        self.reason='类别来自真实机载RGB的YOLO；多帧测距验证几何'
        deadline=w.data.time+1.8
        while w.data.time<deadline:
            e=rec.range_evidence
            recent=0<=w.data.time-e.get('time_s',-1.)<=.25
            quality=e.get('quality_valid',False) or (rec.semantic=='box' and e.get('valid_ratio',0.)>.5)
            if recent and quality and (rec.semantic in ('door','box','suspected_glass') or rec.verification_state=='pending' or rec.traversability=='free'):
                return True
            w.stage='OBSERVE';w.step(yaw=0.)
        return False

    def reliable_interaction(self,rec):
        e=rec.range_evidence
        geometry=(rec.geometry_state=='confirmed' or (e.get('side_support_confirmed',False)
            and e.get('valid_ratio',0.)>=.5 and e.get('ray_count',0)>=5))
        return (rec.semantic in ('door','box') and rec.confirmation=='confirmed'
            and rec.evidence_count>=self.cfg.confirmation_frames and geometry)

    def known_cost_gate(self,rec,location):
        if rec.stable_id in self.prepared_sessions:return self.prepared_sessions[rec.stable_id]
        self.status='类别和几何可靠 / 当前位置比较成本';self.action='Compare'
        self.reason='先比较额外接近和试探成本，再决定是否靠近'
        if self.ui:self.ui.draw();self.ui.root.update()
        try:session=InteractionSession(self,rec)
        except KeyError:return None
        self.display_session=session
        try:admitted=session.evaluate_precontact(location)
        except ValueError as exc:
            session.preview_worker.close()
            self.memory.event(self.world.data.time,rec.stable_id,'remote_comparison_unavailable',reason=str(exc))
            return None
        if not admitted:
            session.preview_worker.close();session.save_record()
        self.prepared_sessions[rec.stable_id]=session
        return session

    def cruise(self,path,stage='CRUISE',stop_when=None):
        reference=self._exit_reference
        self._exit_reference=None
        if reference is not None and self.world.data.time-reference.time>2*self.cfg.timestep+1e-9:
            reference=None
        return cruise(self.world,path,self.planner(),speed=.45,lookahead=self.cfg.cruise_lookahead_m,
            stage=stage,stop_when=stop_when,reference=reference)

    def exit_verified_opening(self,rec,exit_point,yaw):
        world=self.world
        planner=self.planner(verified_opening=rec.stable_id)
        path,_=planner.plan(world.observed()[0],exit_point)
        reference=SmoothFlightReference(world)
        self.status='验证净空 / 平滑飞出开口'
        self.action='Cross'
        self.reason='从当前速度逐步提速，飞离门框后连续接入巡航'
        cruise(world,path,planner,speed=.25,lookahead=self.cfg.cruise_lookahead_m,
               stage='CRUISE_VERIFIED_EXIT',reference=reference,exit_yaw=yaw)
        self._exit_reference=reference
        self.memory.event(world.data.time,rec.stable_id,'verified_exit_complete',
                          exit_speed_mps=float(np.linalg.norm(world.observed()[1][:2])),
                          velocity_slew_mps2=reference.max_velocity_slew)

    def save_key(self,label):
        if label in self.saved_keys:return
        self.saved_keys.add(label)
        from ui.art import save_views
        save_views(self,self.folder/label)
        self.keyframes.append(dict(time=float(self.world.data.time),label=label,file=label+'_dashboard.png',scene=label+'_3d.png'))
        self.write_json(self.folder/'keyframes.json',self.keyframes)

    def save_checkpoint(self,rec):
        """Read-only evaluator fixture. The online mission NEVER restores it."""
        import mujoco
        w=self.world;spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(w.model,spec));mujoco.mj_getState(w.model,w.data,state,spec)
        folder=self.folder/'checkpoints';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(rec.stable_id+'.npz'),state=state,state_spec=int(spec),
            target=w.target,ctrl=w.ctrl,yaw_reference=w.yaw_reference,yaw_target=w.yaw_target,step_count=w.step_count)
        self.write_json(folder/(rec.stable_id+'.json'),dict(time=float(w.data.time),current=rec.stable_id,
            records=self.memory.snapshot()['records'],purpose='offline parameter-study checkpoint, not used by online policy'))

    def run(self):
        try:
            w=self.world;w.hold(self.arch['start_xyz_m'],1.2,stage='INITIAL_SETTLE',yaw=0.)
            self.save_key('00_initial')
            for _ in range(35):
                if np.linalg.norm(w.observed()[0]-self.arch['goal_xyz_m'])<.10:self.success=True;break
                self.local_leg=next_leg(self.arch,self.cfg,w.observed()[0])
                self.status='局部导航 / 下一段规划';self.action='Navigate'
                self.reason='只比较当前隔墙后的共同局部汇合点；巡航中持续观测'
                # Evaluate reliable candidates BEFORE publishing or flying an
                # optimistic path toward one. S01 can be rejected at F01's exit.
                for candidate in self.memory.records.values():
                    if self.local_leg['row_y'] is None:break
                    if abs(candidate.geometry['center_xy_m'][1]-self.local_leg['row_y'])>.01:continue
                    if candidate.bypass_selected or candidate.probe_used or candidate.traversability in ('free','blocked'):continue
                    if self.reliable_interaction(candidate):self.known_cost_gate(candidate,'before_approach')
                path=self.plan(self.local_leg['goal_xy']);oid=self.next_opening(path)
                if oid is None:self.cruise(path);continue
                self.current=oid;rec=self.memory.records[oid];o=rec.geometry;x,y=o['center_xy_m']
                if self.display_session is not None and self.display_session.oid!=oid:
                    self.display_session=None;self.metrics=dict(G=None,p_D=None,D=None,F_star=None)
                if rec.traversability=='free':
                    self.status='已验证净空 / 通行';self.action='Cross';self.cruise(self.plan([x,y+.72]),stage='CRUISE_CROSSING');continue
                session=self.known_cost_gate(rec,'before_approach') if self.reliable_interaction(rec) else None
                if session is not None and session.completed:continue
                dest=approach(o)
                view=rec.ambiguity_evidence or rec.range_evidence
                if rec.semantic=='unknown' and view.get('clearance_support',0.)>.8 and view.get('panel_support',1.)<.15:
                    # Empty-looking apertures use the corridor centreline for
                    # observation. The door's nominal contact-arm offset made
                    # even F01/F02 look like a tentative move toward a door.
                    # This is a viewpoint choice, never a free-space verdict.
                    dest=np.array([x,y-.95,1.25])
                path=self.plan(dest)
                def inspect_while_cruising():
                    nonlocal session
                    if session is None and self.reliable_interaction(rec):
                        before=float(np.linalg.norm(w.observed()[1][:2]))
                        session=self.known_cost_gate(rec,'recognition_while_cruising')
                        self.memory.event(w.data.time,oid,'moving_cost_gate',speed_mps=before,
                            brake_inserted=False,local_leg=self.local_leg)
                        if session is not None and session.completed:return True
                        self.status='巡航接近 / 识别及成本比较已完成';self.action='Navigate'
                        self.reason='没有插入停稳；继续沿当前安全走廊接近'
                    return False
                arrived=self.cruise(path,stage='CRUISE_TO_OBSERVATION',stop_when=inspect_while_cruising)
                if not arrived:continue
                if not self.observe(rec):
                    if session is not None:session.preview_worker.close()
                    rec.bypass_selected=True;self.memory.event(w.data.time,oid,'insufficient_observation_avoid',semantic=rec.semantic);continue
                if rec.semantic in ('door','box'):
                    self.save_checkpoint(rec)
                    try:self.active_session=session if session is not None and session.kind==rec.semantic else InteractionSession(self,rec)
                    except KeyError:
                        self.memory.event(w.data.time,oid,'state_measurement_unavailable_verify',semantic=rec.semantic)
                        verify_opening(self,rec,free_exit=self.exit_verified_opening);continue
                    self.display_session=self.active_session
                    try:self.active_session.execute()
                    finally:self.active_session=None
                elif rec.semantic=='suspected_glass' or rec.verification_state=='pending':
                    self.status='玻璃/歧义开口验证';self.reason='沿用10.31低速接触与IMU残差外力验证，不按真实接触标签判玻璃'
                    verify_opening(self,rec,free_exit=self.exit_verified_opening)
                elif rec.traversability=='free':w.fly(self.plan([x,y+.72])[1:],yaw=0.,speed=.35)
                else:
                    rec.bypass_selected=True;self.memory.event(w.data.time,oid,'unresolved_opening_avoid')
            else:self.failure='navigation iteration budget exhausted'
        except Exception as exc:
            self.failure=f'{type(exc).__name__}: {exc}'
            (self.folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print(traceback.format_exc(),flush=True)
        self.status='任务完成' if self.success else '任务停止';self.reason=self.failure or '到达目标，所有动作在同一物理世界连续执行'
        self.save();self.save_key('99_final')
        return self.success

    def save(self):
        self._sensor_log.flush()
        data=dict(success=self.success,failure=self.failure,simulated_time_s=float(self.world.data.time),
            navigation_revision='paired_window_online_control' if self.cfg.online_decisions else 'early_gate_two_stage_decoupled_preview',
            execution_settings=dict(contact_approach_speed_mps=self.cfg.contact_approach_speed,
                navigation_scope='rolling_local_horizon',local_merge_distance_m=self.cfg.local_merge_distance_m,
                cruise_observation=True,inference_completion='freeze_on_actual_passage',
                completion_recovery='direct_passage_connection',
                verified_exit='continuous_reference_through_waypoint',
                contact_control='debounced_coast_and_reapproach',continuous_motion_reference=True,
                near_contact_band_m=self.cfg.contact_slow_band_m,decision_mode='online' if self.cfg.online_decisions else 'after-probe',
                evidence_window_s=.1,preview_period_s=None if self.cfg.online_decisions else self.cfg.preview_period_s),
            wall_time_s=time.perf_counter()-self.wall_started,final_position=self.world.observed()[0],
            probes=self.probes,replans=self.replans,wall_contact_peak_N=self.world.wall_peak,
            contact_peak_N=self.world.contact_peak,map_version=self.memory.version,vision=self.memory.vision.record(),
            sensor_frames=self.sensors.frames,verification_count=len(self.verifications),
            interactions={oid:r['outcome'] for oid,r in self.interaction_records.items()},
            lambda_m_per_J=self.rate,measurement_sources=dict(vision='actual body-fixed RGB / trained 3-class YOLO',
            glass='original 10.31 calibrated IMU/AHRS/motor residual verification',
            door_box='11.08 simulation reference joint state/contact wrench; not real onboard force accuracy',
            localization='simulation reference own pose/velocity; not a SLAM validation'),
            physical_world='single continuous MuJoCo world; no episode resets or object state edits')
        self.summary=safe(data)
        self.write_json(self.folder/'summary.json',data);self.write_json(self.folder/'online_map.json',self.memory.snapshot())
        self.write_json(self.folder/'events.json',self.memory.events);self.write_json(self.folder/'decisions.json',self.decisions)
        self.write_json(self.folder/'verification_trace.json',self.verifications);self.write_json(self.folder/'vision_trace.json',self.memory.vision_log)
        self.write_json(self.folder/'replay.json',dict(summary=data,map=self.arch,frames=self.frames,camera_frames=self.camera_frames,
            map_history=self.map_history,decisions=self.decisions,events=self.memory.events,interactions=self.interaction_records,
            verifications=self.verifications,telemetry=self.telemetry))
        print('MISSION_RESULT',json.dumps(self.summary,ensure_ascii=False),flush=True)

    def close(self):
        for session in self.prepared_sessions.values():session.preview_worker.close()
        self.memory.close();self.sensors.close();self._sensor_log.close()
        if self.observer:self.observer.close()

"""One 11.08 interaction inside the SAME continuous navigation World."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from interaction_engine import Experiment,path_length,ARM,BOX_HALF
from belief import Belief,ACTIONS
from navigation.geometry import Planner,rectangle
from navigation.interactions import approach,merge_pose,panel_geometry
from navigation.physics import PhysicsFailure
from local_hover import deferred_egress
from route_preview import PlanningView,PreviewWorker,calculate
from decision_rule import evaluate
from online_contact import OnlineContact
from local_navigation import leg_for_opening
import time


class InteractionSession(OnlineContact,Experiment):
    def __init__(self,mission,record):
        # Deliberately do not call Experiment.__init__: no new world, no reset.
        self.mission=mission;self.record=record;self.world=mission.world;self.arch=mission.arch;self.cfg=mission.cfg
        self.opening=record.geometry;self.oid=record.stable_id;self.kind=record.semantic
        if self.kind not in ('door','box'):raise ValueError('visual class not confirmed')
        self.arm=ARM if self.kind=='door' else 1.;self.x,self.y=self.opening['center_xy_m']
        self.staging=approach(self.opening)[:2] if self.kind=='door' else np.array([self.x,self.y-1.03])
        self.local_leg=leg_for_opening(self.arch,self.cfg,self.opening)
        self.goal=np.array(self.local_leg['goal_xy']);self.cap=mission.force_cap
        self.args=SimpleNamespace(case=self.oid,rate=mission.rate,action='guarded_coast',policy='consistent',
            decision_rule='majority',alpha=.1,tie_margin=0.,probe_recovery='local',snapshot=None,
            pause_at_decision=False,playback=1.,output=mission.folder/'interactions'/self.oid)
        self.joint=self.world.model.joint('hinge_'+self.oid).id
        self.surface=self.world.model.geom('surface_'+self.oid).id
        self.envelope=mission.envelope;self.belief=Belief();self.prior=Belief()
        self.events=[];self.trace=[];self.frames=[];self.sensor_frames=[];self.paths={};self.snapshots=[]
        self.work=0.;self.peak=0.;self.last_force=0.;self.rows=None
        self.next_evidence=0.;self.retreat_preview=False;self.next_route_preview=0.;self.last_decision={}
        self.dashboard=None;self.viewer=None;self.snapshot_saved=False;self.validation={}
        self.probed=False;self.committed=False;self.crossed=False;self.probe_recovery=None
        self.probe_recovery_error=None;self.recovery_frames=[];self.status='INIT'
        self.target=self.compute_target();self.completed=False;self.outcome='not_run'
        self.precontact=None;self.preview_worker=PreviewWorker();self.preview_metrics=None
        self.latest_evidence=self.belief;self._evidence_key=None;self._preview_enabled=True
        self._preview_pose=None;self._preview_map=None;self._last_preview=-1.;self._next_feedback=0.
        self.planning_stats=dict(formal_recomputations=0,preview_submissions=0,preview_results=0,
            empty_evidence_skipped=0,unchanged_evidence_skipped=0,formal_wall_s=0.,precontact_evaluations=0)
        self.preview_requests=[];self.preview_errors=[]
        self.initialize_online()

    def planner(self,q,route):
        sealed=[];panels=[]
        for oid,r in self.mission.memory.records.items():
            if oid==self.oid:continue
            if r.traversability=='blocked' or r.bypass_selected:sealed.append(oid)
            if (r.traversability=='free' or r.probe_used or r.bypass_selected) and r.semantic=='door':panels.append(panel_geometry(r.geometry,r.angle)[2])
            elif (r.traversability=='free' or r.probe_used or r.bypass_selected) and r.semantic=='box':
                x,y=r.geometry['center_xy_m'];panels.append(rectangle([x,y+r.linear_position],BOX_HALF))
            if route=='target' and abs(r.geometry['center_xy_m'][1]-self.y)<.01:sealed.append(oid)
        if route=='bypass':sealed.append(self.oid)
        if q is not None:panels.append(self.panel(q))
        return Planner(self.arch,self.cfg,panels=panels,sealed=sealed,footprint=self.envelope.half[:2],region=self.local_leg['region'])

    def event(self,name,**values):
        super().event(name,**values)
        self.mission.memory.event(self.world.data.time,self.oid,name,**values)
        if name=='probe_release':self.mission.status='卸载 / 就近观察';self.mission.action='Observe'
        if name=='probe_recovery_fallback':
            self.mission.status='就近恢复不可行 / 安全回退'
            self.mission.reason='回退原因：'+str(values.get('reason',''))
        if name=='probe_recovery_ready':self.mission.status='安全状态通过 / 执行在线已选动作' if self.cfg.online_decisions else '就近稳定 / 正式决策'
        if name=='global_comparison':
            self.mission.metrics=dict(G=values['gain_m'],p_D=values['p_ok'],D=None,F_star=values['F_star_N'])
            if values['p_ok'] is not None:
                self.mission.status='正式比较完成 / '+('支持续推' if values['push'] else '支持绕行')
                self.mission.reason=f"观测数据权重：续推侧 {values['p_ok']:.1%}；多数门限50%；当前分界 {values['F_star_N']:.3f} N"
        if name=='route_choice':
            self.mission.action='Push' if values['route']!='bypass' else 'Avoid'

    def evidence_snapshot(self,belief,label):
        self.latest_evidence=belief
        key=(len(belief.windows),belief.conflict,belief.precision.tobytes(),belief.linear.tobytes())
        live=label.startswith('live_')
        if live and belief.bounds is None and not belief.conflict:
            self.planning_stats['empty_evidence_skipped']+=1;return None
        if live and key==self._evidence_key:
            self.planning_stats['unchanged_evidence_skipped']+=1;return None
        self._evidence_key=key;snap=belief.snapshot();now=float(self.world.data.time)
        if label=='replan':
            metrics=self.last_decision;source='fresh_formal_geometry'
        elif self.preview_metrics is not None and self._preview_enabled:
            metrics=self.preview_metrics;source='asynchronous_geometry_sample'
        elif not self._preview_enabled and self.last_decision:
            metrics=self.last_decision;source='committed_decision_reference'
        else:metrics={};source='awaiting_geometry_preview' if belief.bounds is not None else 'no_mechanical_evidence'
        snap.update(metrics);snap.update(label=label,time=now,route_preview_status=source,
            boundary_age_s=None if 'geometry_time' not in metrics else max(0.,now-metrics['geometry_time']),
            formal_decision=label=='replan')
        for field in ('push_limit','gain','stroke'):snap.setdefault(field,None)
        limit=snap.get('push_limit');prob,converged=belief.probability(limit) if limit is not None else (None,False)
        snap.update(push_probability=prob,integration_converged=converged)
        snap.update(evaluate(prob,converged,self.args.decision_rule,self.args.alpha,self.args.tie_margin))
        self.snapshots.append(snap)
        self.mission.metrics=dict(G=snap.get('gain'),p_D=snap.get('push_probability'),D=None,F_star=snap.get('push_limit'))
        return snap

    def route_metrics(self,start):
        # Formal decisions always capture NEW state. A preview is never promoted
        # to a decision, even if its wall-clock job happened to finish just now.
        self.mission.status='正式决策 / 按最新状态复算路线'
        self.mission.reason='用当前位置、物体状态与当前地图重新比较；不使用历史预览作决策'
        if self.mission.ui:self.mission.ui.draw();self.mission.ui.root.update()
        started=time.perf_counter();view=PlanningView(self);view.start=np.asarray(start).copy()
        result=calculate(view);self.planning_stats['formal_recomputations']+=1
        self.planning_stats['formal_wall_s']+=time.perf_counter()-started
        return result

    def capture(self,w):
        if self.cfg.online_decisions:
            previous=self.retreat_preview;self.retreat_preview=False
            try:super().capture(w)
            finally:self.retreat_preview=previous
        else:
            self.pump_preview();super().capture(w)

    def measure(self,w):
        if not self.online_active:return super().measure(w)
        self.next_evidence=float('inf')
        super().measure(w)
        if self.rows is not None:
            self.online_row_stages.append(w.stage)
            self.consume_online_windows()

    def pump_preview(self):
        if not self._preview_enabled:return
        result=self.preview_worker.poll()
        if result is not None:
            if 'metrics' in result:
                self.preview_metrics=result['metrics'];self.planning_stats['preview_results']+=1
                self.evidence_snapshot(self.latest_evidence,'route_preview')
            else:
                self.preview_error=result['error'];self.preview_errors.append(result)
                self.preview_metrics=None;self.evidence_snapshot(self.latest_evidence,'route_preview_unavailable')
        if self.latest_evidence.bounds is None or not self.world.stage.startswith('PROBE'):return
        now=float(self.world.data.time)
        if self.preview_worker.future is not None or now-self._last_preview<self.cfg.preview_period_s:return
        pose=np.r_[self.position(),self.state()[0]*self.arm]
        scene=tuple((oid,r.semantic,r.traversability,r.bypass_selected,round(r.angle,2),round(r.linear_position,2))
            for oid,r in self.mission.memory.records.items() if oid!=self.oid)
        changed=self._preview_pose is None or scene!=self._preview_map
        if self._preview_pose is not None:
            changed |= (np.linalg.norm(pose[:2]-self._preview_pose[:2])>=self.cfg.preview_position_change_m
                or abs(pose[2]-self._preview_pose[2])>=self.cfg.preview_object_change_m)
        if not changed:return
        try:view=PlanningView(self)
        except ValueError as exc:self.preview_error=str(exc);self._last_preview=now;return
        if self.preview_worker.submit(view):
            self.preview_requests.append(dict(time=now,position_xy=pose[:2].tolist(),object_s_m=float(pose[2]),
                map_changed=scene!=self._preview_map))
            self._preview_pose=pose;self._preview_map=scene;self._last_preview=now
            self.planning_stats['preview_submissions']+=1

    def approach_speed(self,gap,pos,vel,stage):
        remaining=max(0.,gap-.26);band=self.cfg.contact_slow_band_m
        if remaining>band:
            edge=min(.15,.65*band)
            speed=min(self.cfg.contact_approach_speed,np.sqrt(edge**2+2*self.cfg.contact_braking_accel*(remaining-band)))
            phase='FAST';message='正常接近 / 根据剩余净距制动'
        else:
            speed=float(np.clip(.65*remaining,.055,.15));phase='NEAR';message='近接触减速 / 等待真实接触'
        self.world.stage=stage+'_APPROACH_'+phase
        if self.world.data.time>=self._next_feedback:
            self._next_feedback=self.world.data.time+.10
            self.mission.status=message;self.mission.action='Approach'
            self.mission.reason=f'接触前净距约 {remaining:.3f} m；法向目标速度 {speed:.3f} m/s；尚未接触，不更新力学参数'
        return speed

    def contact_started(self,stage):
        self.mission.status='已接触 / 限力试探' if stage=='PROBE' else '已接触 / 继续推压'
        self.mission.action='Probe' if stage=='PROBE' else 'Push'
        self.mission.reason=('每个有效窗口同步更新权重和边界；支持即续推，不支持即卸载退出' if self.cfg.online_decisions
            else '接触已确认，按有效运动窗口更新参数；达到行程、激励时长或保护条件后主动卸载')

    def on_recovery_status(self,status):
        self.mission.wait_status=status
        labels=dict(detached='脱离接触',clearance='局部净空',envelope_ok='机体包络',speed_ok='速度稳定',
            gyro_ok='角速度稳定',tilt_ok='倾角稳定',yaw_ok='航向对齐',height_ok='高度稳定')
        waiting=[name for key,name in labels.items() if not status[key]]
        if not waiting:
            if not status['exit_path']:
                path_name='通行路径' if status.get('recovery_policy')=='direct_passage_connection' else '退出路径'
                waiting=[path_name+'可行'] if status['exit_path_checked'] else [path_name+'检查']
            elif not status['ready']:waiting=[f"连续稳定确认 {status['dwell_s']:.2f}/0.25 s"]
        self.mission.status='就近恢复 / '+('等待'+ '、'.join(waiting[:3]) if waiting else '全部状态通过')
        if self.cfg.online_decisions and self.online_active:
            prefix=('停止额外推压 / ' if self.online_reason=='no_remaining_navigation_gain' else '已在线选择绕行 / ') if self.online_exit_latched else '续推已达净空 / '
            if self.inference_frozen:prefix='通行几何已满足 / 推断冻结 / '
            if status.get('recovery_policy')=='direct_passage_connection':
                prefix='接入通行路线 / ' if status.get('motion_mode')=='CONNECT' else '脱开后制动与姿态调整 / '
            self.mission.status=prefix+('等待'+'、'.join(waiting[:3]) if waiting else '安全条件通过')
        ready_message='执行已经在线选定的动作' if self.cfg.online_decisions else '按最新状态进行正式决策'
        self.mission.reason=f"接触 {status['contact_N']:.3f} N；速度 {status['speed_mps']:.3f} m/s；角速度 {status['gyro_radps']:.3f} rad/s；"+('待满足：'+ '、'.join(waiting) if waiting else ready_message)
        if status.get('contact_stall_s',0.)>.25:self.mission.reason+='；持续接触且未移动，正在检查局部卸载进展'
        self.mission.action=('Avoid' if self.online_exit_latched else 'Observe') if self.cfg.online_decisions and self.online_active else 'Observe'

    def evaluate_precontact(self,location):
        m,w=self.mission,self.world;start=self.position();q=self.record.angle if self.kind=='door' else self.record.linear_position
        bypass,_=self.plan(start,q,'bypass');candidate,_=self.plan(start,self.target,'target')
        to_stage,_=self.planner(q,'bypass').plan(start,self.staging);staged,_=self.plan(self.staging,q,'bypass')
        gain0=path_length(bypass)-path_length(candidate)
        extra=max(0.,path_length(to_stage)+path_length(staged)-path_length(bypass))
        stroke=min(self.target,np.radians(20) if self.kind=='door' else .30)
        budget=self.args.rate*self.cap*stroke*self.arm;admitted=gain0>extra+budget
        row=dict(time_s=float(w.data.time),opening=self.oid,phase='precontact',semantic=self.kind,G0_m=gain0,
            approach_extra_m=extra,probe_budget_m=budget,action='Probe' if admitted else 'Avoid',p_ok=None,
            reason='geometry_gain_above_probe_budget' if admitted else 'geometry_gain_below_probe_budget',
            gate_location=location,position_xy=start.tolist(),known_geometry_q=q,local_leg=self.local_leg)
        m.decisions.append(row);self.precontact=row;self.planning_stats['precontact_evaluations']+=1
        self.paths['BYPASS_OPTION']=bypass.tolist();self.paths['SHORTCUT_OPTION']=candidate.tolist()
        self.evidence_snapshot(self.belief,'start')
        self.event('precontact_gate',G0_m=gain0,approach_extra_m=extra,probe_budget_m=budget,
            admitted=bool(admitted),location=location,position_xy=start.tolist())
        m.metrics=dict(G=gain0,p_D=None,D=None,F_star=None)
        if not admitted:
            self.record.bypass_selected=True;self.outcome='heuristic_skip';self.completed=True
            m.status='当前位置成本否决 / 直接绕行';m.action='Avoid'
            m.reason=f'几何收益 {gain0:.3f} m ≤ 额外接近 {extra:.3f} + 名义试探 {budget:.3f} m；不再靠近'
            m.save_key(self.oid+'_heuristic_skip')
        return admitted

    def observe_state(self):
        q,v=self.state()
        if self.kind=='door':self.record.angle=q;self.record.angular_velocity=v;self.record.angle_time=self.world.data.time
        else:self.record.linear_position=q;self.record.linear_velocity=v

    def execute(self):
        m,w=self.mission,self.world
        old=w.after_step_observer
        def measure(world):
            if old:old(world)
            self.measure(world)
        w.after_step_observer=measure
        try:
            return self._execute()
        finally:
            self.preview_worker.close()
            w.after_step_observer=old
            self.observe_state();self.save_record()

    def _execute(self):
        m,w=self.mission,self.world
        start=self.position()
        if self.precontact is None and not self.evaluate_precontact('after_required_observation'):return
        if self.precontact['action']!='Probe':return
        to_stage,_=self.planner(self.state()[0],'bypass').plan(start,self.staging)
        extra=self.precontact['approach_extra_m'];budget=self.precontact['probe_budget_m']
        stroke=min(self.target,np.radians(20) if self.kind=='door' else .30)
        m.status='首次启发式：值得试探';m.action='Probe';m.reason='几何收益通过预算门限，尚无质量/阻力概率'
        if np.linalg.norm(start-self.staging)>(.12 if self.cfg.online_decisions else .035):self.fly(to_stage,'TO_INTERACTION')
        self.probed=True;self.record.probe_used=True;m.probes+=1
        if self.cfg.online_decisions:
            self.args.policy='online_majority'
            return self.run_online_contact()
        self.event('probe_choice',action='guarded_coast',approach_extra_m=extra,probe_work_budget_m=budget)
        m.status='限力试探 / 数据约束更新';m.action='Probe'
        reason=self.interact('PROBE',1.,1.4,1.2,stroke)
        self.retreat('AFTER_PROBE');self.observe_state()
        self._preview_enabled=False
        bypass,_,gain,push=self.decision(self.position())
        snap=self.snapshots[-1]
        m.decisions.append(dict(time_s=float(w.data.time),opening=self.oid,phase='postcontact',semantic=self.kind,
            action='Push' if push and reason not in ('impact_guard','no_contact') else 'Avoid',
            p_ok=snap['push_probability'],F_star_N=snap['push_limit'],gain_m=gain,windows=snap['windows'],
            reason=snap['decision_reason'],posterior_source='per-object data-bounded continuous likelihood'))
        m.save_key(self.oid+'_post_probe')
        if not push or reason in ('impact_guard','no_contact'):
            self.event('route_choice',route='bypass',reason=reason if reason in ('impact_guard','no_contact') else snap['decision_reason'])
            prefix=deferred_egress(self,self.position(),self.state()[0])
            if len(prefix)>1:
                m.status='已选择绕行 / 执行退出连接';self.event('bypass_exit_start',distance_m=path_length(prefix))
                w.fly(prefix[1:],yaw=0.,speed=.15,stage='BYPASS_EXIT',position_tolerance=.02)
                self.event('bypass_exit_end')
            self.record.bypass_selected=True;self.outcome='probe_then_bypass';self.completed=True
            return
        self.committed=True;self.event('route_choice',route='push')
        m.status='多数权重支持 / 就近续推';m.action='Push';m.reason='不执行绕行撤离连接，直接重新接触'
        reason='already_clear' if self.state()[0]>=self.target else self.interact('PUSH',1.,15.,2.25,self.target+(.025 if self.kind=='door' else .055))
        self.finish_crossing(reason)

    def finish_crossing(self,reason):
        m,w=self.mission,self.world;self.retreat('AFTER_PUSH');q,v=self.state()
        try:
            if reason=='impact_guard' or abs(v)>.05 or q<self.target:raise ValueError('stable target clearance not confirmed')
            full,_=self.planner(q,'target').plan(self.position(),self.goal)
            # A translated box can occupy the nominal centerline merge point.
            # Stop on the actual clear global path, not inside that box.
            path=[full[0]];cut_y=self.y+1.70
            for a,b in zip(full,full[1:]):
                if a[1]<cut_y<=b[1]:
                    path.append(a+(b-a)*(cut_y-a[1])/(b[1]-a[1]));break
                path.append(b)
                if b[1]>=cut_y:break
            path=np.array(path)
        except ValueError as exc:
            self.event('fallback',reason=str(exc));w.fly([self.staging],yaw=0.,speed=.35,stage='FALLBACK_RETREAT')
            self.record.bypass_selected=True;self.outcome='push_failed_then_bypass';self.completed=True;return
        m.status='实际净空确认 / 穿越';m.action='Cross';m.reason='真实物体位置、速度和完整机体包络通过检查'
        self.record.traversability='free';self.record.verification_state='verified_free';self.record.confirmation='confirmed'
        self.observe_state();self.event('clearance_verified',observed_q=q,path_m=path_length(path))
        self.fly(path,'SHORTCUT');self.crossed=True;self.outcome='shortcut';self.completed=True
        m.save_key(self.oid+'_crossed')

    def save_record(self):
        import json
        self.args.output.mkdir(parents=True,exist_ok=True)
        result=dict(case=self.oid,policy='online_majority' if self.cfg.online_decisions else 'consistent',force_cap_N=self.cap,rate_m_per_J=self.args.rate,
            success=self.completed,failure=None if self.completed else 'interaction did not complete',
            probed=self.probed,committed=self.committed,shortcut=self.crossed,outcome=self.outcome,
            work_J=self.work,contact_peak_N=self.peak,wall_peak_N=self.world.wall_peak,
            path_m=path_length(np.array([f[1:3] for f in self.frames])) if len(self.frames)>1 else 0.,
            target_q=self.target,object_kind=self.kind,envelope_m=self.envelope.half.tolist(),events=self.events,
            paths=self.paths,frames=self.frames,trace=self.trace,snapshots=self.snapshots,sensor_frames=self.sensor_frames,
            force_replay={},action='online_window_majority' if self.cfg.online_decisions else 'guarded_coast',alpha=.1,decision_rule='majority',tie_margin=0.,
            probe_recovery=getattr(getattr(self,'probe_recovery',None),'policy','local'),
            recovery_frames=self.recovery_frames,measurement_mode='simulation_reference',map=self.arch,
            planning_stats=self.planning_stats,
            preview_requests=self.preview_requests,preview_errors=self.preview_errors,
            local_leg=self.local_leg,
            online_pairs=self.online_pairs,online_switches=self.online_switches,online_boundary_calls=self.online_boundary_calls,
            motion_frames=self.motion_frames,contact_tracking='debounced_coast_and_reapproach',
            inference_frozen=self.inference_frozen,inference_end_time=self.inference_end_time,freeze_record=self.freeze_record,
            final_passage_certificate=getattr(self,'final_passage_certificate',None),
            record_scope='one continuous-world interaction; task success is in mission summary')
        (self.args.output/'interaction.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        self.mission.interaction_records[self.oid]=result

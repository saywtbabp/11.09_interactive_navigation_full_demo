"""Window-by-window inference and immediate contact-mode switching."""
import time
import numpy as np
from belief import Belief
from decision_rule import evaluate
from contact_boundary import online_metrics
from interaction_engine import IMPACT_GUARD,BOX_HALF,ARM,path_length
from local_hover import LocalHoverRecovery,deferred_egress,hover_planner,retreat_directions
from navigation.interactions import panel_geometry
from navigation.physics import clip_norm,PhysicsFailure
from probe_recovery import monotone_exit
from passage_completion import try_freeze,current_passage
from passage_transition import PassageConnectionRecovery
from contact_tracking import ContactTracking

class OnlineContact:
    def initialize_online(self):
        self.online_active=False;self.online_revision=0;self.online_pairs=[];self.online_switches=[]
        self.online_choice='identify';self.online_exit_latched=False;self.online_cursor=0
        self.online_row_stages=[];self.online_reason='';self.online_boundary_calls=0
        self.online_motion_finished=False;self.online_contact_onset=None;self._last_online_pose=None
        self.inference_frozen=False;self.inference_end_time=None;self.freeze_record=None;self.passage_certificate=None
        self.contact_tracking=None;self.flight_reference=None;self.motion_frames=[];self.next_motion_sample=0.
        self.contact_blend_time=None

    def consume_online_windows(self):
        if self.inference_frozen:return
        while self.rows is not None and len(self.rows)-self.online_cursor>=51:
            # Completion is checked BEFORE committing this new window or
            # calculating another pushing boundary. Keep the last true result.
            if try_freeze(self):return
            block=np.asarray(self.rows[self.online_cursor:self.online_cursor+51]);self.online_cursor+=50
            before=len(self.belief.windows);conflict=self.belief.conflict
            self.belief.update(self.local_rows(block))
            changed=len(self.belief.windows)!=before or self.belief.conflict!=conflict
            if changed:
                self.online_revision+=1
                self.pair_online_update('online_update',True,float(block[-1,0]))
            elif self.online_contact_onset is not None:
                pose=np.r_[self.position(),self.state()[0]*self.arm]
                if self._last_online_pose is None or np.linalg.norm(pose-self._last_online_pose)>.002:
                    self.pair_online_update('online_geometry',False,float(block[-1,0]))

    def pair_online_update(self,label,changed,evidence_time=None):
        now=float(self.world.data.time);started=time.perf_counter();error=None
        self.online_boundary_calls+=1
        try:metrics=online_metrics(self)
        except ValueError as exc:metrics={};error=str(exc)
        snap=self.belief.snapshot();snap.update(metrics)
        for name in ('push_limit','gain','stroke'):snap.setdefault(name,None)
        limit=snap['push_limit'];p,converged=self.belief.probability(limit) if limit is not None else (None,False)
        verdict=evaluate(p,converged,self.args.decision_rule,self.args.alpha,self.args.tie_margin)
        snap.update(verdict);snap.update(label=label,time=now,evidence_time=now if evidence_time is None else evidence_time,
            weight_revision=self.online_revision,weights_changed=changed,geometry_time=now,formal_decision=True,
            route_preview_status='paired_online_geometry',boundary_age_s=0.,push_probability=p,
            integration_converged=converged,boundary_error=error,update_wall_s=time.perf_counter()-started)
        choice='continue' if verdict['decision_push'] else 'avoid' if p is not None else 'identify'
        if self.belief.conflict:choice='avoid';snap['decision_reason']='inconsistent_mechanical_windows'
        if error is not None:choice='avoid';snap['decision_reason']='online_geometry_unavailable'
        if metrics.get('gain',1.)<=0:snap['decision_reason']='no_remaining_navigation_gain'
        if self.online_exit_latched:choice='avoid'
        snap['control_choice']=choice;snap['exit_latched']=self.online_exit_latched
        if self.online_motion_finished:choice='complete';snap['control_choice']=choice
        self._last_online_pose=np.r_[self.position(),self.state()[0]*self.arm]
        self.snapshots.append(snap);self.latest_evidence=self.belief
        self.online_pairs.append(dict(time=now,revision=self.online_revision,windows=len(self.belief.windows),
            weights_changed=changed,geometry_time=now,evidence_time=snap['evidence_time'],p_ok=p,F_star_N=limit,
            choice=choice,latched=self.online_exit_latched,wall_s=snap['update_wall_s'],error=error))
        m=self.mission;m.metrics=dict(G=snap['gain'],p_D=p,D=None,F_star=limit)
        if label in ('online_update','online_geometry'):
            m.decisions.append(dict(time_s=now,opening=self.oid,phase='online',semantic=self.kind,
                action={'continue':'Push','avoid':'Avoid','identify':'Identify','complete':'Cross'}[choice],p_ok=p,F_star_N=limit,
                gain_m=snap['gain'],windows=snap['windows'],weight_revision=self.online_revision,
                geometry_time=now,evidence_time=snap['evidence_time'],latched=self.online_exit_latched,
                weights_changed=changed,
                reason=snap['decision_reason'],posterior_source='incremental_nonoverlapping_windows'))
        if not self.online_exit_latched and not self.online_motion_finished:
            if choice!=self.online_choice:
                self.online_switches.append(dict(time=now,from_choice=self.online_choice,to_choice=choice,revision=self.online_revision))
                self.event('online_route_choice',route=choice,p_ok=p,F_star_N=limit,windows=snap['windows'],revision=self.online_revision)
            self.online_choice=choice
            if choice=='continue':
                self.committed=True;m.action='Push';m.status='在线多数支持 / 接触状态决定执行方式'
                if self.contact_tracking is not None and self.contact_tracking.state!='contact':
                    m.action='Observe';m.status='在线多数支持 / 已脱开，滑行观测'
                m.reason=f'第{self.online_revision}次参数更新：p={p:.1%}，F*={limit:.3f} N；下一有效窗口继续复算'
            elif choice=='avoid':
                self.online_exit_latched=True;self.online_reason=snap['decision_reason'];m.action='Avoid'
                m.status='停止额外推压 / 卸载后检查现有通道' if snap['decision_reason']=='no_remaining_navigation_gain' else '在线决定绕行 / 立即停止推压'
                m.reason=error or ('继续推压已无新增导航收益；立即卸载并绕开当前对象' if snap['decision_reason']=='no_remaining_navigation_gain' else f'第{self.online_revision}次参数更新不支持续推；开始卸载，安全退出后绕行')
            else:
                m.status='接触辨识 / 边界已算，支持率尚不可定义';m.action='Probe'
                m.reason='当前观测尚不能形成可归一化参数区域；不设置固定窗口数量门槛'
        return snap

    def run_online_contact(self):
        m,w=self.mission,self.world;self._preview_enabled=False;self.online_active=True
        self.online_choice='identify';self.online_exit_latched=False;self.online_cursor=0
        q0,v0=self.state();self.rows=[[float(w.data.time),q0,v0,0.,0.]];self.online_row_stages=['ONLINE_APPROACH']
        begin=float(w.data.time);onset=None;release=None;terminal=None;stall_since=None;continue_saved=False
        m.status='接触辨识 / 实时权重与分界';m.action='Probe'
        self.event('probe_choice',action='online_window_majority',approach_extra_m=self.precontact['approach_extra_m'],
            probe_work_budget_m=self.precontact['probe_budget_m'])
        while w.data.time-begin<30.:
            q,velocity=self.state();pos,vel,_,_=w.observed()
            if release is None and w.step_count%25==0:try_freeze(self)
            if onset is None and self.last_force>.08:
                onset=float(w.data.time);self.contact_started('PROBE')
                self.online_contact_onset=onset
                self.contact_tracking=ContactTracking(self)
                self.pair_online_update('online_contact_start',False)
            if release is None and self.contact_tracking is not None:self.contact_tracking.update()
            if release is None:
                reason=None
                if self.last_force>IMPACT_GUARD:reason='impact_guard'
                elif self.inference_frozen:reason='verified_passage'
                elif self.online_exit_latched:reason='online_avoid'
                elif onset is not None and self.online_choice=='identify' and w.data.time-onset>=1.4:reason='no_identifiable_probability'
                elif onset is None and w.data.time-begin>11.:reason='no_contact'
                elif onset is not None and w.data.time-onset>=15.:reason='online_contact_timeout'
                if onset is not None and abs(velocity)<.009 and self.last_force>.60*self.cap:
                    stall_since=w.data.time if stall_since is None else stall_since
                    if w.data.time-stall_since>=.80:reason='stall'
                else:stall_since=None
                if reason is not None:
                    terminal='complete' if reason=='verified_passage' else 'avoid'
                    self.online_motion_finished=terminal=='complete'
                    if terminal=='avoid':self.online_exit_latched=True;self.online_choice='avoid'
                    self.online_reason=self.online_reason if reason=='online_avoid' and self.online_reason else reason
                    release=float(w.data.time)
                    self.probe_recovery=PassageConnectionRecovery(self) if terminal=='complete' else LocalHoverRecovery(self)
                    self.event('online_release',reason=reason,terminal=terminal,revision=self.online_revision)
            if release is not None:
                w.stage='ONLINE_UNLOAD' if terminal=='avoid' else 'PASSAGE_RELEASE'
                try:force=self.probe_recovery.command()
                except ValueError as exc:self.probe_recovery_error=str(exc);break
                if self.probe_recovery.ready:break
                if w.data.time-release>self.probe_recovery.timeout:
                    self.probe_recovery_error='online recovery timeout';break
                w.step(clip_norm(force,self.cap),yaw=getattr(self.probe_recovery,'yaw_command',0.));continue
            if self.kind=='door':
                tangent=np.array([np.cos(q),np.sin(q)]);normal=np.array([-np.sin(q),np.cos(q)])
                point=panel_geometry(self.opening,q)[0]+ARM*tangent-.035*normal
            else:tangent,normal=np.eye(2);point=np.array([self.x,self.y+q-BOX_HALF[1]])
            if self.contact_tracking is not None and self.contact_tracking.state!='contact':
                force,heading=self.contact_tracking.command(point,tangent,normal)
                w.step(clip_norm(force,self.cap),yaw=heading);continue
            w.stage='ONLINE_CONTINUE' if self.online_choice=='continue' else 'ONLINE_IDENTIFY' if onset is not None else 'ONLINE_APPROACH'
            if self.online_choice=='continue' and not continue_saved:
                m.save_key(self.oid+'_online_continue');continue_saved=True
            w.target=np.r_[pos[:2],1.25]
            ft=4*float((point-pos[:2])@tangent)-3*float(vel[:2]@tangent)
            gap=float((point-pos[:2])@normal)
            if onset is None:
                speed=self.approach_speed(gap,pos,vel,'ONLINE')
                fn=w.mass*3.2*(speed-float(vel[:2]@normal))
            else:
                damping=2.5
                if self.online_choice=='continue':
                    # A freely swinging leaf must approach the clearance target
                    # slowly enough to release before reaching its hinge stop.
                    # Use measured remaining travel, never hidden inertia/drag.
                    ramp,gain=(.80,18.) if self.kind=='door' else (.42,6.)
                    damping+=gain*np.clip(1-(self.target-q)*self.arm/ramp,0.,1.)
                fn=self.cap-damping*float(vel[:2]@normal)
                # Negative flight thrust along the contact normal brakes the
                # UAV; clipping it to zero would let an airborne UAV coast into
                # a freely swinging door's stop despite the damping increase.
                fn=float(np.clip(fn,-self.cap,self.cap)) if self.kind=='door' and self.online_choice=='continue' else max(0.,fn)
            force=clip_norm(ft*tangent+fn*normal,self.cap)
            if self.contact_blend_time is not None:
                fraction=float(np.clip((w.data.time-self.contact_blend_time)/.30,0.,1.));blend=fraction*fraction*(3-2*fraction)
                force=(1-blend)*self.contact_blend_force+blend*force
                if fraction>=1.:self.contact_blend_time=None
            w.step(force,yaw=q if self.kind=='door' else 0.)
        if release is None:
            terminal='avoid';self.online_exit_latched=True;self.online_reason='online_motion_timeout'
        # No post-action probability gate. Recovery only establishes safe flight.
        self.recover_probe();self.observe_state()
        a=np.asarray(self.rows);self.rows=None;self.online_active=False
        self.trace.extend([*row,stage] for row,stage in zip(a.tolist(),self.online_row_stages))
        self.event('online_contact_end',terminal=terminal,reason=self.online_reason,windows=len(self.belief.windows),
            first_contact_time=onset,release_time=release)
        # Stopping additional push is not a commitment to fly back to another
        # aperture. A free-swinging leaf may finish opening during safe release.
        # Reuse that observed opening without recontact, after all safety gates.
        if terminal=='avoid' and self.kind=='door' and self.committed and self.online_reason=='no_remaining_navigation_gain':
            q,velocity=self.state()
            if q>=self.target and abs(velocity)<=.05:
                terminal='complete'
                self.event('clearance_achieved_during_release',observed_q=q,target_q=self.target,
                    reason='no_further_contact_needed',prior_stop_reason=self.online_reason)
        if terminal=='complete':
            self.finish_online_crossing();return
        self.event('route_choice',route='bypass',reason=self.online_reason,selected_during_contact=True)
        prefix=deferred_egress(self,self.position(),self.state()[0])
        if len(prefix)>1:
            m.status='已在线选择绕行 / 安全退出';m.action='Avoid'
            self.event('bypass_exit_start',distance_m=path_length(prefix))
            w.fly(prefix[1:],yaw=0.,speed=.15,stage='BYPASS_EXIT',position_tolerance=.02);self.event('bypass_exit_end')
        self.record.bypass_selected=True;self.outcome='probe_then_bypass';self.completed=True
        m.save_key(self.oid+'_online_avoid')

    def finish_online_crossing(self):
        m,w=self.mission,self.world;q,v=self.state()
        if (q<self.target and not self.inference_frozen) or abs(v)>.05:
            self.record.bypass_selected=True;self.outcome='push_failed_then_bypass';self.completed=True
            self.event('fallback',reason='online target clearance not stable');return
        if self.inference_frozen:
            # The frozen picture is history. Execution still needs a NEW route
            # certificate for the latest measured geometry after safe release.
            for attempt in range(3):
                try:self.final_passage_certificate=current_passage(self)
                except ValueError as exc:raise PhysicsFailure('frozen completion lost actual passage: '+str(exc))
                connector=np.array(self.final_passage_certificate['release_path'])
                if path_length(connector)<1e-6:break
                if isinstance(self.probe_recovery,PassageConnectionRecovery):
                    # A changed measured scene sends us back through the SAME
                    # state-gated transition, never a separate hover/connector.
                    self.event('passage_connection_recheck_required',attempt=attempt+1)
                    self.probe_recovery.ready=False;self.probe_recovery.stable_since=None
                    self.probe_recovery.next_check=0.
                    self.recover_probe();continue
                # Execute the connector in the motion-margin certificate, even
                # when a less conservative nominal planner would skip it.
                self.paths['CROSSING_CONNECTOR']=connector.tolist()
                self.event('certified_crossing_connector',path=connector.tolist(),distance_m=path_length(connector))
                w.fly(connector[1:],yaw=0.,speed=.15,stage='CROSSING_CONNECTOR',position_tolerance=.02)
            else:raise PhysicsFailure('cannot attach actual pose to certified passage after local release')
        planner=self.planner(q,'target');start=self.position();local=hover_planner(self,q)
        if planner.collides(start)[0]:
            connector=None
            for distance in np.arange(.04,1.25,.04):
                for direction in retreat_directions(self,start,q):
                    end=start+distance*direction
                    if planner.collides(end)[0] or not local.clear(start,end) or not monotone_exit(planner,start,end):continue
                    try:planner.plan(end,self.goal)
                    except ValueError:continue
                    connector=np.array([start,end]);break
                if connector is not None:break
            if connector is None:raise PhysicsFailure('no target-side flight-envelope connector after online push')
            w.fly(connector[1:],yaw=0.,speed=.15,stage='CROSSING_CONNECTOR',position_tolerance=.02)
        full=np.array(self.final_passage_certificate['passage_path']) if self.inference_frozen else self.planner(q,'target').plan(self.position(),self.goal)[0]
        path=[full[0]];cut=self.y+1.70
        for a,b in zip(full,full[1:]):
            if a[1]<cut<=b[1]:path.append(a+(b-a)*(cut-a[1])/(b[1]-a[1]));break
            path.append(b)
            if b[1]>=cut:break
        self.record.traversability='free';self.record.verification_state='verified_free';self.record.confirmation='confirmed'
        m.status='在线续推完成 / 实际净空确认后穿越';m.action='Cross'
        self.event('clearance_verified',observed_q=q,path_m=path_length(np.array(path)),
            current_pose_passage_certified=self.inference_frozen,nominal_target_q=self.target)
        m.decisions.append(dict(time_s=float(w.data.time),opening=self.oid,phase='clearance',semantic=self.kind,
            action='Cross',p_ok=None,F_star_N=None,reason='observed_stable_opening_and_actual_target_only_path',
            observed_q=q,target_q=self.target,geometry_time=float(w.data.time)))
        self.fly(np.array(path),'SHORTCUT');self.crossed=True;self.outcome='shortcut';self.completed=True
        m.save_key(self.oid+'_online_crossed')

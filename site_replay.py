"""Continuous, recorded-position site replay for the multi-wall mission.

No dynamics are re-simulated here. Object q, aircraft pose, phase and paths are
read from the same saved mission as the posterior and sensor panels.
"""


def clips_for(data):
    frames=data['frames'];clips=[]
    for oid,result in data['interactions'].items():
        if not result['probed'] or not result['frames']:continue
        opening=next(o for o in data['map']['openings'] if o['id']==oid)
        current=[f['time'] for f in frames if f['current']==oid]
        gates=[r['time_s'] for r in data['decisions'] if r['opening']==oid and r['phase']=='precontact']
        start=min(current+gates+[result['frames'][0][0]])
        finish=result['frames'][-1][0]
        y=opening['center_xy_m'][1]
        # The A01 contact session ends before the flight through F04. Include
        # that ACTUAL subsequent flight; a contact-only trace would hide it.
        after=finish if result['frames'][-1][2]>y+.60 else next((f['time'] for f in frames if f['time']>=finish and f['position'][1]>y+.60),frames[-1]['time'])
        end=max(finish,after)
        selected=[f for f in frames if start-.001<=f['time']<=end+.001]
        if not selected:continue
        start,end=selected[0]['time'],selected[-1]['time']
        points=[f['position'][:2] for f in selected]
        x,y=opening['center_xy_m'];width=opening['width_m']-.18
        points += [[x-width/2,y-.45],[x+width/2,y+max(1.3,result['target_q'] if result['object_kind']=='box' else width)]]
        leg=result.get('local_leg')
        if leg:points.append(leg['goal_xy'])
        bounds=[min(p[0] for p in points)-.55,max(p[0] for p in points)+.55,
                min(p[1] for p in points)-.50,max(p[1] for p in points)+.50]
        markers=[]
        onset=next((e.get('first_contact_time') for e in result['events'] if e['event']=='online_contact_end'),None)
        if onset is not None:
            row=min(result['frames'],key=lambda r:abs(r[0]-onset))
            markers.append(dict(time=onset,label='首次接触 / 辨识',xy=row[1:3],sample_time=row[0]))
        for event in result['events']:
            key=event['event'];label=None
            if key=='online_route_choice':label={'continue':'支持续推','avoid':'决定绕行'}.get(event['route'])
            elif key=='online_release':label='开始卸载'
            elif key=='passage_inference_frozen':label='净空具备 / 冻结'
            elif key=='clearance_verified':label='开始穿越'
            elif key=='probe_recovery_ready':label='安全状态通过'
            elif key=='contact_lost':label='脱开 / 滑行跟踪'
            elif key=='contact_reacquired':label='恢复接触'
            if label:
                row=min(result['frames'],key=lambda r:abs(r[0]-event['time']))
                markers.append(dict(time=event['time'],label=label,xy=row[1:3],sample_time=row[0]))
        # Name the aperture actually crossed, inferred from the logged route.
        crossed=[]
        for a,b in zip(selected,selected[1:]):
            ay,by=a['position'][1],b['position'][1]
            if ay<y<=by:
                cross_x=a['position'][0]+(b['position'][0]-a['position'][0])*(y-ay)/(by-ay)
                hit=next((o['id'] for o in data['map']['openings'] if abs(o['center_xy_m'][1]-y)<.01 and abs(cross_x-o['center_xy_m'][0])<o['width_m']/2),None)
                if hit and hit not in crossed:crossed.append(hit)
        markers.append(dict(time=end,label='本段通行完成'+(' / '+', '.join(crossed) if crossed else ''),xy=selected[-1]['position'][:2],sample_time=end))
        clips.append(dict(id=oid,title=('箱子' if result['object_kind']=='box' else '第一扇门' if not any(c['kind']=='door' for c in clips) else '第二扇门')+' '+oid,
            start=start,end=end,kind=result['object_kind'],opening=opening,envelope=result['envelope_m'],
            bounds=bounds,markers=markers,crossed=crossed,local_leg=leg,
            connection_samples=[{k:r[k] for k in ('time','connection_target_xy','ready')}
                for r in result.get('recovery_frames',[]) if r.get('recovery_policy')=='direct_passage_connection'],
            motion_samples=result.get('motion_frames',[]),
            q_samples=[[r[0],r[4]] for r in result['frames']],planned=result['paths'],outcome=result['outcome']))
    return clips


HTML=r'''<section class="card" id="site-replay" style="margin:15px 0">
<h2>连续现场回放 · 无人机、门箱运动与实际路线</h2>
<div class="controls"><label>回放范围 <select id="site-clip" aria-label="现场回放对象"><option value="mission">整段任务</option></select></label>
<button id="site-play" class="primary">从头播放本段</button><button id="site-prev">上一帧</button><button id="site-next">下一帧</button>
<label>视野 <select id="site-view" aria-label="现场回放视野"><option value="leg">局部整段</option><option value="contact">接触近景</option><option value="map">全图</option></select></label>
<label><input id="site-full-trail" type="checkbox" checked style="width:auto">淡色显示本段完整实飞路线</label>
<label><input id="site-plans" type="checkbox" style="width:auto">显示比较预案（虚线）</label></div>
<div id="site-segment" class="legend"></div><div id="site-phases" class="jumps" style="margin:8px 0"></div>
<svg id="site-map" viewBox="0 0 900 650" style="height:min(61vw,650px);min-height:320px;background:#0d1b26;border-radius:7px" aria-label="连续俯视现场及真实飞行路线"></svg>
<div id="site-readout" class="legend" style="margin-top:8px"></div>
<svg id="site-speed" viewBox="0 0 900 155" style="margin-top:10px" aria-label="交互阶段实际速度与连续速度参考"></svg>
<div id="site-motion-readout" class="legend"></div>
<div class="legend">蓝：接近　紫：辨识推压　绿：接触续推　蓝紫：脱开滑行/低速重接触　黄：制动调姿　浅青：接入通行路线　橙：绕行　青：穿越。黄色空心圆为当前通行连接点，点线表示目标方向。速度图实线为实测速度，虚线为连续速度参考。现场与右侧权重图共用同一时间轴；白框为巡航包络。</div>
</section>'''


SCRIPT=r'''
(()=>{
 const clips=d.site_replays??[],sel=el('site-clip'),view=el('site-view');let active=null;
 for(const c of clips){const o=document.createElement('option');o.value=c.id;o.textContent=c.title+' · '+(c.outcome==='shortcut'?'穿越':'绕行');sel.append(o)}
 const colorFor=(r,c)=>{
  const stage=r.stage;
  if(/COAST|REAPPROACH/.test(stage))return '#93a5ff';
  if(stage==='PASSAGE_CONNECT')return '#93d6d3';
  if(/UNLOAD|HOVER|AFTER_PROBE|CONNECTOR|PASSAGE_RELEASE/.test(stage))return '#ffd979';
  if(/BYPASS/.test(stage))return '#f4a261';
  if(/CONTINUE|^PUSH$/.test(stage))return '#48d5aa';
  if(/IDENTIFY|^PROBE$/.test(stage))return '#b29aff';
  if(/SHORTCUT|CRUISE_CROSSING/.test(stage))return '#76ede7';
  const avoid=c?.markers.find(m=>m.label==='决定绕行');
  if(avoid&&r.time>=avoid.time&&!/VERIFY/.test(stage))return '#f4a261';
  return '#78b7fa';
 };
 function seek(t){playing=false;el('play').textContent='播放';const next=f.findIndex(r=>r.time>=t-.001),index=next<0?f.length-1:next;
  slider.value=index;playTime=f[index].time;lastTick=performance.now();painted=-1;draw(index)}
 function activate(id,play=false){active=clips.find(c=>c.id===id)??null;sel.value=active?.id??'mission';
  if(active){manual=true;selectObject(active.id);view.value='leg'}else{manual=false;view.value='map'}
  seek(active?.start??f[0].time);
  if(play){playing=true;el('play').textContent='暂停';lastTick=performance.now()}
 }
 sel.onchange=()=>activate(sel.value);
 el('site-play').onclick=()=>activate(sel.value,true);
 for(const [id,offset] of [['site-prev',-1],['site-next',1]])el(id).onclick=()=>{
  const lo=active?.start??f[0].time,hi=active?.end??f.at(-1).time;
  const index=Math.max(0,Math.min(f.length-1,Number(slider.value)+offset));seek(Math.max(lo,Math.min(hi,f[index].time)));
 };
 for(const input of [view,el('site-full-trail'),el('site-plans')])input.onchange=()=>window.drawSiteReplay(Number(slider.value));
 window.siteReplayRange=()=>active;
 const escape=esc;
 const label=(x,y,t,attrs='')=>`<text x="${x}" y="${y}" fill="#bbd1de" font-size="12" ${attrs}>${escape(t)}</text>`;
 window.drawSiteReplay=i=>{
  const r=f[i],t=r.time;
  // A jump on the shared timeline must not leave an unrelated clip selected.
  if(active&&(t<active.start-.01||t>active.end+.11)){
   active=clips.find(c=>c.start-.01<=t&&t<=c.end+.11)??null;sel.value=active?.id??'mission';manual=!!active;
   if(!active)view.value='map';
   if(active)selectObject(active.id);else if(d.evidence[r.current])selectObject(r.current);
  }
  const focus=active??clips.find(c=>c.start<=t&&t<=c.end)??null;
  const range=active??{start:f[0].time,end:f.at(-1).time};
  const rows=f.filter(a=>a.time>=range.start-.001&&a.time<=range.end+.001);
  let ext=[bounds.xmin,bounds.xmax,bounds.ymin,bounds.ymax];
  if(focus&&view.value==='leg')ext=focus.bounds;
  if(focus&&view.value==='contact'){const [x,y]=focus.opening.center_xy_m;ext=[x-1.75,x+1.75,y-1.75,y+2.0]}
  const [xmin,xmax,ymin,ymax]=ext,W=900,H=650,pad=32;
  const scale=Math.min((W-2*pad)/(xmax-xmin),(H-2*pad)/(ymax-ymin));
  const ox=(W-(xmax-xmin)*scale)/2,oy=(H-(ymax-ymin)*scale)/2;
  const X=x=>ox+(x-xmin)*scale,Y=y=>H-oy-(y-ymin)*scale;
  const line=(a,b,c,width=2,extra='')=>`<line x1="${X(a[0])}" y1="${Y(a[1])}" x2="${X(b[0])}" y2="${Y(b[1])}" stroke="${c}" stroke-width="${width}" ${extra}/>`;
  const route=(points,c,width=2,attrs='')=>`<polyline points="${points.map(p=>X(p[0])+','+Y(p[1])).join(' ')}" fill="none" stroke="${c}" stroke-width="${width}" stroke-linejoin="round" ${attrs}/>`;
  const box=(x,y,hx,hy,c,extra='')=>`<rect x="${X(x-hx)}" y="${Y(y+hy)}" width="${2*hx*scale}" height="${2*hy*scale}" fill="${c}" ${extra}/>`;
  let svg=`<defs><clipPath id="site-viewport"><rect width="900" height="650"/></clipPath></defs><g clip-path="url(#site-viewport)"><rect width="900" height="650" fill="#0d1b26"/>`;
  const grid=Math.max(xmax-xmin,ymax-ymin)<5?.5:1;
  for(let x=Math.ceil(xmin/grid)*grid;x<=xmax;x+=grid){svg+=line([x,ymin],[x,ymax],'#233847',1);svg+=label(X(x),H-8,fmt(x,grid<1?1:0),'text-anchor="middle"')}
  for(let y=Math.ceil(ymin/grid)*grid;y<=ymax;y+=grid){svg+=line([xmin,y],[xmax,y],'#233847',1);svg+=label(5,Y(y)+4,fmt(y,grid<1?1:0))}
  const thick=d.map.wall_thickness_m/2;
  for(const wall of d.map.internal_walls){let prev=bounds.xmin;
   for(const o of d.map.openings.filter(o=>Math.abs(o.center_xy_m[1]-wall.y_m)<.01).sort((a,b)=>a.center_xy_m[0]-b.center_xy_m[0])){
    const [x,y]=o.center_xy_m,h=o.width_m/2;
    svg+=box((prev+x-h)/2,y,(x-h-prev)/2,thick,'#8b9da7');prev=x+h;
    for(const sign of [-1,1])svg+=box(x+sign*(h-.045),y,.045,.085,'#c1d0d5');
    svg+=label(X(x),Y(y)-12,o.id,'text-anchor="middle"');
   }svg+=box((prev+bounds.xmax)/2,wall.y_m,(bounds.xmax-prev)/2,thick,'#8b9da7');
  }
  svg+=box(bounds.xmin,(bounds.ymin+bounds.ymax)/2,thick,(bounds.ymax-bounds.ymin)/2,'#8b9da7');
  svg+=box(bounds.xmax,(bounds.ymin+bounds.ymax)/2,thick,(bounds.ymax-bounds.ymin)/2,'#8b9da7');
  for(const o of d.map.openings){
   const [x,y]=o.center_xy_m,state=r.map[o.id],record=clips.find(c=>c.id===o.id);
   const sample=record?.q_samples.findLast(a=>a[0]<=t+.001);
   const q=sample?.[1]??(state?.semantic==='box'?state.linear_position:state?.angle??0);
   const kind=record?.kind??state?.semantic,width=o.width_m-.18;
   // A tentative class on an already verified EMPTY aperture is not a
   // physical object. F04, for example, must remain visibly open on the detour.
   if(!record&&state?.traversability==='free')continue;
   if(kind==='door'){
    const pivot=[x-width/2,y+.035],c=Math.cos(q),s=Math.sin(q);
    const points=[[0,-.06],[width,-.06],[width,-.01],[0,-.01]].map(([u,v])=>[pivot[0]+c*u-s*v,pivot[1]+s*u+c*v]);
    svg+=`<polygon points="${points.map(p=>[X(p[0]),Y(p[1])].join(',')).join(' ')}" fill="#c58d56" stroke="#ffca8a" stroke-width="2" data-object="${o.id}" data-q="${q}"/>`;
    svg+=`<circle cx="${X(pivot[0])}" cy="${Y(pivot[1])}" r="4" fill="#ffca8a"/>`;
   }else if(kind==='box')svg+=box(x,y+q,width/2,.25,'#b47d47',`stroke="#ffca8a" stroke-width="2" data-object="${o.id}" data-q="${q}"`);
   else if(kind==='suspected_glass')svg+=box(x,y,width/2,.035,'#72cbe9','opacity=".65"');
  }
  if(el('site-plans').checked&&focus)for(const [key,c] of [['SHORTCUT_OPTION','#67bbb1'],['BYPASS_OPTION','#e3c06d']]){
   if(focus.planned[key])svg+=route(focus.planned[key],c,2,'stroke-dasharray="7 6" opacity=".6" data-route="comparison"');
  }
  if(el('site-full-trail').checked)svg+=route(rows.map(a=>a.position),'#bbd2dd',3,'opacity=".19" data-route="full-recorded"');
  let group=[],lastColor=null;const flush=()=>{if(group.length>1)svg+=route(group,lastColor,3,'data-route="actual-past"');group=[]};
  for(const a of rows){if(a.time>t+.001)break;const c=colorFor(a,active??clips.find(c=>c.start<=a.time&&a.time<=c.end));if(c!==lastColor&&group.length){const end=group.at(-1);flush();group=[end]}lastColor=c;group.push(a.position)}flush();
  const markers=focus?.markers??[];
  for(const m of markers.filter(m=>m.time<=t+.001))svg+=`<circle cx="${X(m.xy[0])}" cy="${Y(m.xy[1])}" r="4" fill="#ffe18a"><title>${escape(m.label)} ${fmt(m.time,2)} s</title></circle>`;
  const goal=focus?.local_leg?.goal_xy??r.local_leg?.goal_xy;
  if(goal){const [x,y]=goal;svg+=`<path d="M${X(x)},${Y(y)-6}l6,6l-6,6l-6,-6z" fill="#ffe18a"/>`+label(X(x)+10,Y(y),'局部汇合点')}
  const connection=/^PASSAGE_(RELEASE|CONNECT)$/.test(r.stage)?focus?.connection_samples?.findLast(a=>a.time<=t+.001):null;
  if(connection){const [tx,ty]=connection.connection_target_xy;
   svg+=line(r.position,[tx,ty],'#ffd979',1.5,'stroke-dasharray="2 5" data-route="connection-target"');
   svg+=`<circle cx="${X(tx)}" cy="${Y(ty)}" r="7" fill="none" stroke="#ffd979" stroke-width="2" data-connection-target="true" data-x="${tx}" data-y="${ty}"/>`+label(X(tx)+10,Y(ty)-10,'通行连接点');
  }
  const envelope=focus?.envelope??[.32,.32];const [x,y,z]=r.position;
  svg+=box(x,y,envelope[0],envelope[1],'none','stroke="white" stroke-width="1.4" opacity=".8" data-uav-envelope="true"');
  const arm=.21*scale,rotor=.065*scale;
  svg+=`<g transform="translate(${X(x)} ${Y(y)}) rotate(${-r.yaw*180/Math.PI})" data-uav="true" data-x="${x}" data-y="${y}"><path d="M${-arm},${-arm}L${arm},${arm}M${-arm},${arm}L${arm},${-arm}" stroke="#d9fff8" stroke-width="3"/>`;
  for(const a of [-1,1])for(const b of [-1,1])svg+=`<circle cx="${a*arm}" cy="${b*arm}" r="${rotor}" fill="#284c59" stroke="#9bfff0" stroke-width="1.5"/>`;
  svg+=`<circle r="${.07*scale}" fill="#e9fff9"/><path d="M0,0V${-.37*scale}m-5,7l5,-7l5,7" fill="none" stroke="#ffe18a" stroke-width="2"/></g></g>`;
  el('site-map').innerHTML=svg;el('site-map').dataset.time=String(t);el('site-map').dataset.clip=active?.id??'mission';
  const q=focus?.q_samples.findLast(a=>a[0]<=t+.001);
  el('site-readout').textContent=`实际记录 ${fmt(t,2)} s · 无人机 (${fmt(x,3)}, ${fmt(y,3)}, ${fmt(z,3)}) m · ${r.stage}`+(q?' · '+(focus.kind==='door'?'门开度 '+fmt(q[1]*180/Math.PI,1)+'°':'箱位移 '+fmt(q[1],3)+' m'):'');
  if(connection)el('site-readout').textContent+=' · 朝通行连接点制动、调整姿态；安全状态满足后接续穿越';
  const motions=focus?.motion_samples??[],motion=motions.findLast(a=>a.time<=t+.001);
  el('site-motion-readout').textContent=motion&&t-motion.time<=.12?`记录 ${fmt(motion.time,2)} s · 控制状态 ${motion.motion_mode} · 接触力 ${fmt(motion.contact_N,3)} N · 实际速度 ${fmt(motion.speed_mps,3)} m/s · 参考速度 ${fmt(Math.hypot(...motion.reference_velocity_xy),3)} m/s`:motion?'滑行/恢复控制已结束；速度图保留本段实际记录。':'';
  if(focus){
   const onset=focus.markers.find(m=>m.label.startsWith('首次接触'))?.time??focus.start;
   const finish=focus.markers.find(m=>m.label==='开始穿越'||m.label==='安全状态通过')?.time??focus.end;
   const lo=onset-.3,hi=finish+.4,samples=rows.filter(a=>lo<=a.time&&a.time<=hi);
   const m=motions.filter(a=>lo<=a.time&&a.time<=hi),top=Math.max(.25,...samples.map(a=>a.speed_mps??0),...m.map(a=>Math.hypot(...a.reference_velocity_xy)))*1.1;
   const sx=a=>40+840*(a-lo)/(hi-lo),sy=a=>125-95*a/top;
   const curve=(points,c,dash='')=>`<polyline points="${points.map(a=>sx(a[0])+','+sy(a[1])).join(' ')}" fill="none" stroke="${c}" stroke-width="2" ${dash}/>`;
   el('site-speed').innerHTML=`<rect width="900" height="155" fill="#0d1b26"/><text x="40" y="17" fill="#b4c9d4" font-size="12">交互速度：实测 / 连续参考 · ${fmt(top,2)} m/s</text><line x1="40" y1="125" x2="880" y2="125" stroke="#526d7e"/>`+curve(samples.map(a=>[a.time,a.speed_mps??0]),'#76ede7')+curve(m.map(a=>[a.time,Math.hypot(...a.reference_velocity_xy)]),'#ffe18a','stroke-dasharray="6 4"')+`<line x1="${sx(Math.max(lo,Math.min(hi,t)))}" y1="23" x2="${sx(Math.max(lo,Math.min(hi,t)))}" y2="125" stroke="white"/><text x="40" y="146" fill="#b4c9d4" font-size="12">${fmt(lo,2)} s</text><text x="800" y="146" fill="#b4c9d4" font-size="12">${fmt(hi,2)} s</text>`;
  }else el('site-speed').innerHTML='';
  if(view.value==='contact'&&(x<xmin||x>xmax||y<ymin||y>ymax))el('site-readout').textContent+=' · 已离开接触近景，切换“局部整段”查看后续路线';
  el('site-segment').textContent=active?`${active.title}：${fmt(active.start,2)}–${fmt(active.end,2)} s · 实际经过 ${active.crossed.join(' → ')||'见记录轨迹'}。选择本段后，播放在本段末尾暂停；总时间轴仍可跳转全任务。`:'整段任务：可选择两扇门或箱子的完整接近、交互与后续通行路线。';
  const key=focus?.id??'';
  if(el('site-phases').dataset.object!==key){el('site-phases').dataset.object=key;el('site-phases').innerHTML='';
   for(const m of markers){const button=document.createElement('button');button.textContent=m.label+' '+fmt(m.time,2)+'s';button.onclick=()=>seek(m.time);el('site-phases').append(button)}
  }
 };
 view.value='map';
 if(!location.hash&&clips.length){active=clips[0];sel.value=active.id;view.value='leg';manual=true;selectObject(active.id)}
})();
'''

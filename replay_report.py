# -*- coding: utf-8 -*-
"""Self-contained mission replay with actual recorded RGB, map and evidence.

Run: python replay_report.py results/<run>. No server or internet is required.
"""
import argparse,base64,json
from pathlib import Path
from interaction_report import PAGE as EVIDENCE_PAGE
from site_replay import clips_for,HTML as SITE_HTML,SCRIPT as SITE_SCRIPT

def data_url(path):
    mime='image/jpeg' if path.suffix.lower() in ('.jpg','.jpeg') else 'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')

def evidence_html(result):
    # Reuse the verified 11.08 constraint-band renderer. Only the parameter and
    # boundary-history sections are exposed; the old one-wall map is hidden.
    css='''<style>body{background:#13232f}main{padding:8px;max-width:none}
    main>small,main>h1,#sub,#stats,main>section,.grid>section{display:none!important}
    .grid{display:block}.grid>div>section{display:none}
    .grid>div>section:nth-child(2),.grid>div>section:nth-child(3){display:block;margin:0 0 8px!important}
    .card{border:0;padding:8px;max-width:850px}h2{font-size:16px}.legend{font-size:12px}</style>'''
    hook='''<script>function externalSeek(t){
      let notice=el('clearance-note');if(!notice){notice=document.createElement('div');notice.id='clearance-note';notice.className='legend';notice.style.color='#64dfb4';el('limits').after(notice)}notice.textContent='';
      const i=d.snapshots.findLastIndex(b=>b.time<=t+.001);
      if(i<0){shown=-1;el('posterior').innerHTML=s(35,80,'此时尚未进行该物体的力学试探');el('limits').textContent='尚无力学先验 / 概率未定义';return}
      drawPosterior(i);drawBoundaryHistory(t);
      const passed=d.events.find(e=>e.event==='clearance_verified'&&e.time<=t+.001);
      if(passed)notice.textContent='实际开度、稳定与净空已通过，执行穿越；图中p_ok判断是否还要推压，不是已经打开通道的穿越许可。';
    }window.addEventListener('message',e=>{if(e.data?.type==='seek')externalSeek(Number(e.data.time))});
    const hashTime=Number(new URLSearchParams(location.hash.slice(1)).get('t'));
    if(location.hash)externalSeek(hashTime);</script>'''
    compact=dict(result);compact['trace']=result['trace'][::5];compact['sensor_frames']=[]
    payload=json.dumps(compact,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    error_hook="<script>window.addEventListener('error',e=>{document.body.dataset.jsError=e.message+' at '+e.lineno});</script>"
    return EVIDENCE_PAGE.replace('__DATA__',payload).replace('</head>',css+'</head>').replace('<script>',error_hook+'<script>',1).replace('</body>',hook+'</body>')

def build(folder):
    folder=Path(folder);d=json.loads((folder/'replay.json').read_text(encoding='utf-8'))
    d['site_replays']=clips_for(d)
    telemetry=d.pop('telemetry',None);d.pop('map_history',None)
    # Backfill display speed from the actual recorded velocity channels for
    # runs made before speed was also copied into the compact frame record.
    if telemetry:
        cursor=0
        for frame in d['frames']:
            while cursor+1<len(telemetry) and telemetry[cursor+1][0]<=frame['time']+1e-8:cursor+=1
            frame.setdefault('speed_mps',sum(float(v)**2 for v in telemetry[cursor][5:8])**.5)
            frame['vertical_speed_mps']=telemetry[cursor][7]
    for item in d['camera_frames']:item['image']=data_url(folder/item['file'])
    keyfile=folder/'keyframes.json';d['keyframes']=json.loads(keyfile.read_text(encoding='utf-8')) if keyfile.exists() else []
    for item in d['keyframes']:
        item['image']=data_url(folder/item['scene']);item['dashboard']=data_url(folder/item['file'])
    evidence={};d['raw_sensors']=[]
    for oid,result in d['interactions'].items():
        d['raw_sensors'].extend(dict(time=s['time'],stage=s['stage'],object=oid,readings=s['readings']) for s in result['sensor_frames'][::2])
        html=evidence_html(result);(folder/'interactions'/oid/'evidence.html').write_text(html,encoding='utf-8')
        evidence[oid]='data:text/html;base64,'+base64.b64encode(html.encode('utf-8')).decode('ascii')
        # Avoid duplicating the dense sensor data and parameter arrays in parent.
        d['interactions'][oid]={k:result[k] for k in ('outcome','object_kind','work_J','contact_peak_N','probed','committed','shortcut','events')}
        d['interactions'][oid].update({k:result.get(k) for k in ('inference_frozen','freeze_record','local_leg')})
    d['evidence']=evidence
    d['raw_sensors'].sort(key=lambda s:s['time'])
    d['sensor_samples']=[]
    sensorfile=folder/'sensors.jsonl'
    if sensorfile.exists():
        for line in sensorfile.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);ranges=row['distances_m'];valid=row['valid']
            good=[float(v) for v,ok in zip(ranges,valid) if ok]
            d['sensor_samples'].append(dict(time=row['time_s'],valid=sum(valid),rays=len(valid),minimum=min(good) if good else None,force=row['force_observer']))
    payload=json.dumps(d,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    html=PAGE.replace('__DATA__',payload)
    # Move the existing evidence panel beside the continuous scene. Keep ONE
    # iframe and one shared clock, rather than duplicating or reloading it.
    start=html.index('<section class="card"><h2>物体独立信念')
    end=html.index('</section>',start)+len('</section>')
    panel=html[start:end].replace('<section class="card">','<section class="card" id="evidence-panel">',1)
    panel=panel.replace('<iframe id="belief"','<div id="evidence-clock" class="legend"></div><iframe id="belief"',1)
    html=html[:start]+html[end:]
    html=html.replace('<div class="grid lower">','<div class="auxiliary-panels">',1)
    workbench='<div class="interaction-workbench">'+SITE_HTML+panel+'</div>'
    html=html.replace('<div class="grid"><section class="card"><h2>真实3D现场关键帧',workbench+'<div class="grid"><section class="card"><h2>真实3D现场关键帧',1)
    css='''<style>.twocol{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.twocol>*{min-width:0}td,th{overflow-wrap:anywhere}table{table-layout:fixed}
    .interaction-workbench{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(420px,1fr);gap:15px;align-items:start;margin:15px 0}
    .interaction-workbench>#site-replay{margin:0!important}.interaction-workbench>#evidence-panel{min-width:0}
    .interaction-workbench #site-map{height:auto!important;aspect-ratio:900/650;min-height:320px}
    .interaction-workbench #belief{height:900px}.auxiliary-panels>div{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:15px 0}
    .auxiliary-panels>div>section{margin-top:0!important}.auxiliary-panels>div>section:last-child{grid-column:1/-1}
    @media(max-width:1050px){.interaction-workbench,.auxiliary-panels>div{grid-template-columns:1fr}.auxiliary-panels>div>section:last-child{grid-column:auto}}</style>'''
    html=html.replace('</head>',css+'</head>',1)
    html=html.replace('支持则保持接触续推，不支持则卸载退出','支持时按实际接触状态续推或滑行跟踪，不支持则卸载退出')
    html=html.replace('const decisionJumps=',SITE_SCRIPT+'\nconst decisionJumps=',1)
    path=folder/'report.html';path.write_text(html,encoding='utf-8')
    (folder/'report_site_replay.html').write_text(html,encoding='utf-8')
    print('HTML_REPORT',path.resolve(),flush=True);return path

PAGE=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>INTERACT｜完整无人机导航回放</title><style>
*{box-sizing:border-box}body{margin:0;background:#0c1824;color:#e5f0f4;font:15px "Microsoft YaHei",system-ui,sans-serif}main{max-width:1780px;margin:auto;padding:25px}h1{font-size:32px;margin:8px 0}h2{font-size:18px;margin:0 0 10px}p{line-height:1.65}.muted,small{color:#9db6c6}.top{color:#75e3c7;letter-spacing:2px}.stats,.controls,.jumps{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.stats{margin:18px 0}.stat{padding:10px 19px;border-left:3px solid #75e3c7;background:#162937}.stat strong{display:block;font-size:23px;color:#75e3c7}.card{background:#132532;border:1px solid #304858;border-radius:10px;padding:16px;min-width:0}.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:15px;margin:15px 0}.lower{grid-template-columns:.8fr 1.2fr}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:10px}.camera{width:100%;display:block;aspect-ratio:1.5;object-fit:contain;background:#08121a}#scene{width:100%;aspect-ratio:1008/540;object-fit:contain;background:#08121a}.scene-wrap{position:relative}.scene-label{font-size:12px;color:#b5c9d5;margin:6px 0}.controls{padding:10px 0}button,select{background:#284758;color:#e5f0f4;border:1px solid #486778;border-radius:5px;padding:7px 13px;cursor:pointer}button.primary{background:#75e3c7;color:#10252c;font-weight:700}input{accent-color:#75e3c7}#slider{flex:1;min-width:200px}.jumps button{font-size:12px;padding:6px 9px}.green{color:#64dfb4}.orange{color:#f4a261}.yellow{color:#ffe18a}svg{width:100%;display:block}#map{height:490px}#belief{width:100%;height:850px;border:0;background:#13232f}.scroll{max-height:230px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:7px;border-bottom:1px solid #304858}#live{min-height:65px}#camera-time{min-height:40px}details{margin:12px 0}a{color:#75e3c7}.gallery{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.gallery img{width:100%;cursor:pointer}.legend{font-size:12px;line-height:1.6;color:#b4c9d4}#big{display:none;position:fixed;inset:0;background:#000e;z-index:10;padding:25px;cursor:zoom-out}#big img{width:100%;height:100%;object-fit:contain}@media(max-width:1050px){.grid,.lower{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main><div class="top">INTERACT / RECORDED CONTINUOUS MUJOCO MISSION</div><h1>从看到障碍，到交互决策，再到全局通行</h1>
<p class="muted">玻璃沿用10.31接触验证；门与导向箱采用11.08数据约束与多数权重。在线模式每次有效窗口同步更新边界并选择续推或退出；试探后决策模式保留作对照。同一物理世界连续执行，控制器不读取场景的质量/阻力真值。</p><div id="stats" class="stats"></div>
<section class="card"><div class="controls"><button id="play" class="primary">播放</button><input id="slider" type="range" min="0" value="0" step="1" aria-label="回放时间"><strong id="clock"></strong><select id="speed" aria-label="回放倍率"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select></div><div id="jumps" class="jumps"></div><p id="live"></p></section>
<div class="grid"><section class="card"><h2>真实3D现场关键帧</h2><div class="scene-wrap"><img id="scene" alt="MuJoCo真实场景关键帧"></div><div id="scene-time" class="scene-label"></div><div class="legend">3D画面为实际渲染的关键帧；连续位置与决策时序见右侧地图。工程实时窗口提供连续3D视图，HTML没有将关键帧伪称为逐帧视频。</div></section>
<section class="card"><h2>无人机的在线认知地图</h2><div class="twocol"><svg id="map" viewBox="0 0 400 570" aria-label="随观测更新的认知地图"></svg><div><div class="legend"><span class="green">绿：已验证净空</span>　红：已验证受阻<br>灰：未知　棕：已识别待验证<br>橙色×：已选择绕行（不等同于物理堵死）<br>墙体/开口几何已知；物体类别及通行状态在线获得。</div><table id="objects"></table></div></div></section></div>
<div class="grid lower"><div><section class="card"><h2>机载RGB / 三类YOLO真实推理</h2><canvas id="camera" class="camera" width="480" height="320"></canvas><div id="camera-time" class="legend"></div><div class="legend">检测分数未经实机概率校准。真实渲染RGB进入ONNX；物体ID由测距几何关联，隐藏场景标签不进入检测器。</div></section>
<section class="card" style="margin-top:15px"><h2>接触与本体传感器</h2><svg id="force" viewBox="0 0 600 210"></svg><div class="legend">橙：仿真合接触力（评估参考）；蓝：玻璃验证使用的IMU/AHRS/电机残差测力。门/箱的参数更新沿用11.08关节位姿与接触力仿真参考读数。</div><table id="readings"></table><details><summary>交互期间全部原始模型传感器 / 控制通道</summary><p class="legend">每0.1秒的实际记录，不插值；这是仿真原始参考通道，不能作为实机测量精度证明。</p><div class="scroll"><table id="raw-readings"></table></div></details></section>
<section class="card" style="margin-top:15px"><h2>全局启发式与交互后决策</h2><div id="decision-table" class="scroll"></div><p class="legend">接触前：无有限质量/阻力先验，只比较G₀与额外接近距离+名义试探成本。接触后：R&lt;F*为续推侧，p_ok&gt;0.5支持续推；无额外收益则停止施力。释放后已达到实际开度、稳定与净空要求时可穿越，无需再推；否则执行绕行。</p></section></div>
<section class="card"><h2>物体独立信念 / 完整约束带 / 决策分界</h2><div class="controls"><label>查看物体 <select id="object-select"></select></label><span id="evidence-label" class="legend"></span></div><iframe id="belief" title="约束交集、权重与分界回放"></iframe><div class="legend">白框：全部有效窗口交集；蓝紫：各窗口约束带；浅灰×：已排除、零权重；绿/橙：续推/停止交互侧。亮度表示相对权重。在线模式每个有效窗口同步更新权重、边界和动作，支持则保持接触续推，不支持则卸载退出。对照模式的后台预览另行标注采样时间与滞后。</div></section></div>
<section class="card"><h2>现场记录</h2><div id="gallery" class="gallery"></div></section>
<section class="card" style="margin-top:15px"><h2>事件日志（当前时刻以前）</h2><div class="scroll"><table id="events"></table></div></section>
<section class="card" style="margin-top:15px"><h2>本次验证的范围与复现</h2><p id="scope"></p><p>这是同一个地图中的功能展示，不是大量独立任务的性能统计。临界物体的阻力经过离线场景设计，以便展示多数分界两侧的行为；在线控制器只使用观测推断，未按物体ID指定续推/绕行动作。相机模型由合成场景训练，真实环境识别与测量不确定度仍需独立验证。箱体采用沿地面方向的一自由度滑动模型，门采用单转轴模型。</p><p>在PyCharm选择原Conda解释器后运行main.py；该HTML已内嵌数据和画面，可断网打开。结果JSON保存所有窗口、快照、传感器与决策；更改场景或任务代价后应重新运行，不能把本次结果当成新场景的保证。</p></section></main><div id="big"><img alt="放大现场记录"></div>
<script>
const d=__DATA__,el=id=>document.getElementById(id),f=d.frames;
const esc=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'),fmt=(x,n=2)=>x==null?'—':Number(x).toFixed(n);
const latest=(a,t)=>a.findLast(x=>x.time<=t+.001),semantic={unknown:'未知',door:'门',box:'箱',suspected_glass:'疑似玻璃'},states={unverified:'待验证',free:'净空',blocked:'受阻'};
const actionLabel=a=>a.phase==='completion'?'通行几何已满足，冻结推断':a.phase==='clearance'?'净空通过，穿越':a.action==='Avoid'&&a.reason==='no_remaining_navigation_gain'?'停止额外推压':({Push:'续推',Avoid:'绕行',Identify:'辨识',Cross:'穿越',Probe:'试探'})[a.action]??a.action;
const sum=d.summary;el('stats').innerHTML=[[sum.success?'到达目标':'未完成','任务结果'],[fmt(sum.simulated_time_s,1)+' s','仿真时间'],[sum.probes,'限力试探次数'],[fmt(sum.wall_contact_peak_N,3)+' N','非预期墙/框接触峰值'],['>50%','续推权重门限']].map(([v,k])=>`<div class="stat"><strong>${v}</strong><small>${k}</small></div>`).join('');
el('scope').textContent=`任务代价λ=${sum.lambda_m_per_J} m/J；测量源：RGB/YOLO + 匿名测距，玻璃使用原残差测力验证，门/箱使用仿真参考力学读数。仿真合接触力全程峰值${fmt(sum.contact_peak_N,3)} N（包含玻璃接触验证），并非指令限力。无墙/框碰撞不代表未接触玻璃或可动物体。`;
const slider=el('slider');slider.max=f.length-1;
if(sum.navigation_revision==='paired_window_online_control')slider.insertAdjacentHTML('afterend','<span class="legend">每窗：权重＋边界＋动作</span>');
el('live').insertAdjacentHTML('afterend','<div id="wait-checks" class="legend"></div>');
el('live').insertAdjacentHTML('afterend','<div id="local-horizon" class="legend"></div>');
const ss=el('object-select');for(const [oid,r] of Object.entries(d.interactions)){const o=document.createElement('option');o.value=oid;o.textContent=oid+' / '+semantic[r.object_kind];ss.append(o)}
let selected='',manual=false,currentTime=0;function selectObject(oid){if(!d.evidence[oid]||selected===oid)return;selected=oid;ss.value=oid;el('belief').src=d.evidence[oid];el('evidence-label').textContent=oid+' · '+d.interactions[oid].outcome;el('belief').onload=()=>seekEvidence(currentTime)}
function seekEvidence(t){el('belief').contentWindow?.postMessage({type:'seek',time:t},'*');const c=el('evidence-clock');if(c){c.textContent='与左侧现场同步：'+fmt(t,2)+' s';c.dataset.time=String(t)}}
ss.onchange=()=>{manual=true;selectObject(ss.value)};selectObject(ss.value);
const bounds=d.map.bounds_wall_centerlines_m,scale=Math.min(350/(bounds.xmax-bounds.xmin),520/(bounds.ymax-bounds.ymin));
const X=x=>25+(x-bounds.xmin)*scale,Y=y=>545-(y-bounds.ymin)*scale;
const line=(a,b,c,w=2,extra='')=>`<line x1="${X(a[0])}" y1="${Y(a[1])}" x2="${X(b[0])}" y2="${Y(b[1])}" stroke="${c}" stroke-width="${w}" ${extra}/>`;
let base=`<rect x="${X(bounds.xmin)}" y="${Y(bounds.ymax)}" width="${scale*(bounds.xmax-bounds.xmin)}" height="${scale*(bounds.ymax-bounds.ymin)}" fill="#0d1a23" stroke="#607988"/>`;
for(const wall of d.map.internal_walls){let prev=bounds.xmin;const openings=d.map.openings.filter(o=>Math.abs(o.center_xy_m[1]-wall.y_m)<.01).sort((a,b)=>a.center_xy_m[0]-b.center_xy_m[0]);for(const o of openings){const x=o.center_xy_m[0],h=o.width_m/2;base+=line([prev,wall.y_m],[x-h,wall.y_m],'#899da9',4);prev=x+h}base+=line([prev,wall.y_m],[bounds.xmax,wall.y_m],'#899da9',4)}
const g=d.map.goal_xyz_m;base+=`<circle cx="${X(g[0])}" cy="${Y(g[1])}" r="5" fill="#ffe18a"/>`;
const imgcache=new Map();function loaded(src,fn){let im=imgcache.get(src);if(!im){im=new Image();imgcache.set(src,im);im.src=src}if(im.complete)fn(im);else im.onload=()=>fn(im)}
let lastCam=null,lastScene=null;
const plotRows=f.filter((_,i)=>i%2===0),last=f.at(-1).time,maxForce=Math.max(1,...f.map(x=>x.contact_N));
const fx=t=>35+550*t/last,fy=v=>180-150*v/maxForce;
const poly=(rows,key,color)=>`<polyline points="${rows.map(r=>fx(r.time)+','+fy(key(r))).join(' ')}" stroke="${color}" fill="none" stroke-width="1.7"/>`;
const forceBase=`<line x1="35" y1="180" x2="585" y2="180" stroke="#78929f"/><text x="35" y="18" fill="#bfd2db" font-size="12">${fmt(maxForce)} N</text><text x="35" y="203" fill="#bfd2db" font-size="12">0 s</text><text x="540" y="203" fill="#bfd2db" font-size="12">${fmt(last,1)} s</text>`+poly(plotRows,r=>r.contact_N,'#f4a261')+poly(plotRows,r=>Math.hypot(...(r.force_estimate?.external_force_N??[0,0,0])),'#78b7fa');
let painted=-1;
function drawWaiting(r){const w=r.wait_status;if(!w){el('wait-checks').textContent='';return}const labels={detached:'脱离接触',speed_ok:'速度',gyro_ok:'角速度',tilt_ok:'倾角',yaw_ok:'航向',height_ok:'高度',envelope_ok:'机体包络',clearance:'局部净空',exit_path:w.recovery_policy==='direct_passage_connection'?'通行路径':'退出路径'};el('wait-checks').innerHTML=Object.entries(labels).map(([key,label])=>`<span class="${w[key]?'green':'orange'}" style="margin-right:12px">${label}${w[key]?' ✓':key==='exit_path'&&!w.exit_path_checked?' 待检查':' 等待'}</span>`).join('')+`<span>连续确认 ${fmt(w.dwell_s)} / 0.25 s</span>`}
function draw(i){if(i===painted)return;painted=i;const r=f[i],t=r.time;currentTime=t;el('clock').textContent=fmt(t,2)+' s';el('live').innerHTML=`<strong>${esc(r.status)}</strong> · ${esc(r.current)} · ${esc(r.action)} · ${esc(r.stage)}<br><span class="muted">${esc(r.reason)}</span>`;
 drawWaiting(r);
 const leg=r.local_leg;el('local-horizon').textContent=leg?(leg.row_y==null?'最后无交互路段 → 任务终点':'局部比较终点 ('+leg.goal_xy.map(v=>fmt(v,2)).join(', ')+') m；两条路线使用相同汇合点，未计入远处终点代价')+(r.speed_mps!=null?' · 当前速度 '+fmt(r.speed_mps,3)+' m/s':''):'';
 if(r.stage==='CRUISE_VERIFIED_EXIT')el('local-horizon').textContent+=
  ' · 高度 '+fmt(r.position[2],3)+' m · 垂直速度 '+fmt(r.vertical_speed_mps,3)+' m/s';
 const selectedOnline=d.decisions.findLast(a=>['online','completion','clearance'].includes(a.phase)&&a.opening===r.current&&a.time_s<=t+.001);if(selectedOnline)el('live').innerHTML+=`<br><span class="${selectedOnline.action==='Push'||selectedOnline.action==='Cross'?'green':selectedOnline.action==='Avoid'?'orange':'muted'}">在线交互选择：${esc(actionLabel(selectedOnline))}${selectedOnline.phase==='online'?' · 参数更新 #'+selectedOnline.weight_revision:selectedOnline.frozen_reference?' · 保留 '+fmt(selectedOnline.reference_time,2)+' s 的最后真实结果':' · 实际开度和路径验证'}${selectedOnline.latched?' · 正在完成安全退出':''}</span>`;
 let svg=base;for(const o of d.map.openings){const s=r.map[o.id],x=o.center_xy_m[0],y=o.center_xy_m[1],h=o.width_m/2;let c=s.traversability==='free'?'#64dfb4':s.traversability==='blocked'?'#ed8787':s.semantic==='unknown'?'#718c9e':s.semantic==='suspected_glass'?'#72cbe9':'#dfb272';svg+=line([x-h,y],[x+h,y],c,4)+`<text x="${X(x-h)}" y="${Y(y)-8}" fill="${c}" font-size="10">${o.id}</text>`;if(s.bypass_selected)svg+=`<text x="${X(x)}" y="${Y(y)+6}" fill="#f4a261" font-size="19">×</text>`;if(s.semantic==='door'&&s.angle>0){const px=x-.59125,py=y+.035;svg+=line([px,py],[px+1.1825*Math.cos(s.angle),py+1.1825*Math.sin(s.angle)],c,3)}if(s.semantic==='box'&&s.probe_used)svg+=`<rect x="${X(x-.59125)}" y="${Y(y+s.linear_position+.25)}" width="${1.1825*scale}" height="${.5*scale}" fill="none" stroke="${c}"/>`}
 svg+=`<polyline points="${f.slice(0,i+1).filter((_,j)=>j%3===0).map(a=>X(a.position[0])+','+Y(a.position[1])).join(' ')}" stroke="#75e3c7" stroke-width="2" fill="none"/>`;
 const [x,y]=r.position;svg+=`<circle cx="${X(x)}" cy="${Y(y)}" r="4" fill="white"/>`+line([x,y],[x-.35*Math.sin(r.yaw),y+.35*Math.cos(r.yaw)],'white',2);el('map').innerHTML=svg;
 if(leg){const [lx,ly]=leg.goal_xy;el('map').innerHTML+=`<path d="M${X(lx)},${Y(ly)-6}l6,6l-6,6l-6,-6z" fill="#ffe18a" data-local-goal="true"/><text x="${X(lx)+9}" y="${Y(ly)+3}" fill="#ffe18a" font-size="10">局部点</text>`}
 el('objects').innerHTML='<tr><th>ID</th><th>语义 / 通行</th></tr>'+Object.entries(r.map).map(([oid,s])=>`<tr><td>${oid}</td><td>${semantic[s.semantic]} / ${states[s.traversability]}${s.bypass_selected?' / 绕行':''}</td></tr>`).join('');
 const cam=latest(d.camera_frames,t);if(cam&&cam!==lastCam){lastCam=cam;loaded(cam.image,im=>{if(lastCam!==cam)return;const c=el('camera').getContext('2d');c.drawImage(im,0,0,480,320);c.font='13px sans-serif';for(const box of cam.detections){if(Math.abs(box.time_s-cam.time)>.15)continue;const [x1,y1,x2,y2]=box.bbox_xyxy;c.strokeStyle=box.class_name==='suspected_glass'?'#72cbe9':'#ffca8a';c.lineWidth=2;c.strokeRect(x1,y1,x2-x1,y2-y1);c.fillStyle=c.strokeStyle;c.fillText((box.stable_id||'')+' '+box.class_name+' '+fmt(box.score),x1+3,Math.max(14,y1-4))}});el('camera-time').textContent=`曝光 ${fmt(cam.time)} s；当前回放与曝光相差 ${fmt(t-cam.time)} s。显示该曝光对应的检测框。`}
 const key=latest(d.keyframes,t);if(key&&key!==lastScene){lastScene=key;el('scene').src=key.image;el('scene-time').textContent=key.label+' / 真实记录 '+fmt(key.time)+' s';el('scene').onclick=()=>enlarge(key.dashboard)}
 el('force').innerHTML=forceBase+`<line x1="${fx(t)}" y1="22" x2="${fx(t)}" y2="185" stroke="white"/>`;
 const sample=latest(d.sensor_samples,t);el('readings').innerHTML=[['仿真机体位置 xyz / m',r.position.map(v=>fmt(v,3)).join(', ')],['当前合接触力（参考）',fmt(r.contact_N,3)+' N'],['有效测距 / 总射线',sample?sample.valid+' / '+sample.rays:'—'],['最近有效测距',fmt(sample?.minimum,3)+' m'],['残差测力读数',esc(JSON.stringify(r.force_estimate))]].map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
 const raw=latest(d.raw_sensors,t);el('raw-readings').innerHTML=raw&&t-raw.time<.16?`<tr><th>曝光时刻 ${fmt(raw.time)} s / ${raw.object}</th><th>原始参考值</th></tr>`+Object.entries(raw.readings).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${Array.isArray(v)?v.map(x=>fmt(x,5)).join(', '):fmt(v,5)}</td></tr>`).join(''):'<tr><td>当前不在门/箱交互记录区间；上方位置、测距和残差测力仍可查看。</td></tr>';
 const decisions=d.decisions.filter(x=>x.time_s<=t+.001);el('decision-table').innerHTML='<table><tr><th>时间 / 物体</th><th>依据</th><th>动作</th></tr>'+decisions.map(a=>`<tr><td>${fmt(a.time_s,2)} / ${a.opening}${a.phase==='online'?'<br>在线 #'+a.weight_revision+(a.weights_changed?' 权重+边界':' 几何刷新'):''}</td><td>${a.phase==='precontact'?'G₀='+fmt(a.G0_m)+' >? '+fmt(a.approach_extra_m+a.probe_budget_m)+' m':a.phase==='clearance'?'实际开度 '+fmt(a.observed_q*180/Math.PI,1)+'°；净空及穿越路径通过':'p='+fmt(a.p_ok,4)+'；F*='+fmt(a.F_star_N)+' N'}${a.latched?'<br>已选择退出':''}</td><td class="${a.action==='Push'||a.action==='Cross'?'green':a.action==='Avoid'?'orange':'muted'}">${esc(actionLabel(a))}</td></tr>`).join('')+'</table>';
 const events=d.events.filter(x=>x.time_s<=t+.001).slice(-70);el('events').innerHTML='<tr><th>时间</th><th>对象 / 事件</th><th>内容</th></tr>'+events.map(a=>`<tr><td>${fmt(a.time_s,2)}</td><td>${esc(a.stable_id)} / ${esc(a.event)}</td><td>${esc(JSON.stringify(Object.fromEntries(Object.entries(a).filter(([k])=>!['time_s','stable_id','event'].includes(k)))))}</td></tr>`).join('');
 if(!manual&&d.evidence[r.current])selectObject(r.current);seekEvidence(t);
 if(window.drawSiteReplay)window.drawSiteReplay(i);
}
slider.oninput=()=>{playTime=f[Number(slider.value)].time;lastTick=performance.now();draw(Number(slider.value))};let playing=false,lastTick=0,playTime=0;
el('play').onclick=()=>{playing=!playing;if(playing){const clip=window.siteReplayRange?.();if(clip&&f[Number(slider.value)].time>=clip.end-.01){slider.value=Math.max(0,f.findIndex(r=>r.time>=clip.start));painted=-1;draw(Number(slider.value))}else if(Number(slider.value)>=f.length-1)slider.value=0;playTime=f[Number(slider.value)].time;lastTick=performance.now()}el('play').textContent=playing?'暂停':'播放'};
function animate(now){if(playing){const end=window.siteReplayRange?.()?.end??f.at(-1).time;playTime=Math.min(end,playTime+(now-lastTick)/1000*Number(el('speed').value));lastTick=now;let i=Number(slider.value);while(i<f.length-1&&f[i+1].time<=playTime+.001)i++;slider.value=i;draw(i);if(playTime>=end-.001||i===f.length-1){playing=false;el('play').textContent='播放'}}requestAnimationFrame(animate)}requestAnimationFrame(animate);
const decisionJumps=[],lastActions={};for(const a of d.decisions){if(a.phase==='precontact'){decisionJumps.push(['首次比较 '+a.opening,a.time_s]);continue}if(a.phase==='online'){if(lastActions[a.opening]!==a.action){decisionJumps.push([({Push:'在线续推 ',Avoid:'在线停止 ',Identify:'在线辨识 ',Cross:'净空已达 '})[a.action]+a.opening,a.time_s]);lastActions[a.opening]=a.action}}else decisionJumps.push([(a.phase==='completion'?'几何完成/冻结 ':a.phase==='clearance'?'净空穿越 ':'交互后 ')+a.opening,a.time_s])}
const verificationJumps=d.verifications
 .filter(x=>x.opening==='G01'||x.outcome==='free')
 .map(x=>[x.opening==='G01'?'玻璃验证':'空开口通过 '+x.opening,x.end_time_s]);
const jumps=[['起飞',f[0].time],...verificationJumps,...decisionJumps,['终点',f.at(-1).time]];
for(const [label,t] of jumps){const b=document.createElement('button');b.textContent=label;b.onclick=()=>{playing=false;el('play').textContent='播放';manual=false;const i=Math.max(0,f.findIndex(r=>r.time>=t-1e-8));slider.value=i;draw(i)};el('jumps').append(b)}
function enlarge(src){el('big').querySelector('img').src=src;el('big').style.display='block'}el('big').onclick=()=>el('big').style.display='none';
el('gallery').innerHTML=d.keyframes.map((k,i)=>`<div><img data-i="${i}" src="${k.dashboard}" alt="${esc(k.label)}"><small>${fmt(k.time,1)} s · ${esc(k.label)}</small></div>`).join('');el('gallery').onclick=e=>{if(e.target.dataset.i!=null)enlarge(d.keyframes[Number(e.target.dataset.i)].dashboard)};
const requested=new URLSearchParams(location.hash.slice(1)).get('t');
const initial=Math.max(0,f.findIndex(r=>r.time>=(requested==null?(d.site_replays?.[0]?.start??d.keyframes[0]?.time??0):Number(requested))));
slider.value=initial;draw(initial);
</script></body></html>'''

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);build(p.parse_args().folder)

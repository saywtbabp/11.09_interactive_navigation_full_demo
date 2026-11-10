"""Self-contained browser replay, with simulation data embedded in the page."""
import json

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>交互导航｜MuJoCo 回放</title><style>
*{box-sizing:border-box} body{margin:0;background:#101921;color:#e9f3f2;font:15px system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:32px 26px} h1{margin:.1em 0 .3em;font-size:clamp(28px,4vw,49px)}
small,.muted{color:#a3b8bf} h2{font-size:18px;margin:0 0 10px} p{line-height:1.55}
.grid{display:grid;grid-template-columns:1.05fr 1fr;gap:16px} .card{background:#18262f;border:1px solid #30434c;border-radius:12px;padding:17px}
#stats{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0} .metric{background:#21323a;padding:10px 17px;border-radius:9px}
.metric strong{display:block;color:#79e6cc;font-size:22px} svg{display:block;width:100%;height:auto}
button{background:#6bd9c0;color:#10252c;border:0;border-radius:6px;padding:8px 16px;cursor:pointer;font-weight:700}
.checkpoints button{padding:5px 8px;font-size:12px;background:#29414a;color:#e9f3f2}
.checkpoints button[aria-pressed="true"]{background:#6bd9c0;color:#10252c}
input{width:64%;accent-color:#6bd9c0;vertical-align:middle} .controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse} td,th{border-bottom:1px solid #30434c;padding:7px;text-align:left}
.legend{font-size:13px;color:#b9cad0;margin:9px 0} .events{max-height:310px;overflow:auto;margin-top:16px}
@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body><main><small>ROTOR-ONLY FLIGHT / RECORDED SIMULATION</small>
<h1>接触、净空与全局路线</h1><div id="sub" class="muted"></div><section id="stats"></section>
<section class="card" style="margin-bottom:16px"><h2>观测怎样更新决策</h2><div id="decision"></div>
<div id="milestones" class="controls checkpoints" style="margin-top:12px"></div>
<div class="legend">在线模式每个有效窗口同步更新权重、当前几何分界及动作；无需等待完整试探或退开。对照模式的后台预览单独标注时间及滞后。不可归一化时支持率留空，不编造概率。</div></section>
<div class="grid"><section class="card"><h2>俯视回放</h2><svg id="map" viewBox="0 0 500 560" aria-label="飞行轨迹与障碍物"></svg>
<div class="controls"><button id="play">播放</button><input id="slider" type="range" min="0" value="0" aria-label="回放时间"><span id="time"></span></div>
<div id="stage"></div><div id="recovery" class="legend"></div><div class="legend">青色实线 · 实飞　青色虚线 · 假设可通行路线　黄色虚线 · 绕行方案　白框 · 飞行包络</div></section>
<div><section class="card"><h2>接触力与物体响应</h2><svg id="signals" viewBox="0 0 490 232"></svg><div class="legend">橙色 · 合接触力 N　青色 · 物体位移（门 ° / 箱 m）</div></section>
<section class="card" style="margin-top:16px"><h2>数据约束区域与导航分界</h2>
<div id="checkpoints" class="controls checkpoints" aria-label="选择决策时刻"></div>
<div class="controls">约束高亮：<select id="window-focus" aria-label="选择窗口约束"></select></div>
<div class="controls">排除区：<select id="excluded-view" aria-label="显示已排除候选"><option value="show">浅灰叉显示</option><option value="hide">隐藏</option></select></div>
<svg id="posterior" viewBox="0 0 680 380"></svg>
<div id="limits" class="legend"></div>
<div class="legend">浅灰× · 仅满足部分窗口、已被累计硬约束排除（有效权重为0，不参与p_ok）　蓝紫 · 原约束带　白框 · 累计交集　黄色 · F*<br>当前是连续密度，灰叉只是绘图网格，不是新增粒子；只显示当前视野。绿色支持续推，橙色支持停止额外推压：通道仍受阻则绕行，已满足实际开度和净空则可穿越。白框内越亮权重越大。</div>
<details><summary>查看当前所有窗口的两条不等式</summary><div id="constraint-equations"></div></details></section>
<section class="card" style="margin-top:16px"><h2>分界与概率的计算采样</h2><svg id="boundary-history" viewBox="0 0 680 190"></svg><div class="legend">黄：F*；绿：p_ok；虚线：决策门限；白线：回放时间。横轴是结果记录时刻，几何采样时刻见上方标注。正式决策使用最新状态；断线表示不可用，不虚构中间分界。</div></section>
<section class="card" style="margin-top:16px"><h2>测量与模型说明</h2><svg id="replay" viewBox="0 0 490 150"></svg>
<div id="fit" class="legend"></div></section></div></div>
<section class="card events"><h2>同步传感器读数（随时间滑块变化）</h2><table id="sensors"></table></section>
<section class="card events"><h2>路线与试探日志</h2><table id="events"></table></section></main>
<script>
const d=__DATA__;
const el=id=>document.getElementById(id);
const fmt=(v,n=2)=>v==null?'—':Number(v).toFixed(n);
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const s=(x,y,t,extra='')=>`<text x="${x}" y="${y}" fill="#bccdd3" font-size="12" ${extra}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,color,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
const poly=(points,color,extra='')=>`<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" ${extra}/>`;
const isDoor=d.object_kind==='door', convert=q=>isDoor?q*180/Math.PI:q, units=isDoor?'°':'m';
el('sub').textContent=`${d.case} / ${d.policy} / ${d.action} · 持续水平推力指令 ${d.force_cap_N} N · λ = ${d.rate_m_per_J} m/J`;
const metrics=[[d.success?(d.shortcut?'通过捷径':'到达 · 绕行'):'运行中止','任务结果'],
 [fmt(d.path_m)+' m','实飞距离'],[fmt(d.work_J,3)+' J','正接触功'],
 [fmt(d.contact_peak_N)+' N','峰值接触力'],[fmt(d.wall_peak_N)+' N','墙 / 门框峰值']];
el('stats').innerHTML=metrics.map(([v,l])=>`<div class="metric"><strong>${v}</strong><small>${l}</small></div>`).join('');
const X=x=>250+39*x, Y=y=>522-38*y;
const box=(x,y,hx,hy,color,extra='')=>`<rect x="${X(x-hx)}" y="${Y(y+hy)}" width="${78*hx}" height="${76*hy}" fill="${color}" ${extra}/>`;
const a=d.map, row=a.internal_walls[0].y_m, doors=a.openings;
let base=box(0,6.5,5.5,6.5,'#111d25','stroke="#82949c"');
let prev=-5.5;
for(const o of doors){
 const [x,y]=o.center_xy_m, hw=o.width_m/2;
 base+=box((prev+x-hw)/2,y,(x-hw-prev)/2,.06,'#9aa8ac');
 if(o.id==='Oglass')base+=box(x,y,hw-.09,.045,'#56b6e5');
 for(const sign of [-1,1])base+=box(x+sign*(hw-.045),y,.045,.085,'#c2d2d8');
 base+=s(X(x),Y(y)-11,{Oglass:'固定玻璃',Otarget:'交互通道',Obypass:'开放绕行'}[o.id],'text-anchor="middle"');
 prev=x+hw;
}
base+=box((prev+5.5)/2,row,(5.5-prev)/2,.06,'#9aa8ac');
for(const [name,p,c] of [['起点',a.start_xyz_m,'#8cead4'],['目标',a.goal_xyz_m,'#faf2b0']]){
 base+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="5" fill="${c}"/>`+s(X(p[0])+9,Y(p[1])-7,name);
}
for(const [k,c] of [['BYPASS_OPTION','#d9b878'],['SHORTCUT_OPTION','#459e9a']]){
 if(d.paths[k])base+=poly(d.paths[k].map(([x,y])=>[X(x),Y(y)]),c,'stroke-dasharray="5 5" opacity=".76"');
}
const f=d.frames, slider=el('slider');
slider.max=Math.max(f.length-1,0);
const last=f.at(-1)?.[0]||1;
let axis='';
for(let i=0;i<=4;i++){
 const x=38+i*108;
 axis+=line(x,12,x,195,'#30434c')+s(x,214,fmt(last*i/4,1)+' s','text-anchor="middle"');
}
const sx=t=>38+432*t/last;
const fq=Math.max(d.contact_peak_N,1), qmax=Math.max(convert(d.target_q),...f.map(r=>convert(r[4])));
const measured=d.trace.filter((_,i)=>i%4===0);
if(measured.length) axis+=poly(measured.map(r=>[sx(r[0]),190-162*r[4]/fq]),'#ffa66c');
axis+=poly(f.map(r=>[sx(r[0]),190-162*convert(r[4])/qmax]),'#6bd9c0');
axis+=line(38,190-162*convert(d.target_q)/qmax,470,190-162*convert(d.target_q)/qmax,'#6bd9c0','stroke-dasharray="4 5" opacity=".6"');
axis+=s(38,16,fmt(fq)+' N')+s(470,16,fmt(qmax)+units,'text-anchor="end"');
function draw(i, checkpoint){
 const r=f[i]; if(!r)return;
 const [t,x,y,z,q,force,work,stage]=r, [ox,oy]=doors[1].center_xy_m;
 const recovery=d.recovery_frames?.findLast(r=>r.time<=t+.001);
 let body;
 if(isDoor){
  const c=Math.cos(q),sine=Math.sin(q), pivot=[ox-1.1825/2,oy+.035];
  const points=[ [0,-.06],[1.1825,-.06],[1.1825,-.01],[0,-.01],[0,-.06] ];
  body=poly(points.map(([u,v])=>[X(pivot[0]+c*u-sine*v),Y(pivot[1]+sine*u+c*v)]),'#ffa66c');
 }else body=box(ox,oy+q,1.1825/2,.25,'#c97946');
 const trail=poly(f.slice(0,i+1).map(r=>[X(r[1]),Y(r[2])]),'#7aebd2');
 el('map').innerHTML=base+body+trail+
  box(x,y,d.envelope_m[0],d.envelope_m[1],'none','stroke="white" opacity=".72"')+
  ((recovery?.local_hover && ['PROBE_UNLOAD','PROBE_HOVER'].includes(stage)) ? (recovery.hover_shapes??[]).map(g=>box(x+g.offset[0],y+g.offset[1],g.half[0],g.half[1],'none','stroke="#81e9cb" opacity=".35"')).join('') : '')+
  `<circle cx="${X(x)}" cy="${Y(y)}" r="5" fill="#ddfff5"/>`;
 el('signals').innerHTML=axis+line(sx(t),10,sx(t),195,'#fff','opacity=".7"');
 el('time').textContent=fmt(t,1)+' s';
 el('stage').textContent=`${stage} · 高度 ${fmt(z)} m · ${fmt(force)} N · 累计 ${fmt(work,3)} J`;
 el('recovery').textContent=recovery && ['PROBE_UNLOAD','PROBE_HOVER','AFTER_PROBE_STATE'].includes(stage) ?
  (recovery.local_hover?'就近悬停（选绕行后才撤离）：':'状态触发恢复：')+[['detached','脱离接触'],['stable','姿态/速度稳定'],['envelope_ok','机体包络'],['clearance',recovery.local_hover?'局部悬停净空':'当前位置净空'],['exit_path',recovery.local_hover?'退出预案可执行':'退出路径']].map(([k,n])=>n+(recovery[k]?' ✓':' …')).join(' / ')+` · 连续满足 ${fmt(recovery.dwell_s,2)}/0.25 s`+(recovery.local_hover?' · 白框为巡航包络，浅绿细框为局部姿态扫掠包络':'') : '';
 const sample=d.sensor_frames?.findLast(r=>r.time<=t+.001);
 el('sensors').innerHTML=sample?'<tr><th>通道</th><th>当前值</th></tr>'+Object.entries(sample.readings).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${Array.isArray(v)?v.map(x=>fmt(x,4)).join(', '):fmt(v,4)}</td></tr>`).join(''):'<tr><td>此实验只保存任务与接触时序；直接运行 main.py 可记录全部模型传感器。</td></tr>';
 const live=d.snapshots.findLastIndex(b=>b.label!=='preview' && b.time<=t+.001);
 drawPosterior(checkpoint ?? Math.max(0,live));
 drawBoundaryHistory(t);
}
slider.oninput=()=>draw(Number(slider.value));
let timer=null;
el('play').onclick=()=>{
 if(timer){clearInterval(timer);timer=null;el('play').textContent='播放';return;}
 if(Number(slider.value)===Number(slider.max))slider.value=0;
 el('play').textContent='暂停';
 timer=setInterval(()=>{
  slider.value=Number(slider.value)+2; slider.oninput();
  if(Number(slider.value)>=Number(slider.max))el('play').onclick();
 },50);
};
const prior=d.snapshots[0], resistanceUnits='N';
function criterion(b){
 const rule=b.decision_rule??d.decision_rule??'cautious',margin=b.decision_tie_margin??d.tie_margin??0;
 const high=rule==='majority'?.5+margin:1-d.alpha,low=rule==='majority'?.5-margin:high,p=b.push_probability;
 const valid=p!=null&&b.integration_converged;
 const push=valid&&(rule==='majority'?p>high:p>=high);
 const tie=valid&&rule==='majority'&&p>=low&&p<=high;
 return {rule,high,low,p,valid,push,tie,
  verdict:!valid?'概率不可用或积分未收敛：回退':tie?'两侧权重未分胜负：未定并回退':push?'支持继续交互':rule==='majority'?'退回绕行':'保守门限未满足：退回绕行'};
}
const formal=d.snapshots.find(b=>b.label==='replan')??d.snapshots.find(b=>b.label==='online_update'&&b.push_probability!=null);
if(formal){
 const c=criterion(formal),p=c.p;
 el('decision').textContent=`本记录规则：${c.rule==='majority'?'多数权重（默认超过50%续推）':'保守门限（1−α）'}；续推门限 ${fmt(c.high*100,1)}%。F*=min(能力界 ${fmt(formal.capacity_limit,3)}, 经济界 ${formal.economic_limit==null?'无限':fmt(formal.economic_limit,3)})=${fmt(formal.push_limit,3)} N；绿色权重 ${p==null?'未定义':fmt(p*100,2)+'%'}，橙色权重 ${p==null?'未定义':fmt((1-p)*100,2)+'%'} ⇒ ${c.verdict}。G=${fmt(formal.gain,3)} m，剩余行程=${fmt(formal.stroke,3)} m。`;
 el('decision').style.color=c.tie?'#bccbd4':c.push?'#48d5aa':'#f4a261';
}else el('decision').textContent='本次没有试探后的正式重规划决策。';
for(const [name,label] of [['试探结束','probe'],['正式决策（看分界两侧）','replan']]){
 const idx=d.snapshots.findIndex(b=>b.label===label);if(idx<0)continue;
 const button=document.createElement('button');button.textContent=name;
 button.onclick=()=>{const j=f.findIndex(r=>r[0]>=d.snapshots[idx].time-.001);if(j>=0)slider.value=j;draw(Number(slider.value),idx)};
 el('milestones').appendChild(button);
}
const snapshotName=b=>b.label;
const checkpointButtons=[];
const select=document.createElement('select'); select.style.maxWidth='100%';
d.snapshots.forEach((b,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${b.label} ${fmt(b.time,1)} s`;select.appendChild(o)});
select.onchange=()=>{const i=Number(select.value);const j=f.findIndex(r=>r[0]>=d.snapshots[i].time);if(j>=0)slider.value=j;draw(Number(slider.value),i)};
el('checkpoints').appendChild(select);
let shown=-1;
let focusCount=-1;
const windowFocus=el('window-focus');windowFocus.value='-1';
windowFocus.onchange=()=>{const i=shown;shown=-1;drawPosterior(i)};
el('excluded-view').value='show';el('excluded-view').onchange=()=>{const i=shown;shown=-1;drawPosterior(i)};
const bandColors=['#70b6ff','#b69aff','#66d6ec','#d29ee8','#91b8e9','#88a0ff','#63c4e3','#b188f3','#8bcaef','#d0b6ff','#6b93d6','#aab5ee','#ae85cf','#86b3d0'];
function drawBoundaryHistory(time){
 const duringRecovery=b=>b.label==='route_preview'||b.label==='retreat_route_preview'||b.label.startsWith('live_PROBE_UNLOAD')||b.label.startsWith('live_PROBE_HOVER');
 const online=d.policy==='online_majority',decoupled=d.snapshots.some(b=>b.route_preview_status==='fresh_formal_geometry');
 const rows=d.snapshots.filter(b=>online?b.route_preview_status==='paired_online_geometry':decoupled?(b.label==='route_preview'||b.label==='route_preview_unavailable'||b.label==='replan'):(duringRecovery(b)||b.label==='probe'||b.label==='replan'));
 if(!rows.length){el('boundary-history').innerHTML=s(30,60,'当前记录没有几何分界样本');return}
 const lo=rows[0].time,hi=rows.at(-1).time,tx=t=>48+590*(t-lo)/Math.max(.001,hi-lo);
 const maxF=Math.max(d.force_cap_N,...rows.map(b=>b.push_limit??0));let svg='';
 for(const [key,color,top,height,max] of [['push_limit','#ffea82',22,55,maxF],['push_probability','#48d5aa',106,55,1]]){
  let segment=[];const flush=()=>{if(segment.length>1)svg+=poly(segment,color);segment=[]};
  rows.forEach(b=>{if(b[key]==null){flush();return}segment.push([tx(b.time),top+height*(1-b[key]/max)])});flush();
  rows.forEach(b=>{if(b[key]!=null)svg+=`<circle cx="${tx(b.time)}" cy="${top+height*(1-b[key]/max)}" r="${b.label==='replan'?4:2.5}" fill="${color}"><title>${b.label} / ${fmt(b.time,2)} s</title></circle>`});
  svg+=line(48,top+height,638,top+height,'#647a8a')+s(3,top+12,key==='push_limit'?'F* N':'p_ok');
  if(key==='push_probability'){
   const c=criterion(formal??rows.at(-1));
   svg+=line(48,top+height*(1-c.high),638,top+height*(1-c.high),'#a9bdc8','stroke-dasharray="4 4"');
   if(c.low!==c.high)svg+=line(48,top+height*(1-c.low),638,top+height*(1-c.low),'#a9bdc8','stroke-dasharray="4 4"');
  }
  if(time>=lo&&time<=hi)svg+=line(tx(time),top,tx(time),top+height,'white','opacity=".7"');
 }
 svg+=s(48,185,fmt(lo,2)+' s')+s(638,185,fmt(hi,2)+' s','text-anchor="end"');
 el('boundary-history').innerHTML=svg;
}
function clipBand(polygon,a,b,c){
 let out=[];for(let i=0;i<polygon.length;i++){
  const p=polygon[i],q=polygon[(i+1)%polygon.length],u=a*p[0]+b*p[1]-c,v=a*q[0]+b*q[1]-c;
  if(u<=1e-10)out.push(p);if((u<0)!==(v<0))out.push([p[0]+(q[0]-p[0])*u/(u-v),p[1]+(q[1]-p[1])*u/(u-v)]);
 }return out;
}
function drawPosterior(index){
 if(!prior||index===shown)return; shown=index;select.value=index;
 const b=d.snapshots[index];let svg='';
 const constraints=b.constraints??[];
 if(!b.bounds&&!constraints.length){
  if(b.push_limit!=null){const hi=Math.max(2.,b.push_limit*1.15),yy=315-270*b.push_limit/hi;
   el('posterior').innerHTML=`<rect x="58" y="45" width="560" height="${yy-45}" fill="#3c271d"/><rect x="58" y="${yy}" width="560" height="${315-yy}" fill="#12362e"/>`+line(58,yy,618,yy,'#ffea82','stroke-width="3" data-boundary="decision"')+s(65,32,'当前几何分界 F*='+fmt(b.push_limit,3)+' N')+s(65,345,'尚无有限参数区域；背景仅表示决策两侧，不是参数先验');
   el('limits').textContent='边界已计算，支持率尚不可定义；等待可辨识运动窗口，没有固定窗口数量门槛。';
  }else{el('posterior').innerHTML=s(35,70,'未形成有限参数区域，p_ok 未定义')+s(35,105,b.status);el('limits').textContent='接触后同步计算当前几何边界。'}return;
 }
 const x0=0,x1=b.bounds?Math.max(1.5,b.bounds[0][1]*1.2):4;
 const p=b.precision,h=b.linear,n=120,left=58,top=35,width=560,height=280;
 const limit=b.push_limit,y0=0,y1=Math.max(b.bounds?b.bounds[1][1]*1.2:2,Number.isFinite(limit)?Math.max(0,limit)*1.15:0,.1);
 if(focusCount!==constraints.length){
  const old=windowFocus.options.length?Number(windowFocus.value):-1;windowFocus.innerHTML='';
  for(let i=-1;i<constraints.length;i++){const o=document.createElement('option');o.value=i;o.textContent=i<0?'全部窗口':`窗口 ${i+1}`;windowFocus.appendChild(o)}
  windowFocus.value=old<constraints.length?String(old):'-1';focusCount=constraints.length;
 }
 const focus=Number(windowFocus.value);
 const px=x=>left+width*(x-x0)/(x1-x0),py=y=>top+height*(y1-y)/(y1-y0);
 const split=py(Math.max(y0,Math.min(y1,limit??0)));
 svg+=`<rect data-decision="bypass" x="${left}" y="${top}" width="${width}" height="${split-top}" fill="${limit==null?'#667b91':'#f4a261'}" opacity=".16"/>`;
 svg+=`<rect data-decision="continue" x="${left}" y="${split}" width="${width}" height="${top+height-split}" fill="${limit==null?'#667b91':'#48d5aa'}" opacity=".16"/>`;
 const segments=[];
 constraints.forEach(([dv,T,Y,ev,ey],i)=>{
  let band=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]];
  band=clipBand(band,dv-ev,T,Y+ey);band=clipBand(band,-dv-ev,-T,-Y+ey);
  const color=bandColors[i%bandColors.length],active=focus<0||focus===i;
  if(band.length>2)svg+=`<polygon points="${band.map(([x,y])=>[px(x),py(y)].join(',')).join(' ')}" fill="${color}" opacity="${focus===i?.15:.018}" data-band="${i+1}"/>`;
  for(const [a,c] of [[dv-ev,Y+ey],[dv+ev,Y-ey]]){
   let ends=[];
   for(const x of [x0,x1]){const y=(c-a*x)/T;if(y>=y0-1e-9&&y<=y1+1e-9)ends.push([x,y])}
   if(Math.abs(a)>1e-12)for(const y of [y0,y1]){const x=(c-T*y)/a;if(x>=x0-1e-9&&x<=x1+1e-9)ends.push([x,y])}
   if(ends.length>=2)segments.push({i,color,active,ends:[ends[0],ends.at(-1)]});
  }
 });
 const inside=(x,y)=>b.polygon.length>2&&b.polygon.every((a,i)=>{const c=b.polygon[(i+1)%b.polygon.length];return(c[0]-a[0])*(y-a[1])-(c[1]-a[1])*(x-a[0])>=-1e-9});
 let cells=[];
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){
  const x=x0+(i+.5)*(x1-x0)/n,y=y1-(j+.5)*(y1-y0)/n;
  if(inside(x,y))cells.push([i,j,h[0]*x+h[1]*y-.5*(p[0][0]*x*x+2*p[0][1]*x*y+p[1][1]*y*y),y<limit]);
 }
 const peak=Math.max(...cells.map(c=>c[2]));
 cells.forEach(([i,j,v,good])=>{svg+=`<rect x="${left+i*width/n}" y="${top+j*height/n}" width="${width/n+.2}" height="${height/n+.2}" fill="${limit==null?'#86c7ed':good?'#48d5aa':'#f4a261'}" opacity="${Math.exp((v-peak)*.6)}"/>`});
 if(el('excluded-view').value!=='hide')for(let i=0;i<28;i++)for(let j=0;j<26;j++){
  const x=x0+(i+.5)*(x1-x0)/28,y=y0+(j+.5)*(y1-y0)/26;
  const some=constraints.some(([dv,T,Y,ev,ey])=>(dv-ev)*x+T*y<=Y+ey+1e-10&&(dv+ev)*x+T*y>=Y-ey-1e-10);
  const allPass=constraints.every(([dv,T,Y,ev,ey])=>(dv-ev)*x+T*y<=Y+ey+1e-10&&(dv+ev)*x+T*y>=Y-ey-1e-10);
  if(some&&!allPass)svg+=`<path d="M${px(x)-2},${py(y)-2}l4,4m0,-4l-4,4" stroke="#a6b2bd" opacity=".48" stroke-width=".8" data-excluded="true"><title>满足部分窗口，未通过全部硬约束；有效权重=0；仅为显示候选</title></path>`;
 }
 for(const z of segments)svg+=line(px(z.ends[0][0]),py(z.ends[0][1]),px(z.ends[1][0]),py(z.ends[1][1]),z.color,`opacity="${z.active?.65:.14}" stroke-width="${focus===z.i?2.5:1}" data-constraint="${z.i+1}"`);
 if(b.polygon.length)svg+=poly([...b.polygon,b.polygon[0]].map(([x,y])=>[px(x),py(y)]),'white','data-effective-region="true"');
 if(limit!=null&&limit>=y0&&limit<=y1)svg+=line(left,py(limit),left+width,py(limit),'#ffea82','stroke-width="3" data-boundary="decision"');
 svg+=s(left,19,limit==null?'当前几何比较不可用：不插值边界':`分界 F* = ${fmt(limit,3)} N · 线下绿 / 线上橙`,'style="fill:#ffea82"');
 const [mx,ry]=d.truth_for_evaluation??[NaN,NaN];
 if(mx>=x0&&mx<=x1&&ry>=y0&&ry<=y1)svg+=line(px(mx)-4,py(ry),px(mx)+4,py(ry),'white')+line(px(mx),py(ry)-4,px(mx),py(ry)+4,'white');
 svg+=s(left,338,fmt(x0)+' kg')+s(left+width,338,fmt(x1)+' kg','text-anchor="end"')+s(5,top+10,fmt(y1))+s(5,315,fmt(y0))+s(135,365,`有效质量 kg / 阻力 N · ${constraints.length} 个窗口、${constraints.length*2} 条约束`);
 el('posterior').innerHTML=svg;
 el('constraint-equations').innerHTML=constraints.map(([dv,T,Y,ev,ey],i)=>`<p style="color:${bandColors[i%bandColors.length]}">#${i+1}：(${fmt(dv-ev,6)})m + ${fmt(T,3)}R ≤ ${fmt(Y+ey,6)}<br>　　(${fmt(dv+ev,6)})m + ${fmt(T,3)}R ≥ ${fmt(Y-ey,6)}</p>`).join('');
 const c=criterion(b),defined=c.p!=null;
 const verdict=limit==null?'当前几何比较不可用（非参数失效）':b.decision_reason==='no_remaining_navigation_gain'?'继续推压无额外导航收益，停止施力':c.verdict;
 const source={paired_online_geometry:'在线同一更新：权重、边界、动作同步计算',fresh_formal_geometry:'正式决策：最新状态重新计算',asynchronous_geometry_sample:'后台几何预览（不用于正式决策）',awaiting_geometry_preview:'等待后台几何预览',no_mechanical_evidence:'尚无有效力学窗口',committed_decision_reference:'已提交动作，显示正式决策时刻的分界'};
 source.frozen_after_geometric_completion='通行几何已满足：推断结束，保留最后一次真实权重和边界（历史冻结图）';
 el('limits').textContent=`${verdict} · p_ok=${!defined?'未定义':fmt(c.p,3)} / ${c.rule==='majority'?'多数':'保守'}门限 ${fmt(c.high)}\n${b.label} · ${b.windows} 个运动窗口 · F*=${limit==null?'不可用':fmt(limit,3)+' N'} · G=${fmt(b.gain)} m · 剩余行程 ${fmt(b.stroke)} m\n${source[b.route_preview_status]??b.route_preview_status??'正式记录：由该时刻几何计算分界'}${b.geometry_time!=null?' · 几何采样 '+fmt(b.geometry_time,2)+' s / 滞后 '+fmt(b.boundary_age_s,2)+' s':''}`;
 el('limits').style.color=!defined||c.tie?'#bccbd4':c.push?'#48d5aa':'#f4a261';
 if(b.exit_latched)el('limits').textContent+='\\n已在线选择停止施力；后续参数更新不触发重新接触。';
 if(b.inference_frozen)el('limits').textContent+='\\n冻结于 '+fmt(b.frozen_at,2)+' s；图中权重和分界来自 '+fmt(b.frozen_reference_time,2)+' s，不是新的计算。传感器和执行净空检查继续。';
 el('limits').style.whiteSpace='pre-line';
}

draw(0);
const fit=d.force_replay;
if(fit.time){
 const bottom=Math.min(...fit.actual,...fit.prior,...fit.posterior), top=Math.max(...fit.actual,...fit.prior,...fit.posterior);
 const dur=fit.time.at(-1)-fit.time[0]||1;
 const lineFor=(ys,c)=>poly(ys.map((y,i)=>[38+432*(fit.time[i]-fit.time[0])/dur,122-98*(y-bottom)/Math.max(top-bottom,.001)]),c);
 el('replay').innerHTML=lineFor(fit.prior,'#d9b878')+lineFor(fit.posterior,'#6bd9c0')+lineFor(fit.actual,'#e9f3f2')+
  s(38,15,fmt(convert(top))+units)+s(38,142,'0 s')+s(470,142,fmt(dur)+' s','text-anchor="end"');
 el('fit').textContent=`白 · 实测　黄 · 先验　青 · 交互后。RMSE ${fmt(convert(fit.prior_rmse),3)} → ${fmt(convert(fit.posterior_rmse),3)} ${units}`;
}else{
 el('replay').style.display='none';el('fit').textContent='本版使用仿真参考测量：接触求解器力、关节位姿和无人机位姿。门用 s=r₀q、v=r₀ω、F∥=τ/r₀，箱用平移及沿轴力。没有门上传感器；尚未接回10.31的含噪机载估计和视觉识别。局部常阻力近似及观测误差界需独立标定，p_ok不是经实机校准的成功率。';
}
el('events').innerHTML='<thead><tr><th>时间 s</th><th>事件</th><th>记录</th></tr></thead><tbody>'+
 d.events.map(({time,event,...rest})=>`<tr><td>${fmt(time)}</td><td>${esc(event)}</td><td>${Object.entries(rest).map(([k,v])=>esc(k)+' '+esc(typeof v==='number'?fmt(v,3):v)).join(' · ')}</td></tr>`).join('')+'</tbody>';
</script></body></html>"""


def save_report(path, result):
    data = json.dumps(result, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    path.write_text(PAGE.replace('__DATA__', data), encoding='utf-8')

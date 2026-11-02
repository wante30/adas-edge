"""Create a traceable PNG asset library; originals are never edited."""
import collections,csv,datetime,hashlib,html,json,math,pathlib,shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=pathlib.Path(__file__).resolve().parents[1]
TOPICS={'01_system_overview':'完整车机与系统总览','02_multitask_segmentation':'MTL 多任务分割与道路语义','03_multicamera_detection':'FAST-Q4、FAST-MOSAIC 与目标检测','04_tesla_bev':'Tesla BEV 与空间感知','05_hazard_warning':'真实危险预警与风险可视化','06_control_models_performance':'FPGA 控制、模型菜单与性能证据'}
USE={'01_system_overview':'项目书封面、系统介绍、答辩开场','02_multitask_segmentation':'多任务感知、检测与分割联合输出','03_multicamera_detection':'单路/整图一次推理、模型方案对比','04_tesla_bev':'跟踪结果与相对空间可视化','05_hazard_warning':'危险感知闭环、真实告警案例','06_control_models_performance':'FPGA 反馈、模型切换、软硬件联动'}
SOURCE={'REAL_PCIE_LIVE':'本次真实PCIe','HISTORICAL_REAL_PCIE':'历史真实PCIe','HISTORICAL_REPLAY':'历史录制回放','HISTORICAL_DESKTOP':'历史桌面参考'}
FONT='C:/Windows/Fonts/msyh.ttc'
def font(n):return ImageFont.truetype(FONT,n)
def lines(path):
    result=[]
    if path.exists():
        for l in path.read_text(encoding='utf-8').splitlines():
            try:result.append(json.loads(l))
            except ValueError:pass
    return result

def active(m):return [e for e in (m.get('ai_before') or {}).get('hazard_details') or [] if isinstance(e,dict) and e.get('lifecycle_state','ACTIVE')=='ACTIVE' and e.get('valid',True)]
def caption(m):
    a=m.get('ai_before') or {};v=m.get('vehicle_before') or {};s=m.get('section');counts=a.get('detection_counts') or {};ev=active(m)
    text=f'RK3566 真实 PCIe 输入、原生 C++/Qt 车机；模型 {a.get("model_id")}，截图前最近 AI 结果帧 {a.get("frame_id")}。'
    if counts:text+=' 检测计数：'+'、'.join(f'{k} {n}' for k,n in counts.items())+'。'
    if s=='mtl_front':text+=' 实际 MTL 检测与可行驶区域/车道等分割输出，非人工标注。'
    if s in ('fast_mosaic','mosaic_lite'):text+=' 四象限来自真实 PCIe 帧；Q1 当前非黑但短采样未变化，未证明四个独立动态摄像头全部验活。'
    if s=='mosaic_lite':text+=' Lite 为纯检测配置，OCR、风险控制和 BEV 等负载停用，不可当作完整系统性能。'
    if s=='tesla_bev':text+=' BEV 由实际目标跟踪驱动，是相对空间示意，非米级测距或三维重建精度证明。'
    if ev:text+=' 截图前最近事件日志（异步显示，以图中状态为准）：'+'、'.join(sorted(set(str(e.get('kind'))+' ['+str(e.get('severity'))+']' for e in ev)))+'。'
    if m.get('overlay')=='model_menu':text+=' 真实模型菜单及可用方案，状态按原图保留。'
    if m.get('overlay')=='fpga_read_panel':text+=' 已打开 FPGA 查询面板并尝试 READ，但画面未显示有效应答，不作为硬件反馈证据。'
    if m.get('probe_label'):text+=' 本图来自单次受控硬件刹车探测，不是 AI 自动触发；关联 ACK 与结束反馈见 controlled_fpga_probe.json。'
    if v.get('current_action')=='brake' and v.get('brake_allow'):text+=f' 截图前硬件反馈为刹车允许，序号 {v.get("last_sequence")}；不代表实体机械制动验收。'
    return text+' 界面 FPS 是含截图负载的瞬时值；页脚为板端时钟，采集日期以电脑端记录为准。'

def load():
    out=[];p=ROOT/'evidence/controlled_fpga_probe.json';probe=p.exists() and json.loads(p.read_text())['report'].get('passed')
    for r in lines(ROOT/'evidence/acquisition.jsonl'):
        if r.get('kind')!='frame':continue
        m=r['metadata'];out.append(dict(id='live_%04d'%m['sequence'],file=r['file'],source_type='REAL_PCIE_LIVE',captured_at=r['pc_received_at'],section=m['section'],metadata=m,caption_zh=caption(m),probe_passed=probe))
    p=ROOT/'evidence/historical_imports.json'
    if p.exists():
        for i,r in enumerate(json.loads(p.read_text(encoding='utf-8')),1):out.append(dict(id='history_%03d'%i,file=r['file'],source_type=r['source_type'],captured_at=None,section='historical',metadata={},caption_zh=r['note_zh'],original_source=r['source_path']))
    return out

VISUAL_REVIEW=json.loads((ROOT/'evidence/visual_review.json').read_text(encoding='utf-8')) if (ROOT/'evidence/visual_review.json').exists() else {}

def assess(rows):
    seen={}
    for r in rows:
        p=ROOT/r['file'];r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['exclusions']=[]
        with Image.open(p) as im:
            im.load();r['width'],r['height']=im.size;rgb=np.array(im.convert('RGB'))
        roi=rgb[307:537,503:987] if r['section'] in ('fast_mosaic','mosaic_lite') else rgb[245:537,20:987]
        gray=cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY);r['sharpness']=round(float(cv2.Laplacian(gray,cv2.CV_64F).var()),3)
        values=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()[1:];bits=values>np.median(values)
        r['phash']=f'{sum(int(b)<<i for i,b in enumerate(bits)):016x}'
        hsv=cv2.cvtColor(roi,cv2.COLOR_RGB2HSV);r['color_fraction']=round(float(np.mean((hsv[:,:,1]>140)&(hsv[:,:,2]>100))),5)
        if im.size!=(1280,720):r['exclusions'].append('非原生1280x720')
        if r['sha256'] in seen:r['exclusions'].append('逐字节重复：'+seen[r['sha256']])
        else:seen[r['sha256']]=r['id']
        m=r['metadata'];a=m.get('ai_before') or {};b=m.get('ai_after') or {};ev=active(m)
        r.update(model=a.get('model_id','historical'),overlay=m.get('overlay','none'),detection_count=sum((a.get('detection_counts') or {}).values()),hazard_count=len(ev),critical_count=sum(e.get('severity')=='critical' for e in ev),hardware_brake=(m.get('vehicle_before') or {}).get('brake_allow') is True)
        if r['source_type']=='REAL_PCIE_LIVE':
            if a.get('model_id')!=b.get('model_id'):r['exclusions'].append('截图期间模型切换')
            if not a.get('frame_id'):r['exclusions'].append('缺少有效AI结果帧')
            if r['section']=='fpga_control' and not r['probe_passed']:r['exclusions'].append('硬件检查未通过')
        else:r['exclusions'].append('仅历史参考，不进入本次精品')
        if 'offline' in r['file']:r['exclusions'].append('历史离线状态')
        if r['section']=='fpga_register_read':r['exclusions'].append('查询面板未显示有效应答')
        if r['sharpness']<25:r['exclusions'].append('道路清晰度偏低')
        r['critical_metadata_before']=r['critical_count']
        review=VISUAL_REVIEW.get(r['id'],{})
        r['visual_review']=review
        if review.get('duplicate_of'):r['exclusions'].append('人工复核近重复：'+review['duplicate_of'])
        r['critical_count']=int(review.get('visible_critical',False))
        r['score']=round(math.log1p(r['sharpness'])*2+min(r['detection_count'],18)*.8+min(r['hazard_count'],3)*1.5+r['critical_count']*7+r['color_fraction']*8+(8 if r['hardware_brake'] else 0),3)
    return rows

def dist(a,b):return (int(a['phash'],16)^int(b['phash'],16)).bit_count()
def choose(pool,n,used,caps=None,threshold=7):
    result=[];counts=collections.Counter()
    for limit in (threshold,4,2):
        for r in sorted(pool,key=lambda r:r['score'],reverse=True):
            if r['id'] in used or r['exclusions']:continue
            if caps and counts[r['section']]>=caps.get(r['section'],n):continue
            if any(x['model']==r['model'] and dist(x,r)<limit for x in result):continue
            r['selection_dedup_distance']=limit;result.append(r);used.add(r['id']);counts[r['section']]+=1
            if len(result)>=n:return result
    return result

def select(rows):
    live=[r for r in rows if r['source_type']=='REAL_PCIE_LIVE'];used=set();groups={}
    pool=[r for r in live if r['section'] in ('fpga_control','model_menu','fpga_register_read') and (r['section']!='fpga_control' or r['hardware_brake'])]
    control=choose([r for r in pool if r['section']=='fpga_control'],1,used)
    for model in ('MTL-FAST','FAST-Q4','FAST-MOSAIC','FAST-MOSAIC-LITE'):
        control+=choose([r for r in pool if r['section']=='model_menu' and r['model']==model],1,used,threshold=3)
    control+=choose([r for r in live if r['hardware_brake'] and not r['metadata'].get('probe_label') and not r['visual_review'].get('visible_critical') and r['overlay']=='none'],1,used,threshold=3)
    for model in ('MTL-FAST','FAST-Q4'):
        control+=choose([r for r in live if r['model']==model and r['section'] in ('mtl_front','fast_q4')],1,used,threshold=3)
    groups['06_control_models_performance']=control
    groups['04_tesla_bev']=choose([r for r in live if r['section']=='tesla_bev'],8,used)
    groups['05_hazard_warning']=choose([r for r in live if r['hazard_count'] and r['overlay']=='none' and r['section'] in ('mtl_front','fast_q4','fast_mosaic','natural_hazard')],8,used)
    groups['03_multicamera_detection']=choose([r for r in live if r['section'] in ('fast_q4','fast_mosaic','mosaic_lite')],12,used,{'fast_q4':4,'fast_mosaic':6,'mosaic_lite':2})
    groups['02_multitask_segmentation']=choose([r for r in live if r['section']=='mtl_front'],10,used)
    groups['01_system_overview']=choose([r for r in live if r['section'] in ('mtl_front','fast_q4')],8,used)
    for t,rs in groups.items():
        for i,r in enumerate(rs,1):r.update(topic=t,curated_rank=i,recommended_use=USE[t])
    return groups

def sheets(name,title,rows):
    folder=ROOT/'contact_sheets';folder.mkdir(exist_ok=True);paths=[];per_page=24 if name=='core' else 20
    for page in range(math.ceil(len(rows)/per_page)):
        batch=rows[page*per_page:(page+1)*per_page];canvas=Image.new('RGB',(1440,74+math.ceil(len(batch)/4)*248),'#08131d');d=ImageDraw.Draw(canvas);d.text((20,15),title+f' | {page+1}/{math.ceil(len(rows)/per_page)}',font=font(22),fill='#dceff3')
        for i,r in enumerate(batch):
            x=12+i%4*360;y=72+i//4*248
            with Image.open(ROOT/r['file']) as im:canvas.paste(ImageOps.contain(im.convert('RGB'),(336,189)),(x,y))
            d.text((x,y+190),r['id']+' · '+r['model'][:19],font=font(14),fill='#b5e4ef');d.text((x,y+210),SOURCE.get(r['source_type'],r['source_type'])+(' · 红色危险' if r['critical_count'] else ''),font=font(13),fill='#87cbb5')
            if r['exclusions']:d.text((x,y+228),r['exclusions'][0][:25],font=font(12),fill='#ed9090')
        target=folder/(name+f'_{page+1:02d}.png');canvas.save(target,optimize=True);paths.append(target.relative_to(ROOT).as_posix())
    return paths

def export(groups):
    selected=[]
    for topic,rs in sorted(groups.items()):
        folder=ROOT/'curated'/topic
        for n in ('originals','ppt_16x9','project_book','focus_16x9'):(folder/n).mkdir(exist_ok=True,parents=True)
        text=['# '+TOPICS[topic],'','均来自本次真实 PCIe；原图和 PPT 全图未重绘。项目书版仅裁剪，聚焦版只裁剪和居中置于深色 16:9 画布。','']
        for i,r in enumerate(rs,1):
            stem=f'{topic[3:]}_{i:02d}_{r["id"]}.png';paths={n:folder/n/stem for n in ('originals','ppt_16x9','project_book','focus_16x9')}
            shutil.copy2(ROOT/r['file'],paths['originals']);shutil.copy2(ROOT/r['file'],paths['ppt_16x9'])
            roi=(20,77,987,537)
            if r['section']=='model_menu':roi=(844,56,1264,612)
            elif r['section']=='fpga_register_read':roi=(40,40,1240,680)
            elif r['section']=='fpga_control':roi=(1000,77,1260,459)
            with Image.open(ROOT/r['file']) as src:
                src=src.convert('RGB');src.crop((20,77,1260,692)).save(paths['project_book'],optimize=True)
                crop=src.crop(roi);focus=Image.new('RGB',(1280,720),'#030a10');focus.paste(crop,((1280-crop.width)//2,(720-crop.height)//2));focus.save(paths['focus_16x9'],optimize=True)
            r['exports']={k:p.relative_to(ROOT).as_posix() for k,p in paths.items()};r['crop_coordinates']={'project_book':[20,77,1260,692],'focus_16x9':list(roi)}
            text += [f'## {i:02d} · {r["id"]}','',r['caption_zh'],'',f'- 推荐位置：{USE[topic]}',f'- 电脑端时间：{r["captured_at"]}',f'- 原始 SHA-256：`{r["sha256"]}`',f'- [原图](originals/{stem}) · [PPT全图](ppt_16x9/{stem}) · [项目书裁剪](project_book/{stem}) · [聚焦16:9](focus_16x9/{stem})',''];selected.append(r)
        (folder/'README.md').write_text('\n'.join(text),encoding='utf-8')
    return selected

def core(selected):
    out=[]
    for topic,n in zip(TOPICS,(3,4,4,3,3,3)):
        pool=sorted([r for r in selected if r['topic']==topic],key=lambda r:r['curated_rank'])
        if topic=='06_control_models_performance':pool.sort(key=lambda r:0 if r['hardware_brake'] else 1 if r['section']=='model_menu' else 2)
        if topic=='03_multicamera_detection':
            preferred=[]
            for s in ('fast_mosaic','fast_q4','mosaic_lite'):
                hit=next((r for r in pool if r['section']==s),None)
                if hit:preferred.append(hit)
            pool=preferred+sorted([r for r in pool if r not in preferred],key=lambda r:0 if r['section']=='fast_mosaic' else 1)
        out+=pool[:n]
    if not any(r['section']=='model_menu' for r in out):
        menu=next((r for r in selected if r['section']=='model_menu'),None)
        if menu:out.append(menu)
    for i,r in enumerate(out,1):
        p=ROOT/'core'/(f'{i:02d}_'+r['topic'][3:]+'_'+r['id']+'.png');shutil.copy2(ROOT/r['file'],p);r.update(core_file=p.relative_to(ROOT).as_posix(),core_rank=i)
    return out

def gallery(rows):
    (ROOT/'thumbnails').mkdir(exist_ok=True)
    for r in rows:
        p=ROOT/'thumbnails'/(r['id']+'.png')
        with Image.open(ROOT/r['file']) as im:ImageOps.contain(im.convert('RGB'),(480,270)).save(p,optimize=True)
        r.setdefault('exports',{})['thumbnail']=p.relative_to(ROOT).as_posix()
    data=[{k:r.get(k) for k in ('id','file','source_type','section','model','score','exclusions','topic','caption_zh','exports','core_rank','recommended_use')} for r in rows]
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ADAS Edge 比赛素材库</title><style>*{box-sizing:border-box}body{background:#071119;color:#dbeaf1;font:15px "Microsoft YaHei",sans-serif;margin:0}header{padding:28px 4vw 16px}h1{margin:0 0 12px;font-size:28px}p{color:#9db6c3;line-height:1.8}nav{position:sticky;top:0;background:#0b1924ee;padding:16px 4vw;display:flex;gap:10px;flex-wrap:wrap;z-index:2}select,input,button{background:#122c3a;color:#dceef6;border:1px solid #365a6d;border-radius:5px;padding:9px}#grid{padding:24px 4vw;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}article{background:#0d202c;border:1px solid #284551;border-radius:8px;overflow:hidden}article img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#02080d;cursor:pointer}.text{padding:12px;line-height:1.6}.tag{font-size:12px;color:#72d4bb}h3{font-size:15px;margin:6px 0}a{color:#7cdbf2}details{font-size:13px;color:#a4bdc8}dialog{background:#071119;color:#deeff8;border:1px solid #40738a;width:min(1400px,96vw);max-height:95vh;padding:15px}dialog::backdrop{background:#000c}dialog img{width:100%;max-height:72vh;object-fit:contain}.warning{color:#f1c67f}#count{margin-left:auto}</style><header><h1>ADAS Edge · FPGA + RK3566 比赛素材库</h1><p>原生 1280×720 Qt 实机画面 · 原始 PNG 与裁剪版本 · 可追溯状态和硬件证据</p><p class="warning">Q1 当前非黑，但短采样未变化；不能声称四个独立动态摄像头全部验活。瞬时 FPS 不是正式成绩。受控 FPGA 探测与 AI 自动事件分开记录，机械制动未验收。</p><p><a href="OVERVIEW.md">素材总览</a> · <a href="catalogue.csv">完整清单 CSV</a> · <a href="evidence/controlled_fpga_probe.json">受控硬件证据</a> · <a href="QUALITY_AND_PROVENANCE.md">质量与来源说明</a></p></header><nav><select id="tier"><option value="core">核心推荐</option><option value="curated">全部精品</option><option value="all">原始候选（含历史）</option></select><select id="topic"><option value="">全部主题</option>OPTIONS</select><input id="search" placeholder="搜索模型、告警、来源"><span id="count"></span></nav><main id="grid"></main><dialog id="detail"><button onclick="document.querySelector('dialog').close()">关闭</button><img id="large"><h3 id="name"></h3><p id="caption"></p><p id="links"></p></dialog><script>const rows=PAYLOAD;const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));function show(id){const r=rows.find(x=>x.id===id);$('#large').src=r.file;$('#name').textContent=r.id+' · '+r.model;$('#caption').textContent=r.caption_zh;$('#links').innerHTML='<a target="_blank" href="'+r.file+'">原始截图</a>'+Object.entries(r.exports||{}).filter(([k])=>k!=='thumbnail').map(([k,v])=>' · <a target="_blank" href="'+v+'">'+esc(k)+'</a>').join('');$('#detail').showModal()}function render(){let out=rows.filter(r=>$('#tier').value==='core'?r.core_rank:$('#tier').value==='curated'?r.topic:true).filter(r=>!$('#topic').value||r.topic===$('#topic').value).filter(r=>!$('#search').value||JSON.stringify(r).toLowerCase().includes($('#search').value.toLowerCase()));out.sort((a,b)=>(a.core_rank||999)-(b.core_rank||999)||b.score-a.score);$('#count').textContent=out.length+' 张';$('#grid').innerHTML=out.map(r=>'<article><img loading="lazy" src="'+r.exports.thumbnail+'" onclick="show(this.dataset.id)" data-id="'+r.id+'"><div class="text"><span class="tag">'+esc(r.source_type)+(r.core_rank?' · CORE '+r.core_rank:'')+'</span><h3>'+esc(r.id)+' · '+esc(r.model)+'</h3><div>'+esc(r.recommended_use||r.section)+'</div><details><summary>图片说明与来源</summary>'+esc(r.caption_zh)+(r.exclusions?.length?'<p class="warning">'+esc(r.exclusions.join('；'))+'</p>':'')+'</details></div></article>').join('')}['#tier','#topic','#search'].forEach(s=>$(s).addEventListener('input',render));render();</script></html>'''
    options=''.join(f'<option value="{k}">{v}</option>' for k,v in TOPICS.items())
    (ROOT/'index.html').write_text(template.replace('PAYLOAD',json.dumps(data,ensure_ascii=False).replace('</','<\\/')).replace('OPTIONS',options),encoding='utf-8')

def reports(rows,selected,cores,groups):
    counts=collections.Counter(r['source_type'] for r in rows)
    doc=['# ADAS Edge 比赛素材总览','','采集整理：2026-11-02（Asia/Shanghai）。基于当前实际运行的 RK3566 C++/Qt + PCIe 版本，来源、状态及 SHA-256 随库提供。','',f'原始候选 **{len(rows)} 张**，精品 **{len(selected)} 张**，核心推荐 **{len(cores)} 张**。另有 5 张合成 UI 参考，隔离保存，不计入推荐。','','## 快速使用','','双击 index.html 可按核心/精品/全部候选、主题和关键词筛选，每张图可打开原图及裁剪版本。contact_sheets 为 PNG 缩略图总览。','','## 六类主题','','| 主题 | 精品 | 推荐位置 |','|---|---:|---|']
    for t in TOPICS:doc.append(f'| [{TOPICS[t]}](curated/{t}/README.md) | {len(groups[t])} | {USE[t]} |')
    doc+=['','## 核心图片推荐','','| 编号 | 图片 | 推荐位置 | 采集前日志摘要（异步） |','|---|---|---|---|']
    for r in cores:doc.append(f'| {r["core_rank"]:02d} | [{r["id"]}]({r["core_file"]}) | {r["recommended_use"]} | {r["model"]}；目标 {r["detection_count"]}；事件 {r["hazard_count"]}'+('；硬件刹车允许' if r['hardware_brake'] else '')+' |')
    doc+=['','## 推荐展示叙事','','1. 系统总览：完整车机实际运行。','2. MTL：车辆、行人与道路语义联合输出。','3. FAST-Q4 / FAST-MOSAIC：单路与整图一次推理；Lite 要明确说明纯检测负载。','4. 真实预警与 Tesla BEV：从检测、跟踪到风险可视化。','5. FPGA 控制：用受控探测截图和 JSON 说明请求、ACK、允许反馈及结束闭环。','6. 模型菜单与性能：显示实际可切换方案；正式性能引用历史独立验收，不能用截图时瞬时 FPS 替代。','','## 真实性边界','','- 精品全部为本次 REAL_PCIE_LIVE，未添加或重绘框、分割、FPS、ACK 或控制状态。','- Q1 非黑但短采样未变化，未证明四个独立动态摄像头验活，也未定位旧上游问题根因。','- 受控探测为人工授权单次请求，不伪装成 AI 自动触发。FPGA 接收/允许不等于机械制动验收。','- BEV 是相对空间示意。原图页脚为板端时钟，日期与电脑不一致；采集日期以清单电脑端时间为准，关联采用单调时间。','- 旧回放、历史实测和合成演示独立标注，未混作本次硬件证据。','','## 目录','','- raw_candidates：全部原始候选 PNG。','- curated：六主题精品，提供 originals、ppt_16x9、project_book、focus_16x9。','- core：可直接使用的核心全图。','- manifest.json / catalogue.csv：说明、模式、推荐位置、筛选和 SHA-256。','- evidence：状态、硬件联动、象限数据、历史报告和版本来源。','- tools：可复用采集与整理脚本。']
    (ROOT/'OVERVIEW.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
    quality=['# 质量、来源与处理说明','','## 版本与实际采集','','常规采集使用板端 xfce4-screenshooter 获取完整 X11 桌面；自然告警补采使用原生 xwd 抓取并在电脑无损解码为 RGB PNG（原始 XWD 也保留），PNG 经 SSH 直接传回电脑；批量图片不落板端磁盘。截图前后分别记录异步 AI、硬件及界面状态，这界定采集区间，不声称全部面板对应同一 AI 帧。','','本次只修复模型按钮遮住时钟/日期的问题，将二者移到页脚，并让快捷键提示留出位置。AI、模型、阈值、FPGA 协议未为素材采集改动。当前运行包含先前未全部提交的开发文件，以 live_runtime_provenance.json 逐文件散列描述实际版本，不能用 Git HEAD 冒充完整运行版本。','','## 来源统计','','| 来源 | 数量 |','|---|---:|']
    for k,v in counts.items():quality.append(f'| {SOURCE.get(k,k)} | {v} |')
    quality+=['','## 筛选方式','','完整解码、分辨率和 SHA-256 校验；逐字节重复保留原图但不重复推荐。道路 ROI 拉普拉斯清晰度、颜色丰富度与真实目标/事件数量辅助排序。感知哈希在实际动态道路 ROI 上计算，避免静态界面边框主导去重，选择距离见清单。排除切换中、缺少结果帧、离线和未通过硬件检查的候选。自动评分不等于识别准确率，已查看全部精品大尺寸总览，并对红色告警及硬件关键图查看完整原图。日志短暂 critical 而图中已为黄色的画面，不标成红色告警；人工复核近重复项单独记录。','','## 图像处理','','原图逐字节保留；PPT 全图为原生 1280×720，无上采样。项目书版仅裁剪为 1240×615；聚焦版按清单坐标裁剪后放在 1280×720 深色画布，未增加模型或硬件信息。说明只位于文档、网页及总览边缘，不覆盖原图。','','## 未证明的事项','','未声称 Q1 上游根因已解决、四个独立摄像头均实时工作、机械制动已验收、BEV 达到米级精度。截图 FPS 含采集负载，不代替正式整机测量。PCIe 实机指链路与运行环境真实，不等于上游画面是现场道路摄像；未以本次截图证明现场路采。若没有真实识别的红绿灯或斑马线告警，不用合成框补齐。']
    (ROOT/'QUALITY_AND_PROVENANCE.md').write_text('\n'.join(quality)+'\n',encoding='utf-8')
    (ROOT/'manifest.json').write_text(json.dumps(dict(created_at=datetime.datetime.now().astimezone().isoformat(),counts=dict(raw_candidates=len(rows),curated=len(selected),core=len(cores)),topics=TOPICS,candidates=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    keys=['id','file','source_type','captured_at','topic','model','score','detection_count','hazard_count','critical_count','hardware_brake','core_rank','recommended_use','caption_zh','sha256','exclusions']
    with (ROOT/'catalogue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:('；'.join(r[k]) if k=='exclusions' else r.get(k,'')) for k in keys})
    (ROOT/'README.md').write_text('# ADAS Edge 比赛素材库\n\n入口：[素材总览](OVERVIEW.md)；双击 index.html 浏览图片。来源与限制见 [质量与来源](QUALITY_AND_PROVENANCE.md)。\n',encoding='utf-8')

def main():
    for name in ('curated','core','contact_sheets','thumbnails'):
        target=(ROOT/name).resolve()
        assert target.parent==ROOT.resolve() and target.name==name
        if target.exists():shutil.rmtree(target)
        target.mkdir()
    rows=assess(load());groups=select(rows);selected=export(groups);cores=core(selected)
    for r in rows:r['selection']='core' if r in cores else 'curated' if r in selected else 'raw_only'
    contacts={'core':sheets('core','核心推荐 · 本次真实 PCIe',cores)}
    for t,rs in sorted(groups.items()):contacts[t]=sheets(t,TOPICS[t],rs)
    contacts['all']=sheets('all_candidates','全部原始候选 · 来源/筛除原因见标签',rows)
    gallery(rows);reports(rows,selected,cores,groups)
    (ROOT/'contact_sheets/index.json').write_text(json.dumps(contacts,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(raw=len(rows),curated=len(selected),core=len(cores),topics={k:len(v) for k,v in groups.items()}),ensure_ascii=False))
if __name__=='__main__':main()
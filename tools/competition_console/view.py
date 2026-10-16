"""Native automotive display; observations and simulation remain distinct."""
from __future__ import annotations
import importlib.util
from functools import lru_cache
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from competition_ai.scenario import Boundary, Direction, Lane, Light, Maneuver, Mode, Motion
from tools.competition_console.ego_asset import load_ego_asset

ROOT = Path(__file__).resolve().parents[2]
SIZE = (1600, 900)
VIDEO_RECT = (464, 106, 1104, 621)
BG, PANEL, EDGE = '#0b1118', '#141e29', '#263441'
INK, MUTED, ACCENT, RED, AMBER = '#f1f5f7', '#91a2b3', '#80d5cf', '#ff655e', '#ffc577'
LIGHT_TEXT = {Light.RED:'红灯',Light.YELLOW:'黄灯',Light.GREEN:'绿灯',Light.UNKNOWN:'未确认'}
ACTION_TEXT = {Maneuver.STOP:'请停车',Maneuver.BRAKE:'立即制动',Maneuver.FORWARD:'直行',
               Maneuver.TURN_LEFT:'向左转弯',Maneuver.TURN_RIGHT:'向右转弯',
               Maneuver.CHANGE_LEFT:'向左变道',Maneuver.CHANGE_RIGHT:'向右变道'}
ALARM_TEXT = {'RED_LIGHT':'红灯下强制行驶','UNKNOWN_SIGNAL':'灯色未确认',
 'SIGNAL_UNASSOCIATED':'尚未关联控制灯','STALE_SIGNAL':'控制灯观测失效',
 'SOLID_LANE':'实线禁止变道','UNKNOWN_LANE':'车道线未确认',
 'PEDESTRIAN_DANGER':'行人横穿路径','COLLISION_WARNING':'前车快速逼近',
 'STALE_HAZARD':'危险观测失效','WRONG_SCENE':'当前场景不支持此动作',
 'WRONG_LANE':'转向与所选车道不符','YELLOW_LIGHT':'黄灯停车','NO_NAVIGATION':'等待导航指令'}
MODE_TEXT = {Mode.INTERSECTION:'路口通行',Mode.ROAD:'道路变道',Mode.NAVIGATION:'导航辅助',Mode.HAZARD:'危险预警'}


def navigation_display(observation, frame_id, fps, ended=False):
    """Display only an explicitly valid navigation observation from this source.

    The navigation producer owns the 750 ms TTL. This view additionally rejects
    future/missing source identities and never fills unreadable text from a prior
    observation. Route hints are not immediate driving instructions.
    """
    if observation is None:
        return None
    direction = observation.get('direction', 'none')
    source_id, timestamp = observation.get('source_frame_id'), observation.get('timestamp_ns')
    now = round(frame_id/fps*1e9)
    source_valid = (type(source_id) is int and 0 <= source_id <= frame_id
                    and type(timestamp) is int and 0 <= timestamp <= now)
    valid = bool(observation.get('valid')) and source_valid and not ended
    valid = valid and direction in ('left', 'right', 'forward', 'uturn')
    value = observation.get('distance_m')
    distance = None
    if valid and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        distance = f'{value:g} 米' if value < 1000 else f'{value/1000:g} 公里'
    label = {'left':'左转', 'right':'右转', 'forward':'直行', 'uturn':'掉头'}.get(direction, '')
    headline = (f'{distance}后{label}' if distance else f'前方{label}') if valid else '导航暂不可读'
    if ended:
        headline = '导航已停止更新'
    return {'valid': valid, 'direction': direction if valid else 'none',
            'headline': headline, 'road_name': str(observation.get('road_name') or '') if valid else '',
            'source_frame_id': source_id, 'age_ms': (now-timestamp)/1e6 if source_valid else None,
            'reason': str(observation.get('reason') or 'unconfirmed')}

class NativeView:
    def __init__(self, upstream_path=None):
        self.upstream_path = Path(upstream_path or ROOT/'.deps/Tesla-BevCam-Clone/main.py')
        self.upstream = None
        self.fonts, self.buttons = {}, []
        self.content_rect = VIDEO_RECT
        self.ego = load_ego_asset((280, 300))

    def font(self,size,numeric=False):
        key = (size,numeric)
        if key not in self.fonts:
            paths = ([Path('C:/Windows/Fonts/bahnschrift.ttf')] if numeric else []) + [
                Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
                Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
            path = next((p for p in paths if p.exists()),None)
            self.fonts[key] = ImageFont.truetype(str(path),size) if path else ImageFont.load_default(size=size)
        return self.fonts[key]

    @lru_cache(maxsize=768)
    def text_tile(self,value,size,color,numeric,anchor):
        font=self.font(size,numeric)
        bounds=font.getbbox(value,anchor=anchor)
        left,top,right,bottom=bounds
        tile=Image.new('RGBA',(max(1,right-left),max(1,bottom-top)))
        ImageDraw.Draw(tile).text((-left,-top),value,font=font,fill=color,anchor=anchor)
        return tile,left,top

    def _bev(self,items,frame_shape):
        if self.upstream is None:
            spec = importlib.util.spec_from_file_location('competition_tesla_upstream',self.upstream_path)
            if spec is None or spec.loader is None:
                raise RuntimeError('Cannot load Tesla-BevCam renderer')
            self.upstream = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.upstream)
            self.upstream.NATIVE_WINDOW = True
        up = self.upstream
        height,width = frame_shape[:2]
        objects = []
        for item in sorted(items,key=lambda item:item.get('confidence',0),reverse=True)[:18]:
            name = item['class_name']
            if name not in up.DIMS: continue
            x1,y1,x2,y2 = item['bbox']
            if y2 > height*.84 and abs((x1+x2)/2-width/2)<width*.22 and (x2-x1)/max(1,y2-y1)>2.2: continue
            distance = up.H_REAL[name]*up.FOCAL_DEPTH/max(1,(y2-y1)*up.CAM_H/height)
            lateral = (((x1+x2)/2/width-.5)*up.CAM_W)*distance/up.FOCAL_DEPTH
            objects.append({'cls':name,'ego_x':lateral,'ego_y':distance})
        up.update_bev_camera(0.)
        ego = {'lane_source':'NO_MODEL','motion_mode':'NO_POSE','telemetry_valid':False,'lateral_m':0.,'speed_mps':0.}
        result = up.render_tesla_ui(objects,ego,0.,None)
        result[650:,:330] = up.BEV_BG
        result[650:,550:] = up.BEV_BG
        return result[92:]

    def render(self,frame,snapshot,controls,decision,*,frame_id,fps,paused=False,binding=False,message='',
               overlay=True,diagnostics=False,bev=False,total_frames=None,source_name='',
               evidence='VIDEO',perception=None,ended=False,transport_status='PCIe 未连接',drag_box=None,
               navigation_image=None,navigation=None):
        image = Image.new('RGB',SIZE,BG)
        draw = ImageDraw.Draw(image)
        self.buttons = []
        def text(pos,value,size=18,color=INK,numeric=False,anchor=None):
            tile,left,top=self.text_tile(str(value),size,color,numeric,anchor)
            image.paste(tile,(round(pos[0]+left),round(pos[1]+top)),tile)
        def fit_text(value,size,width):
            value=str(value)
            if self.font(size).getlength(value)<=width:return value
            while value and self.font(size).getlength(value+'…')>width:value=value[:-1]
            return value+'…'
        def box(rect,color=PANEL,radius=18,outline=None):
            draw.rounded_rectangle(rect,radius,fill=color,outline=outline)
        def button(rect,label,key,active=False,danger=False,size=18):
            box(rect,'#61302f' if danger and active else '#28524f' if active else PANEL,12,
                RED if danger and active else ACCENT if active else EDGE)
            text(((rect[0]+rect[2])/2,(rect[1]+rect[3])/2),label,size,
                 RED if danger else ACCENT if active else INK,anchor='mm')
            self.buttons.append((key,rect))
        def arrow(cx,cy,direction,color=ACCENT,scale=1.):
            if direction=='uturn':
                width=max(2,round(9*scale))
                draw.arc((cx-25*scale,cy-35*scale,cx+25*scale,cy+15*scale),180,360,fill=color,width=width)
                draw.line((cx+25*scale,cy-10*scale,cx+25*scale,cy+30*scale),fill=color,width=width)
                draw.line((cx-25*scale,cy-10*scale,cx-25*scale,cy+18*scale),fill=color,width=width)
                draw.polygon([(cx-25*scale,cy+34*scale),(cx-43*scale,cy+13*scale),(cx-7*scale,cy+13*scale)],fill=color)
                return
            points = [(0,30),(0,-25),(0,-43)] if direction==Direction.FORWARD else [(0,30),(0,-16),(-38 if direction==Direction.LEFT else 38,-16)]
            draw.line([(cx+x*scale,cy+y*scale) for x,y in points],fill=color,width=max(2,round(9*scale)),joint='curve')
            head = [(0,-48),(-18,-27),(18,-27)] if direction==Direction.FORWARD else [(-49,-16),(-27,-34),(-27,2)] if direction==Direction.LEFT else [(49,-16),(27,-34),(27,2)]
            draw.polygon([(cx+x*scale,cy+y*scale) for x,y in head],fill=color)

        draw.ellipse((26,25,54,53),outline=ACCENT,width=2)
        draw.line((40,30,40,49),fill=ACCENT,width=2)
        text((67,20),'行车辅助',27)
        text((204,29),'本机预演',15,MUTED)
        for index,(mode,key) in enumerate(zip(Mode,('i','d','n','h'))):
            button((460+index*165,18,616+index*165,64),MODE_TEXT[mode],key,controls.mode==mode,size=19)
        button((1154,18,1274,64),'打开视频','o',size=17)
        button((1286,18,1406,64),'诊断','f',diagnostics,size=17)
        button((1418,18,1576,64),'空间示意','c',bev,size=17)
        draw.line((24,82,1576,82),fill=EDGE)
        serious = not ended and any(a in decision.alarms for a in ('RED_LIGHT','SOLID_LANE','PEDESTRIAN_DANGER','COLLISION_WARNING'))
        attention = bool(decision.alarms)
        tint = RED if serious else AMBER if attention else ACCENT
        action = ACTION_TEXT[decision.maneuver]
        if decision.reason in ('selected_lane_signal_unconfirmed','source_signal_expired_or_missing','unknown_signal_stop'): action='等待灯色'
        if 'NO_NAVIGATION' in decision.alarms: action='等待导航'
        if ended: action='播放结束'
        nav = navigation_display(navigation,frame_id,fps,ended)
        navigation_panel = controls.mode==Mode.NAVIGATION and nav is not None

        box((24,106,440,727))
        if navigation_panel:
            text((48,126),'行驶状态',16,MUTED)
            text((48,158),'未测量',32)
            text((48,205),'历史回看' if ended else '仅导航提示',16,MUTED)
        else:
            text((48,126),'模拟车辆',16,MUTED)
            text((48,150),'—' if ended else 'D' if decision.simulated_motion==Motion.GO else 'P',65,numeric=True)
            text((110,181),'历史回看' if ended else '行驶中' if decision.simulated_motion==Motion.GO else '已停止',21)
        text((416,139),'导航提示' if navigation_panel else '系统建议',16,MUTED,anchor='ra')
        if navigation_panel and nav['valid']: arrow(375,209,nav['direction'],ACCENT,.76)
        elif not navigation_panel and decision.direction!=Direction.NONE and not ended: arrow(375,209,decision.direction,tint,.76)
        else:
            draw.ellipse((350,176,398,224),outline=tint,width=3)
            draw.line((362,200,386,200),fill=tint,width=4)
        text((232,254),fit_text(nav['headline'],34,368) if navigation_panel else action,
             34 if navigation_panel else 37,ACCENT if navigation_panel and nav['valid'] else AMBER if navigation_panel else tint,anchor='mm')
        detail = ALARM_TEXT.get(decision.alarms[0],'') if attention else ('控制灯为红色，请保持停止' if snapshot.signal.light==Light.RED else '按选择的模拟车道通行' if controls.mode==Mode.INTERSECTION else '按当前场景规则判断')
        subtitle = nav['road_name'] or ('道路名称未读出' if nav['valid'] else '当前画面没有可确认的导航指令') if navigation_panel else detail
        text((232,299),'已停止更新，结果仅供回看' if ended else fit_text(subtitle,17,368),17,MUTED,anchor='mm')
        if navigation_panel:
            box((44,332,420,676),'#101923',16,EDGE)
            text((62,347),'同帧导航识别',20,ACCENT)
            driving_hint = '灯控待确认' if action=='等待灯色' else action
            if nav['valid'] and nav['direction']=='uturn' and action=='等待导航':
                driving_hint = '掉头执行未接入'
            text((62,382),'行驶建议 · '+driving_hint,18,tint)
            nx,ny,nw,nh=60,420,344,194
            box((nx,ny,nx+nw,ny+nh),'#070e15',8)
            if isinstance(navigation_image,np.ndarray) and navigation_image.ndim==3 and navigation_image.shape[2]==3 and navigation_image.size:
                ratio=min(nw/navigation_image.shape[1],nh/navigation_image.shape[0])
                iw,ih=round(navigation_image.shape[1]*ratio),round(navigation_image.shape[0]*ratio)
                source_pixels=cv2.cvtColor(cv2.resize(navigation_image,(iw,ih)),cv2.COLOR_BGR2RGB)
                image.paste(Image.fromarray(source_pixels),(nx+(nw-iw)//2,ny+(nh-ih)//2))
            else:
                text((232,517),'导航画面不可用',19,MUTED,anchor='mm')
            age=f"{nav['age_ms']:.0f} ms" if nav['age_ms'] is not None else '时间未确认'
            text((62,631),f"识别源帧 {nav['source_frame_id'] if nav['source_frame_id'] is not None else '—'} · {age}",14,MUTED)
            text((62,654),'原视频导航画面 · 路线提示不等于立即转向',13,MUTED)
        else:
            draw.ellipse((76,572,388,643),fill='#0e171f',outline='#233642',width=2)
            draw.arc((68,564,396,651),10,170,fill=tint,width=3)
            image.paste(self.ego,(232-self.ego.width//2,342),self.ego)
            text((232,662),'车辆状态示意 · 姿态未测量',14,MUTED,anchor='mm')
        for index,direction in enumerate((Direction.LEFT,Direction.FORWARD,Direction.RIGHT)):
            active = decision.direction==direction and not ended
            x=128+index*104
            draw.ellipse((x-5,696,x+5,706),fill=ACCENT if active else EDGE)
            text((x+12,692),('左','直','右')[index],14,ACCENT if active else MUTED)

        vx,vy,vw,vh=VIDEO_RECT
        box((vx,vy,vx+vw,vy+vh),'#060a10')
        pixels=self._bev(snapshot.main,frame.shape) if bev else frame.copy()
        if not bev:
            if overlay:
                for item in snapshot.main:
                    x1,y1,x2,y2=map(round,item['bbox'])
                    color=(207,213,128) if item['class_name']!='person' else (119,187,246)
                    length=max(4,min(16,(x2-x1)//3,(y2-y1)//3))
                    for x,dx in ((x1,length),(x2,-length)):
                        for y,dy in ((y1,length),(y2,-length)):
                            cv2.line(pixels,(x,y),(x+dx,y),color,2)
                            cv2.line(pixels,(x,y),(x,y+dy),color,2)
            if snapshot.signal.bbox:
                x1,y1,x2,y2=map(round,snapshot.signal.bbox)
                color={Light.RED:(94,101,255),Light.GREEN:(170,215,128),Light.YELLOW:(119,197,255)}.get(snapshot.signal.light,(180,160,145))
                cv2.rectangle(pixels,(x1,y1),(x2,y2),color,3)
            for event in snapshot.hazards.events:
                x1,y1,x2,y2=map(round,event['bbox'])
                color=(119,197,255) if event['kind']=='LEAD_BRAKE_LAMP_RISE' else (94,101,255)
                cv2.rectangle(pixels,(x1,y1),(x2,y2),color,3)
            if diagnostics and snapshot.hazards.reason!='disabled':
                poly=np.asarray([(round(x*frame.shape[1]),round(y*frame.shape[0])) for x,y in snapshot.hazards.path],np.int32)
                cv2.polylines(pixels,[poly],True,(207,213,128),2)
            if binding and drag_box is not None:
                x1,y1,x2,y2=map(round,drag_box)
                cv2.rectangle(pixels,(x1,y1),(x2,y2),(207,213,128),3)
        factor=min(vw/pixels.shape[1],vh/pixels.shape[0])
        w,h=round(pixels.shape[1]*factor),round(pixels.shape[0]*factor)
        x,y=vx+(vw-w)//2,vy+(vh-h)//2
        self.content_rect=None if bev else (x,y,w,h)
        image.paste(Image.fromarray(cv2.cvtColor(cv2.resize(pixels,(w,h)),cv2.COLOR_BGR2RGB)),(x,y))
        mask=Image.new('L',(vw,vh),0)
        ImageDraw.Draw(mask).rounded_rectangle((0,0,vw-1,vh-1),18,fill=255)
        cropped=image.crop((vx,vy,vx+vw,vy+vh))
        image.paste(Image.new('RGB',(vw,vh),BG),(vx,vy))
        image.paste(cropped,(vx,vy),mask)
        draw=ImageDraw.Draw(image)
        box((484,124,771 if evidence!='VIDEO' else 711,164),'#131d28',10)
        draw.ellipse((498,139,506,147),fill=AMBER if evidence!='VIDEO' else ACCENT)
        text((517,131),'规则用例 · 非视觉识别' if evidence!='VIDEO' else '空间示意' if bev else '前视感知 / 真实视频',17)
        if bev:
            box((784,124,1546,164),'#131d28',10)
            text((802,133),'Tesla-BevCam 原渲染 · 距离未标定 · 车道与自车姿态未接入',16,AMBER)
        if serious:
            draw.rounded_rectangle((vx,vy,vx+vw,vy+vh),18,outline=RED,width=4)
            box((780,181,1258,238),'#932e30',12)
            text((1019,209),ALARM_TEXT[decision.alarms[0]]+(' · 请制动' if decision.maneuver==Maneuver.BRAKE else ' · 操作被禁止'),22,INK,anchor='mm')
        box((484,661,1548,708),'#101b26',12)
        button((492,667,583,702),'继续' if paused else '暂停',' ',size=16)
        seconds=frame_id/fps
        duration=(total_frames or frame_id+1)/fps
        text((599,677),f'{int(seconds)//60:02}:{int(seconds)%60:02}',17,numeric=True)
        draw.line((660,685,912,685),fill=EDGE,width=3)
        draw.line((660,685,660+252*min(1,seconds/max(.001,duration)),685),fill=ACCENT,width=3)
        text((927,677),f'{int(duration)//60:02}:{int(duration)%60:02}',17,MUTED,numeric=True)
        button((985,667,1067,702),'逐帧','.',size=16)
        button((1076,667,1158,702),'重播','r',size=16)
        button((1167,667,1327,702),'取消选灯' if binding else '选择控制灯','b',binding,size=16)
        button((1336,667,1538,702),'检测标记：开' if overlay else '检测标记：关','v',overlay,size=16)
        if binding:
            box((574,579,1472,634),'#203d4b',12)
            text((1023,606),'拖框选择完整横向红 / 黄 / 绿灯组；B 或 Esc 取消',21,INK,anchor='mm')
        elif message and not message.startswith('比赛假设'):
            box((484,604,1548,648),'#20313e',12)
            text((502,614),message[:65],17,AMBER)
        box((24,743,1576,799),PANEL,14)
        stale_signal = 'STALE_SIGNAL' in decision.alarms
        displayed_light = Light.UNKNOWN if ended or stale_signal or not snapshot.signal.associated else snapshot.signal.light
        light_color={Light.RED:RED,Light.GREEN:ACCENT,Light.YELLOW:AMBER}.get(displayed_light,MUTED)
        draw.ellipse((45,764,59,778),fill=light_color)
        text((72,757),'视频已结束' if ended else '控制灯  已过期' if stale_signal else f'控制灯  {LIGHT_TEXT[displayed_light]}',19,light_color)
        if controls.mode==Mode.INTERSECTION:
            text((310,761),'车道关联由场景配置 / 选灯提供',16,MUTED)
        elif controls.mode==Mode.ROAD:
            p=perception or snapshot.as_perception()
            names={Boundary.UNKNOWN:'未确认',Boundary.SOLID:'实线',Boundary.DASHED:'虚线'}
            text((310,759),f'左侧线  {names[p.left_boundary]}     右侧线  {names[p.right_boundary]}',18)
            text((680,761),'车道视觉未接入' if evidence=='VIDEO' else '边界输入来自规则用例',16,AMBER)
        elif controls.mode==Mode.NAVIGATION:
            label={Direction.NONE:'尚未接入',Direction.LEFT:'左转',Direction.RIGHT:'右转',Direction.FORWARD:'直行'}
            if nav is not None:
                text((310,759),fit_text('导航识别  '+nav['headline'],18,430),18,ACCENT if nav['valid'] else AMBER)
                text((760,761),'来源：同一视频内导航屏',16,MUTED)
            else:
                text((310,759),'导航指令  '+label[decision.navigation_hint],18,ACCENT if controls.navigation_valid else AMBER)
                text((650,761),'路线提示与即时转向分开判断',16,MUTED)
        else:
            lamp=any(e['kind']=='LEAD_BRAKE_LAMP_RISE' for e in snapshot.hazards.events)
            text((310,759),'危险规则输入' if evidence!='VIDEO' else '制动灯疑似变亮' if lamp else '图像事件分析' if snapshot.hazards.reason!='disabled' else '危险分析未启用',18,AMBER if lamp or snapshot.hazards.reason=='disabled' else INK)
            text((650,761),'急刹 / 行人闯红灯尚未确认',16,AMBER)
        button((1264,753,1564,789),'场景：有信号灯' if controls.signalized or controls.mode==Mode.INTERSECTION else '场景：无信号灯','t',controls.signalized,size=17)
        text((30,828),'比赛模拟',16,MUTED)
        if controls.mode==Mode.INTERSECTION:
            group=[('左转车道','1',controls.lane==Lane.LEFT),('直行车道','2',controls.lane==Lane.STRAIGHT),('右转车道','3',controls.lane==Lane.RIGHT)]
        elif controls.mode==Mode.ROAD:
            group=[('向左变道','q',controls.request==Maneuver.CHANGE_LEFT),('保持直行','a',controls.request==Maneuver.AUTO),('向右变道','e',controls.request==Maneuver.CHANGE_RIGHT)]
        elif controls.mode==Mode.NAVIGATION:
            group=[('请求左转','j',controls.request==Maneuver.TURN_LEFT),('跟随提示','a',controls.request==Maneuver.AUTO),('请求右转','l',controls.request==Maneuver.TURN_RIGHT)]
        else:
            group=[('路口灯控','i',False),('道路变道','d',False),('保持直行','a',controls.request==Maneuver.AUTO)]
        for index,(label,key,active) in enumerate(group):
            button((130+index*166,818,286+index*166,867),label,key,active)
        text((674,833),'车辆状态',16,MUTED)
        for index,(label,key,active) in enumerate((('自动模拟','0',controls.motion==Motion.AUTO),('停止','s',controls.motion==Motion.STOP),('强制行驶','g',controls.motion==Motion.GO))):
            button((776+index*178,818,944+index*178,867),label,key,active,key=='g')
        text((1570,829),'输出：本机预览',16,MUTED,anchor='ra')
        text((1570,853),transport_status,14,AMBER,anchor='ra')
        text((28,879),'模拟按键只改变建议与报警，不改变原视频中的车辆运动。',12,MUTED)
        text((1573,879),'前视优先 / RK3566 待联调',12,MUTED,anchor='ra')
        if diagnostics:
            box((766,181,1548,633),'#111e2b',16,EDGE)
            text((792,198),'诊断与证据来源',26)
            navigation_source = '导航解析：未接入' if nav is None else '导航解析：同帧导航屏识别 / '+('有效' if nav['valid'] else '未确认')
            navigation_time = ('导航源帧：'+str(nav['source_frame_id'])+' / 龄期：'+(f"{nav['age_ms']:.0f} ms" if nav['age_ms'] is not None else '未确认')+' / '+nav['reason']) if nav is not None else '车道视觉：未接入    导航解析：未接入'
            rows=[f'视频：{Path(source_name).name or "未指定"}',
                  f'源帧 {frame_id}   视频时间 {seconds:.2f} s   输入 {frame.shape[1]} × {frame.shape[0]}',
                  f'主模型目标 {len(snapshot.main)}   灯专项候选 {len(snapshot.specialist)}',
                  f'结果龄期：主 {snapshot.main_age_ms} ms / 灯专项 {snapshot.specialist_age_ms} ms',
                  '灯组来源：'+snapshot.binding_source,'灯色判据：'+snapshot.signal.reason,
                  navigation_source,navigation_time,'危险事件：'+snapshot.hazards.reason,
                  '控制输出：'+transport_status+'；未写真实硬件',
                  '推理环境：PC ONNX FP32；未验证板端联合性能']
            for index,line in enumerate(rows):text((792,245+index*28),fit_text(line,16,730),16,MUTED)
            text((792,575),'图像路径是实验假设，未标定距离；灯组关联来自显式配置。',16,AMBER)
            text((792,605),'F 或 Esc 关闭诊断',15)
            self.buttons=[(key,rect) for key,rect in self.buttons if not(rect[1]<633 and rect[3]>181 and rect[2]>766)]
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    def hit(self,x,y):
        return next((key for key,(x1,y1,x2,y2) in self.buttons if x1<=x<=x2 and y1<=y<=y2),None)

    def video_point(self,x,y,shape):
        if self.content_rect is None:return None
        vx,vy,vw,vh=self.content_rect
        if vx<=x<vx+vw and vy<=y<vy+vh and 170<=y<650:
            return ((x-vx)/vw*shape[1],(y-vy)/vh*shape[0])
        return None

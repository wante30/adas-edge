"""Native pixels and controls; the BEV pixels come from upstream main.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from competition_ai.scenario import Direction, Lane, Light, Maneuver, Mode, Motion

ROOT = Path(__file__).resolve().parents[2]
VIDEO_RECT = (24,104,940,529)
SIZE = (1440,900)
BACKGROUND = '#eef2f5'
INK = '#182b3a'
BLUE = '#246fa7'
MUTED = '#607684'
LIGHT_TEXT = {Light.RED:'红灯',Light.YELLOW:'黄灯',Light.GREEN:'绿灯',Light.UNKNOWN:'未知'}
ACTION_TEXT = {Maneuver.STOP:'停车',Maneuver.BRAKE:'制动',Maneuver.FORWARD:'直行',
               Maneuver.TURN_LEFT:'左转',Maneuver.TURN_RIGHT:'右转',
               Maneuver.CHANGE_LEFT:'左变道',Maneuver.CHANGE_RIGHT:'右变道'}
ALARM_TEXT = {'RED_LIGHT':'闯红灯报警','UNKNOWN_SIGNAL':'灯色尚未确认','SIGNAL_UNASSOCIATED':'所选车道尚未绑定灯组',
              'STALE_SIGNAL':'灯组结果过期或未确认','SOLID_LANE':'实线变道报警','UNKNOWN_LANE':'车道类型尚未确认',
              'PEDESTRIAN_DANGER':'行人横穿路径：制动','COLLISION_WARNING':'前车快速逼近：制动',
              'STALE_HAZARD':'危险结果未确认','WRONG_SCENE':'动作与场景不匹配','WRONG_LANE':'转弯与所选车道不匹配'}


class NativeView:
    def __init__(self, upstream_path=None):
        path = Path(upstream_path or ROOT / '.deps/Tesla-BevCam-Clone/main.py')
        spec = importlib.util.spec_from_file_location('competition_tesla_upstream',path)
        if spec is None or spec.loader is None:
            raise RuntimeError('Cannot load Tesla-BevCam upstream renderer')
        self.upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.upstream)
        self.upstream.NATIVE_WINDOW = True
        self.fonts = {}
        self.buttons = {}
        for size in (14,16,18,20,22,28,42,52):
            candidates = [Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')]
            font = next((p for p in candidates if p.exists()),None)
            self.fonts[size] = ImageFont.truetype(str(font),size) if font else ImageFont.load_default(size=size)

    def _bev(self, items, frame_shape):
        up = self.upstream
        height,width = frame_shape[:2]
        objects = []
        for item in sorted(items,key=lambda item:item.get('confidence',0),reverse=True)[:18]:
            name = item['class_name']
            if name not in up.DIMS:
                continue
            x1,y1,x2,y2 = item['bbox']
            if y2 > height*.84 and abs((x1+x2)/2-width/2) < width*.22 and (x2-x1)/max(1,y2-y1) > 2.2:
                continue
            distance = up.H_REAL[name]*up.FOCAL_DEPTH/max(1,(y2-y1)*up.CAM_H/height)
            lateral = (((x1+x2)/2/width-.5)*up.CAM_W)*distance/up.FOCAL_DEPTH
            objects.append({'cls':name,'ego_x':lateral,'ego_y':distance})
        up.update_bev_camera(0.)
        ego = {'lane_source':'NO_MODEL','motion_mode':'NO_POSE','telemetry_valid':False,'lateral_m':0.,'speed_mps':0.}
        bev = up.render_tesla_ui(objects,ego,0.,None)
        # The telemetry labels are not observations in this front-video test.
        bev[650:,:330] = up.BEV_BG
        bev[650:,550:] = up.BEV_BG
        return bev[92:]

    def render(self, frame, snapshot, controls, decision, *, frame_id, fps, paused=False, binding=False, message=''):
        canvas = np.full((SIZE[1],SIZE[0],3),240,np.uint8)
        canvas[:] = (245,242,238)
        x,y,w,h = VIDEO_RECT
        camera = cv2.resize(frame,(w,h))
        sx,sy = w/frame.shape[1],h/frame.shape[0]
        for item in snapshot.main:
            x1,y1,x2,y2 = item['bbox']
            colour = (215,165,55) if item['class_name'] != 'person' else (90,135,230)
            cv2.rectangle(camera,(round(x1*sx),round(y1*sy)),(round(x2*sx),round(y2*sy)),colour,1)
        hazard = snapshot.hazards
        if hazard.reason != 'disabled':
            polygon = np.asarray([(round(px*w),round(py*h)) for px,py in hazard.path],np.int32)
            cv2.polylines(camera,[polygon],True,(230,165,45),2)
            for item in hazard.tracks:
                x1,y1,x2,y2 = item['bbox']
                label = f"ID {item['id']}"+(' LEAD' if item['id']==hazard.lead_id else '')
                cv2.putText(camera,label,(round(x1*sx),max(15,round(y1*sy)-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,165,45),1)
            for event in hazard.events:
                x1,y1,x2,y2 = event['bbox']
                colour = (30,170,235) if event['kind']=='LEAD_BRAKE_LAMP_RISE' else (55,55,235)
                cv2.rectangle(camera,(round(x1*sx),round(y1*sy)),(round(x2*sx),round(y2*sy)),colour,3)
        if snapshot.signal.bbox:
            x1,y1,x2,y2 = snapshot.signal.bbox
            colour = {Light.RED:(55,55,235),Light.GREEN:(90,180,35),Light.YELLOW:(35,180,225)}.get(snapshot.signal.light,(130,130,130))
            cv2.rectangle(camera,(round(x1*sx),round(y1*sy)),(round(x2*sx),round(y2*sy)),colour,3)
        canvas[y:y+h,x:x+w] = camera
        bev = self._bev(snapshot.main,frame.shape)
        bev = cv2.resize(bev,(408,188))
        canvas[674:862,998:1406] = bev
        image = Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        def text(position,value,size=18,colour=INK):
            draw.text(position,value,font=self.fonts[size],fill=colour)
        def box(rect,fill='#ffffff',outline=None,radius=12):
            draw.rounded_rectangle(rect,radius,fill=fill,outline=outline,width=2)
        def button(rect,label,key,active):
            box(rect,BLUE if active else '#e4ebf0')
            text((rect[0]+13,rect[1]+10),label,18,'#ffffff' if active else INK)
            self.buttons[key] = rect
        self.buttons = {}
        danger = any(alarm in decision.alarms for alarm in ('PEDESTRIAN_DANGER','COLLISION_WARNING'))
        if 'RED_LIGHT' in decision.alarms or danger:
            draw.rectangle((x,y,x+w,y+h),outline='#cc343e',width=5)
            box((x+18,y+18,x+264,y+68),'#bb3039',radius=8)
            text((x+33,y+29),'危险报警 · 请制动' if danger else '闯红灯报警 · 请停车',22,'#ffffff')
        text((24,19),'竞赛感知测试台',28)
        text((25,57),'真实前视视频 · 按键模拟车辆 · 动作与报警',16,MUTED)
        mode_name = {Mode.INTERSECTION:'路口灯控',Mode.ROAD:'道路变道',Mode.NAVIGATION:'导航提示',Mode.HAZARD:'危险场景'}[controls.mode]
        box((1136,20,1416,71),'#dce8f1')
        text((1152,33),f'PC 本机  /  {mode_name}',20,BLUE)
        text((24,84),'前视视频与主模型检测',16,MUTED)
        text((790,84),'已暂停' if paused else '播放中',16,BLUE)
        seconds = frame_id/fps
        text((25,639),f'原视频 {seconds:05.2f} s  ·  源帧 {frame_id}  ·  主模型 {len(snapshot.main)} 个目标  ·  灯专项 {len(snapshot.specialist)} 个候选',16,MUTED)
        if binding:
            text((25,659),'请拖框选择完整的横向红/黄/绿灯组；仅绑定当前所选车道',16,BLUE)
        elif message:
            text((25,659),message[:66],16,BLUE)
        else:
            text((25,659),'灯色来自绑定灯位的像素判别；车道关联由比赛场景配置或鼠标绑定提供。',16,MUTED)
        box((24,692,964,879))
        text((42,700),'模拟车辆所在车道',18)
        for index,(lane,label) in enumerate(((Lane.LEFT,'1  左转车道'),(Lane.STRAIGHT,'2  直行车道'),(Lane.RIGHT,'3  右转车道'))):
            button((42+index*184,735,210+index*184,779),label,str(index+1),controls.lane==lane)
        text((626,702),'模拟车辆状态',18)
        for index,(motion,label,key) in enumerate(((Motion.AUTO,'0 自动','0'),(Motion.STOP,'S 停止','s'),(Motion.GO,'G 强制行驶','g'))):
            button((626,732+index*46,945,773+index*46),label,key,controls.motion==motion)
        button((42,805,165,851),'O 打开视频','o',False)
        button((179,805,325,851),'空格 暂停',' ',paused)
        button((339,805,486,851),'B 绑定灯组','b',binding)
        button((500,805,607,851),'R 重播','r',False)
        box((988,104,1416,531))
        text((1008,121),'系统建议',22)
        colour = {Light.RED:'#ce4148',Light.GREEN:'#2e9964',Light.YELLOW:'#cb9a24'}.get(snapshot.signal.light,'#8496a2')
        draw.ellipse((1270,128,1286,144),fill=colour)
        text((1295,118),LIGHT_TEXT[snapshot.signal.light],20,colour)
        action = ACTION_TEXT[decision.maneuver]
        if decision.direction == Direction.NONE:
            box((1063,183,1341,313),'#f7e5e6' if snapshot.signal.light==Light.RED else '#e9eef2')
            text((1149 if len(action)==2 else 1120,218),action,52,colour if snapshot.signal.light==Light.RED else INK)
        else:
            draw.line((1202,306,1202,214),fill=BLUE,width=22)
            if decision.direction == Direction.FORWARD:
                draw.polygon([(1202,180),(1168,230),(1236,230)],fill=BLUE)
            else:
                sign = -1 if decision.direction==Direction.LEFT else 1
                draw.line((1202,220,1202+sign*72,220),fill=BLUE,width=22)
                draw.polygon([(1202+sign*96,220),(1202+sign*48,188),(1202+sign*48,252)],fill=BLUE)
            text((1148 if len(action)==2 else 1126,321),action,42,BLUE)
        text((1008,390),'建议指示灯（PC 模拟）',16,MUTED)
        for index,label in enumerate(('LED0 左','LED1 直','LED2 右')):
            active = bool(decision.led_mask & (1<<index))
            cx = 1060+index*139
            draw.ellipse((cx-10,424,cx+10,444),fill='#2e9964' if active else '#d5dfe6')
            text((cx-34,451),label,16,MUTED)
        actual = '行驶' if decision.simulated_motion==Motion.GO else '停止'
        text((1008,492),f'模拟状态：{actual}   请求许可：'+('允许' if decision.request_allowed else '未授予'),18)
        alert = decision.alarms
        box((988,546,1416,620),'#f9e1e2' if 'RED_LIGHT' in alert or danger else '#fff1d4' if alert else '#e1f0e8')
        text((1008,555),ALARM_TEXT.get(alert[0],alert[0]) if alert else '无报警',20,'#b72f3a' if 'RED_LIGHT' in alert else INK)
        reason = {'observed':'绑定灯组像素观测','unbound':'请绑定所选车道的灯组',
                  'colour_unknown_or_confirming':'灯色未知或正在确认','observation_expired':'旧观测已过期',
                  'housing_appearance_mismatch':'灯组跟踪丢失，需重新绑定'}.get(snapshot.signal.reason,'灯组等待确认')
        if danger:
            reason = '视频轨迹触发 · 蓝色路径为图像假设'
        text((1008,586),reason,14,MUTED)
        lamp_hint = any(e['kind']=='LEAD_BRAKE_LAMP_RISE' for e in hazard.events)
        text((988,635),('制动灯疑似变亮 · 急刹未确认' if lamp_hint else '急刹车：未确认  /  行人闯红灯：未确认') if hazard.reason!='disabled' else 'Tesla-BevCam 原版 BEV 示意',14,MUTED)
        if hazard.reason!='disabled':
            status = '轨迹判断中' if hazard.reason=='tracking' else '转弯/背景未确认：暂缓判断'
            text((988,654),status+' · 实验模式',14,BLUE)
        text((1000,863),'位置未标定 · 未接自车姿态与车道模型',14,MUTED)
        text((25,881),'I 路口 / D 道路    Q / E 左右变道（车道结果待接入）    . 逐帧    Esc 退出',14,MUTED)
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    def hit(self,x,y):
        return next((key for key,(x1,y1,x2,y2) in self.buttons.items() if x1<=x<=x2 and y1<=y<=y2),None)

    @staticmethod
    def video_point(x,y,shape):
        vx,vy,vw,vh = VIDEO_RECT
        if vx<=x<=vx+vw and vy<=y<=vy+vh:
            return ((x-vx)/vw*shape[1],(y-vy)/vh*shape[0])
        return None

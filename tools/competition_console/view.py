"""Native automotive display; observations and simulation remain distinct."""
from __future__ import annotations
import importlib.util
from functools import lru_cache
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from competition_ai.scenario import Boundary, Light

ROOT = Path(__file__).resolve().parents[2]
SIZE = (1600, 900)
VIDEO_RECT = (464, 106, 1104, 621)
BG, PANEL, EDGE = '#0b1118', '#141e29', '#263441'
INK, MUTED, ACCENT, RED, AMBER = '#f1f5f7', '#91a2b3', '#80d5cf', '#ff655e', '#ffc577'
LIGHT_TEXT = {Light.RED:'红灯',Light.YELLOW:'黄灯',Light.GREEN:'绿灯',Light.UNKNOWN:'未确认'}


def font_text_width(font, value):
    measure = getattr(font, 'getlength', None)
    if measure is not None:
        return measure(value)
    return font.getsize(value)[0]


def font_bbox(font, value, anchor=None):
    measure = getattr(font, 'getbbox', None)
    if measure is not None:
        try:
            return measure(value, anchor=anchor) if anchor else measure(value)
        except TypeError:
            return measure(value)
    width, height = font.getsize(value)
    left, top = font.getoffset(value) if hasattr(font, 'getoffset') else (0, 0)
    horizontal, vertical = (anchor or 'lt')[:2]
    if horizontal == 'm':
        left -= width / 2
    elif horizontal == 'r':
        left -= width
    if vertical == 'm':
        top -= height / 2
    elif vertical in ('b', 'd'):
        top -= height
    left, top = round(left), round(top)
    return left, top, left + width, top + height


def navigation_display(observation, frame_id, fps, ended=False, source_timestamp_ns=None):
    """Display only an explicitly valid navigation observation from this source.

    The navigation producer owns the 750 ms TTL. This view additionally rejects
    future/missing source identities and never fills unreadable text from a prior
    observation. Route hints are not immediate driving instructions.
    """
    if observation is None:
        return None
    direction = observation.get('direction', 'none')
    source_id, timestamp = observation.get('source_frame_id'), observation.get('timestamp_ns')
    if type(source_timestamp_ns) is int and source_timestamp_ns >= 0:
        now = source_timestamp_ns
    elif isinstance(fps,(int,float)) and math.isfinite(fps) and fps > 0:
        now = round(frame_id/fps*1e9)
    else:
        now = None
    source_valid = (now is not None and type(source_id) is int and 0 <= source_id <= frame_id
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
        # Fixed-size chrome is reused; neither image is mutated during rendering.
        self._background = Image.new('RGB',SIZE,BG)
        vx, vy, width, height = VIDEO_RECT
        self._video_mask = Image.new('L',(width,height),0)
        mask_draw = ImageDraw.Draw(self._video_mask)
        if hasattr(mask_draw, 'rounded_rectangle'):
            mask_draw.rounded_rectangle((0,0,width-1,height-1),18,fill=255)
        else:
            mask_draw.rectangle((0,0,width-1,height-1),fill=255)
        self._video_clear = Image.new('RGB',(width,height),BG)
        self._video_corners = []
        for x,y in ((0,0),(width-18,0),(0,height-18),(width-18,height-18)):
            bounds=(x,y,x+18,y+18)
            outside=self._video_mask.crop(bounds).point(lambda value:255-value)
            self._video_corners.append(((vx+x,vy+y),self._video_clear.crop(bounds),outside))

    def font(self,size,numeric=False):
        key = (size,numeric)
        if key not in self.fonts:
            paths = ([Path('C:/Windows/Fonts/bahnschrift.ttf')] if numeric else []) + [
                Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
                Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
            path = next((p for p in paths if p.exists()),None)
            if path:
                self.fonts[key] = ImageFont.truetype(str(path),size)
            else:
                try:
                    self.fonts[key] = ImageFont.load_default(size=size)
                except TypeError:
                    self.fonts[key] = ImageFont.load_default()
        return self.fonts[key]

    @lru_cache(maxsize=768)
    def text_tile(self,value,size,color,numeric,anchor):
        font=self.font(size,numeric)
        bounds=font_bbox(font,value)
        placement=font_bbox(font,value,anchor)
        left,top,right,bottom=bounds
        tile=Image.new('RGBA',(max(1,right-left),max(1,bottom-top)))
        # Older Pillow releases ignore `anchor` in ImageDraw.text. Draw into a
        # glyph-sized tile without it, then position that tile using the
        # version-compatible anchor bounds above.
        ImageDraw.Draw(tile).text((-left,-top),value,font=font,fill=color)
        return tile,placement[0],placement[1]

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

    def render(self,frame,snapshot,controls=None,decision=None,*,frame_id,fps,paused=False,binding=False,message='',
               overlay=True,diagnostics=False,bev=False,total_frames=None,source_name='',
               evidence='VIDEO',perception=None,ended=False,transport_status='PCIe 未连接',drag_box=None,
               navigation_image=None,navigation=None,runtime_status='PC ONNX FP32',output_rgb=False,
               source_timestamp_ns=None,source_start_timestamp_ns=None,live=False,fpga_perception=None,
               test_status=None):
        image = self._background.copy()
        draw = ImageDraw.Draw(image)
        self.buttons = []
        def text(pos,value,size=18,color=INK,numeric=False,anchor=None):
            tile,left,top=self.text_tile(str(value),size,color,numeric,anchor)
            image.paste(tile,(round(pos[0]+left),round(pos[1]+top)),tile)
        def fit_text(value,size,width):
            value=str(value)
            if font_text_width(self.font(size),value)<=width:return value
            while value and font_text_width(self.font(size),value+'…')>width:value=value[:-1]
            return value+'…'
        def box(rect,color=PANEL,radius=18,outline=None):
            rounded = getattr(draw, 'rounded_rectangle', None)
            if rounded is not None:
                rounded(rect,radius,fill=color,outline=outline)
            else:
                draw.rectangle(rect,fill=color,outline=outline)
        def button(rect,label,key,active=False,danger=False,size=18):
            box(rect,'#61302f' if danger and active else '#28524f' if active else PANEL,12,
                RED if danger and active else ACCENT if active else EDGE)
            text(((rect[0]+rect[2])/2,(rect[1]+rect[3])/2),label,size,
                 RED if danger else ACCENT if active else INK,anchor='mm')
            if key is not None:
                self.buttons.append((key,rect))
        draw.ellipse((26,25,54,53),outline=ACCENT,width=2)
        draw.line((40,30,40,49),fill=ACCENT,width=2)
        text((67,20),'前视感知',27)
        text((204,29),runtime_status,15,MUTED)
        if not live:
            button((1154,18,1274,64),'打开视频','o',size=17)
        button((1286,18,1406,64),'诊断','f',diagnostics,size=17)
        button((1418,18,1576,64),'空间示意','c',bev,size=17)
        draw.line((24,82,1576,82),fill=EDGE)
        nav = navigation_display(navigation,frame_id,fps,ended,source_timestamp_ns)
        if fpga_perception is not None:
            traffic = fpga_perception.get('traffic') if isinstance(fpga_perception,dict) else None
            evidence_name = fpga_perception.get('traffic_evidence') if isinstance(fpga_perception,dict) else None
            if traffic == 'RED':
                displayed_light = Light.RED
            elif traffic == 'GREEN':
                displayed_light = Light.GREEN
            elif evidence_name == 'YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL':
                displayed_light = Light.YELLOW
            else:
                displayed_light = Light.UNKNOWN
        else:
            displayed_light = snapshot.signal.light if snapshot.signal.associated else Light.UNKNOWN
        if ended:
            displayed_light = Light.UNKNOWN
        signal_status = LIGHT_TEXT[displayed_light]
        if fpga_perception is not None:
            signal_reasons = {
                'NO_SIGNAL_ROI':'未检出信号灯候选',
                'NO_CONFIDENT_SIGNAL_ROI':'候选置信度不足',
                'SIGNAL_COLOR_UNCONFIRMED':'灯框已检出，颜色证据不足',
                'CONFLICTING_CONFIRMED_SIGNAL_COLORS':'多个灯色冲突，保持 OFF',
                'MULTIPLE_SIGNAL_ROIS_UNASSOCIATED':'多个候选灯框未确认',
                'YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL':'已识别黄灯，HUD 协议不支持',
                'USER_BOUND_SIGNAL_COLOR_UNCONFIRMED':'绑定区域颜色未确认',
                'USER_BOUND_YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL':'已识别黄灯，HUD 协议不支持',
                'USER_BOUND_RED_SIGNAL':'人工绑定颜色已确认',
                'USER_BOUND_GREEN_SIGNAL':'人工绑定颜色已确认',
                'ONE_DETECTED_RED_SIGNAL':'检测框颜色已确认',
                'ONE_DETECTED_GREEN_SIGNAL':'检测框颜色已确认',
                'MULTIPLE_RED_SIGNALS_AGREE':'多个灯框颜色一致',
                'MULTIPLE_GREEN_SIGNALS_AGREE':'多个灯框颜色一致',
            }
            evidence_name = fpga_perception.get('traffic_evidence') if isinstance(fpga_perception,dict) else None
            signal_reason = signal_reasons.get(evidence_name)
            if signal_reason is None and isinstance(evidence_name,str) and evidence_name.endswith('_WITH_UNRESOLVED_CANDIDATES'):
                signal_reason = '颜色已确认，另有候选未确认'
            if signal_reason is None:
                signal_reason = '颜色已确认' if displayed_light in (Light.RED,Light.GREEN) else '颜色未确认'
        else:
            signal_reasons = {'observed':'当前观测','awaiting_colour':'等待颜色观测',
                              'colour_unknown_or_confirming':'颜色未确认',
                              'observation_expired':'观测已过期','unbound':'未关联观测'}
            signal_reason = signal_reasons.get(snapshot.signal.reason,'观测状态未确认')
        fpga_lanes = fpga_perception if isinstance(fpga_perception,dict) else {}
        lane_values = {'NONE':'未检出','SOLID':'实线','DASHED':'虚线','0':'未检出',
                       '1':'实线','2':'虚线'}
        left_boundary = lane_values.get(str(fpga_lanes.get('lane_left','')).upper(),'未关联')
        right_boundary = lane_values.get(str(fpga_lanes.get('lane_right','')).upper(),'未关联')

        box((24,106,440,727))
        text((48,126),'感知状态',22)
        text((416,134),'历史回看' if ended else '实时观测',15,MUTED,anchor='ra')
        box((44,169,420,270),'#101923',14,EDGE)
        text((62,181),'信号灯识别',16,MUTED)
        text((62,207),'信号灯 '+signal_status,29,ACCENT if signal_status!='未确认' else AMBER)
        text((62,246),signal_reason,14,MUTED)

        box((44,286,420,390),'#101923',14,EDGE)
        text((62,299),'道路目标',16,MUTED)
        text((402,296),f'{len(snapshot.main)} 个',25,INK,numeric=True,anchor='ra')
        class_names={'person':'行人','bicycle':'自行车','car':'汽车','motorcycle':'摩托车',
                     'bus':'公交车','truck':'货车'}
        counts={}
        for item in snapshot.main:
            name=class_names.get(str(item.get('class_name','未知')),str(item.get('class_name','未知')))
            counts[name]=counts.get(name,0)+1
        class_summary=' · '.join(f'{name} {count}' for name,count in sorted(counts.items())) or '未检出目标'
        text((62,350),fit_text(class_summary,15,336),15,MUTED)

        box((44,406,420,491),'#101923',14,EDGE)
        text((62,419),'危险事件',16,MUTED)
        hazard_count=len(snapshot.hazards.events)
        hazard_text='分析未启用' if snapshot.hazards.reason=='disabled' else f'{hazard_count} 项' if hazard_count else '未检出'
        text((402,420),hazard_text,22,AMBER if hazard_count else MUTED,anchor='ra')
        event_names={'LEAD_BRAKE_LAMP_RISE':'前车制动灯变化','PEDESTRIAN_CROSSING':'行人横穿'}
        events=' · '.join(event_names.get(event.get('kind'),str(event.get('kind','未知事件')))
                          for event in snapshot.hazards.events[:2])
        if events:
            text((62,461),fit_text(events,14,336),14,MUTED)

        box((44,507,420,700),'#101923',14,EDGE)
        text((62,520),'同帧导航识别' if nav is not None else '导航识别',16,MUTED)
        if nav is not None:
            text((62,547),fit_text(nav['headline'],23,336),23,ACCENT if nav['valid'] else AMBER)
            if nav['road_name']:
                text((62,581),fit_text(nav['road_name'],16,336),16,INK)
            source_id = nav['source_frame_id'] if nav['source_frame_id'] is not None else '未确认'
            source_age = f"{nav['age_ms']:.0f} ms" if nav['age_ms'] is not None else '未确认'
            text((62,603),f'识别源帧 {source_id} · {source_age}',13,MUTED)
            nx,ny,nw,nh=60,625,344,64
            box((nx,ny,nx+nw,ny+nh),'#070e15',8)
            if isinstance(navigation_image,np.ndarray) and navigation_image.ndim==3 and navigation_image.shape[2]==3 and navigation_image.size:
                ratio=min(nw/navigation_image.shape[1],nh/navigation_image.shape[0])
                iw,ih=round(navigation_image.shape[1]*ratio),round(navigation_image.shape[0]*ratio)
                source_pixels=cv2.cvtColor(cv2.resize(navigation_image,(iw,ih)),cv2.COLOR_BGR2RGB)
                image.paste(Image.fromarray(source_pixels),(nx+(nw-iw)//2,ny+(nh-ih)//2))
            else:
                text((62,642),fit_text(nav['road_name'] or '导航文本未确认',14,336),14,MUTED)
        else:
            text((62,554),'导航信息未接入',18,MUTED)
        if ended:
            text((62,706),'视频结束 · 显示结果仅供回看',13,MUTED)

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
        factor=min(vw/pixels.shape[1],vh/pixels.shape[0])
        w,h=round(pixels.shape[1]*factor),round(pixels.shape[0]*factor)
        x,y=vx+(vw-w)//2,vy+(vh-h)//2
        self.content_rect=None if bev else (x,y,w,h)
        image.paste(Image.fromarray(cv2.cvtColor(cv2.resize(pixels,(w,h)),cv2.COLOR_BGR2RGB)),(x,y))
        # The old clear-and-mask composite changes only pixels outside the
        # rounded rectangle. Those pixels fit in its four 18 x 18 corners.
        for position,clear,outside in self._video_corners:
            image.paste(clear,position,outside)
        draw=ImageDraw.Draw(image)
        box((484,124,771 if evidence!='VIDEO' else 711,164),'#131d28',10)
        draw.ellipse((498,139,506,147),fill=AMBER if evidence!='VIDEO' else ACCENT)
        text((517,131),'规则用例 · 非视觉识别' if evidence!='VIDEO' else '空间示意' if bev else '前视感知 / 真实视频',17)
        if bev:
            box((784,124,1546,164),'#131d28',10)
            text((802,133),'Tesla-BevCam 原渲染 · 距离未标定 · 车道与自车姿态未接入',16,AMBER)
        valid_fps=isinstance(fps,(int,float)) and math.isfinite(fps) and fps>0
        if (type(source_timestamp_ns) is int and source_start_timestamp_ns is not None
                and type(source_start_timestamp_ns) is int):
            seconds=max(0,source_timestamp_ns-source_start_timestamp_ns)/1e9
        else:
            seconds=frame_id/fps if valid_fps else 0.
        duration=(total_frames/fps) if total_frames is not None and valid_fps else None
        if not live:
            box((484,661,1548,708),'#101b26',12)
            button((492,667,583,702),'继续' if paused else '暂停',' ',size=16)
            text((599,677),f'{int(seconds)//60:02}:{int(seconds)%60:02}',17,numeric=True)
            draw.line((660,685,912,685),fill=EDGE,width=3)
            if duration is not None:
                draw.line((660,685,660+252*min(1,seconds/max(.001,duration)),685),fill=ACCENT,width=3)
                right_time=f'{int(duration)//60:02}:{int(duration)%60:02}'
            else:
                right_time='LIVE'
            text((927,677),right_time,17,MUTED,numeric=duration is not None)
            button((985,667,1067,702),'逐帧','.',size=16)
            button((1076,667,1158,702),'重播','r',size=16)
        button((1336,667,1538,702),'检测标记：开' if overlay else '检测标记：关','v',overlay,size=16)
        if message and not message.startswith('比赛假设'):
            box((484,604,1548,648),'#20313e',12)
            text((502,614),message[:65],17,AMBER)
        box((24,743,1576,799),PANEL,14)
        light_color={Light.RED:RED,Light.GREEN:ACCENT,Light.YELLOW:AMBER}.get(displayed_light,MUTED)
        draw.ellipse((45,764,59,778),fill=light_color)
        text((72,757),'视频结束' if ended else f'检测灯色  {signal_status}',19,light_color)
        text((310,757),f'左边界  {left_boundary}',18,MUTED)
        text((490,757),f'右边界  {right_boundary}',18,MUTED)
        text((670,757),'车道关联  未关联',17,AMBER)
        nav_summary=nav['headline'] if nav is not None else '导航信息未接入'
        text((865,757),fit_text('导航  '+nav_summary,18,500),18,ACCENT if nav and nav['valid'] else MUTED)
        text((1570,757),transport_status,15,AMBER,anchor='ra')
        text((28,828),'只展示感知与导航信息；控制信号由 FPGA 处理。',16,MUTED)
        text((1573,853),'前视优先 / '+runtime_status,12,MUTED,anchor='ra')
        if diagnostics:
            box((766,181,1548,633),'#111e2b',16,EDGE)
            text((792,198),'诊断与证据来源',26)
            navigation_source = '导航解析：未接入' if nav is None else '导航解析：同帧导航屏识别 / '+('有效' if nav['valid'] else '未确认')
            navigation_time = ('导航源帧：'+str(nav['source_frame_id'])+' / 龄期：'+(f"{nav['age_ms']:.0f} ms" if nav['age_ms'] is not None else '未确认')+' / '+nav['reason']) if nav is not None else '车道视觉：未接入    导航解析：未接入'
            rows=[f'视频：{Path(source_name).name or "未指定"}',
                  f'源帧 {frame_id}   视频时间 {seconds:.2f} s   输入 {frame.shape[1]} × {frame.shape[0]}',
                  f'主模型目标 {len(snapshot.main)}   灯专项候选 {len(snapshot.specialist)}',
                  f'结果龄期：主 {snapshot.main_age_ms} ms / 灯专项 {snapshot.specialist_age_ms} ms',
                  '灯组来源：'+snapshot.binding_source,'灯色判据：'+signal_reason,
                  navigation_source,navigation_time,'危险事件：'+snapshot.hazards.reason,
                  'FPGA 链路：'+transport_status,
                  '推理环境：'+runtime_status]
            for index,line in enumerate(rows):text((792,245+index*25),fit_text(line,16,730),16,MUTED)
            text((792,608),'显示内容来自当前帧；未知项目保持未确认。F 或 Esc 关闭诊断',14,AMBER)
            self.buttons=[(key,rect) for key,rect in self.buttons if not(rect[1]<633 and rect[3]>181 and rect[2]>766)]
            if live and test_status is not None:
                enabled = bool(test_status.get('enabled'))
                phase = str(test_status.get('phase','DISABLED'))
                text((792,514),f'FPGA 手动测试：{"已启用" if enabled else "未启用"} / {phase}',16,
                     AMBER if phase in ('BLOCKED','REJECTED','TIMEOUT') else MUTED)
                button((792,538,970,582),'禁用测试' if enabled else '启用测试','fpga_test_enable',enabled,size=16)
                button((982,538,1130,582),'左转灯','fpga_test_left' if enabled else None,enabled,danger=True,size=16)
                button((1142,538,1290,582),'右转灯','fpga_test_right' if enabled else None,enabled,danger=True,size=16)
                button((1302,538,1450,582),'熄灭转向灯','fpga_test_off' if enabled else None,enabled,size=16)
                detail=f"序号 {test_status.get('sequence',0)} · {test_status.get('reason','')}"
                text((792,588),fit_text(detail,13,730),13,MUTED)
        pixels=np.asarray(image)
        return pixels if output_rgb else cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)

    def hit(self,x,y):
        return next((key for key,(x1,y1,x2,y2) in self.buttons if x1<=x<=x2 and y1<=y<=y2),None)

    def video_point(self,x,y,shape):
        if self.content_rect is None:return None
        vx,vy,vw,vh=self.content_rect
        if vx<=x<vx+vw and vy<=y<vy+vh and 170<=y<650:
            return ((x-vx)/vw*shape[1],(y-vy)/vh*shape[0])
        return None

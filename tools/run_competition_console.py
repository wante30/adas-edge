"""Native PC competition workbench using the Tesla-BevCam upstream renderer."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from competition_ai.pc_detector import PcDetector
from competition_ai.scenario_policy import decide
from tools.competition_console.controls import Controller
from tools.competition_console.pipeline import FrontPipeline
from tools.competition_console.view import NativeView, SIZE

DEFAULT_VIDEO = Path('D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_113539.mp4')
DEFAULT_MAIN = Path('D:/Datasets/adas_edge_training/rect_models/yolo11n_512x288_split.onnx')
DEFAULT_SPECIALIST = Path('D:/Datasets/adas_edge_training/tl_specialist/tl_specialist_512x192_split.onnx')
WINDOW = 'Tesla-BevCam | Competition PC'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def source_capture(path):
    capture = cv2.VideoCapture(str(path))
    fps,total = float(capture.get(cv2.CAP_PROP_FPS)),int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or fps <= 0 or total <= 0:
        capture.release()
        raise ValueError(f'Cannot decode video: {path}')
    return capture,fps,total


def run(args):
    global WINDOW
    WINDOW = args.window_title or WINDOW
    output = args.output or ROOT/'outputs'/('competition_live_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'trace.jsonl').exists():
        raise ValueError('Choose a fresh output directory; previous evidence is preserved')
    profile = json.loads(args.profile.read_text(encoding='utf-8')) if args.profile else {}
    video = args.video.resolve()
    video_digest = digest(video)
    if profile and profile['video_sha256'] != video_digest:
        raise ValueError('This calibration profile belongs to a different video; use manual B binding')
    main = PcDetector(args.main_model,(512,288))
    specialist = PcDetector(args.specialist_model,(512,192),specialist=True)
    pipeline,controller,view = FrontPipeline(main,specialist),Controller(),NativeView(args.upstream)
    controller.paused = args.paused
    capture,fps,total = source_capture(video)
    frame_id = round(args.start*fps)
    if not 0 <= frame_id < total:
        raise ValueError('Start is outside video')
    capture.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
    end_frame = min(total,round(args.end*fps)) if args.end is not None else total
    if end_frame <= frame_id:
        raise ValueError('End must follow start')
    binding_events = {int(event['frame']):event for event in profile.get('bindings',[])}
    control_events = {int(event['frame']):event['key'] for event in profile.get('scripted_controls',[])} if args.scripted_controls else {}
    snapshot_frames = set(profile.get('snapshot_frames',[]))
    pending_keys,pending_boxes = [],[]
    state = {'frame':None,'drag':None,'message':profile.get('binding_message','')}
    events,seen_events,colour_counts,action_counts,alarm_counts = [],set(),Counter(),Counter(),Counter()
    writer = cv2.VideoWriter(str(output/'native_mp4v.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,SIZE) if args.record else None
    if writer is not None and not writer.isOpened():
        raise RuntimeError('Cannot create recording')
    if not args.headless:
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW,1280,800)
        def mouse(event,x,y,flags,param):
            current = state['frame']
            if current is None:
                return
            point = view.video_point(x,y,current.shape)
            if controller.binding and point is not None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    state['drag'] = point
                elif event == cv2.EVENT_LBUTTONUP and state['drag'] is not None:
                    a,b = state['drag'],point
                    pending_boxes.append([min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])])
                    state['drag'] = None
            elif event == cv2.EVENT_LBUTTONUP:
                key = view.hit(x,y)
                if key:
                    pending_keys.append(key)
        cv2.setMouseCallback(WINDOW,mouse)
    processed,loops,visible = 0,0,False
    current = None
    advance = True
    last_written_frame = None
    display_stamp = None
    started = time.perf_counter()
    try:
        with (output/'trace.jsonl').open('w',encoding='utf-8') as trace:
            while True:
                iteration = time.perf_counter()
                if advance:
                    if frame_id >= end_frame:
                        if args.headless or args.close_at_end:
                            break
                        controller.paused = True
                        advance = False
                        frame_id = end_frame-1
                    else:
                        ok,current = capture.read()
                        if not ok:
                            raise RuntimeError(f'Early video end at frame {frame_id}')
                if current is None:
                    break
                state['frame'] = current
                if frame_id in binding_events and ('bind',frame_id) not in seen_events:
                    event = binding_events[frame_id]
                    pipeline.bind(current,event['bbox'],event['lanes'],source='PROFILE_SHARED_CIRCULAR_SIGNAL_ASSUMPTION')
                    controller.controls = replace(controller.controls,signalized=True)
                    seen_events.add(('bind',frame_id))
                    events.append({'frame':frame_id,'source':'EXPLICIT_SCENE_PROFILE','binding':event})
                if frame_id in control_events and ('key',frame_id) not in seen_events:
                    controller.key(control_events[frame_id])
                    seen_events.add(('key',frame_id))
                    events.append({'frame':frame_id,'source':'SCRIPTED_SIMULATION_BUTTON','key':control_events[frame_id]})
                action = None
                input_changed = bool(pending_keys or pending_boxes)
                for key in pending_keys:
                    action = controller.key(key) or action
                    events.append({'frame':frame_id,'source':'NATIVE_KEYBOARD_OR_MOUSE','key':key})
                pending_keys.clear()
                if action == 'quit':
                    break
                if action in ('restart','open'):
                    if action == 'open':
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk();root.withdraw()
                        chosen = filedialog.askopenfilename(title='打开前视视频',filetypes=[('Video','*.mp4 *.avi *.mkv *.mov')])
                        root.destroy()
                        if not chosen:
                            advance = False
                            continue
                        next_capture,next_fps,next_total = source_capture(Path(chosen))
                        capture.release()
                        capture,fps,total = next_capture,next_fps,next_total
                        video = Path(chosen).resolve()
                        video_digest = digest(video)
                        # Loading another video invalidates the old calibration.
                        binding_events,control_events,snapshot_frames = {},{},set()
                        profile = {}
                        controller = Controller()
                        state['message'] = '新视频：请先按 B 绑定所选车道的灯组'
                    else:
                        loops += 1
                        capture.set(cv2.CAP_PROP_POS_FRAMES,0)
                    frame_id,end_frame,current,advance,last_written_frame = 0,total,None,True,None
                    pipeline.reset();seen_events.clear()
                    continue
                if pending_boxes:
                    for bbox in pending_boxes:
                        try:
                            pipeline.bind(current,bbox,[controller.controls.lane])
                            controller.controls = replace(controller.controls,signalized=True)
                            # One source image is measured once after binding;
                            # green still needs the next distinct frame.
                            signal,_ = pipeline.bindings[controller.controls.lane]
                            signal.update(current,frame_id,round(frame_id/fps*1e9))
                            events.append({'frame':frame_id,'source':'USER_ROI_BINDING','lane':controller.controls.lane.value,'bbox':bbox})
                            state['message'] = '已绑定当前车道的横向红/黄/绿灯组；可逐帧核对。'
                        except ValueError as exc:
                            state['message'] = str(exc)
                    pending_boxes.clear();controller.binding = False
                timestamp_ns = round(frame_id/fps*1e9)
                pipeline.advance(current,frame_id,timestamp_ns)
                snapshot = pipeline.snapshot(controller.controls.lane,timestamp_ns)
                decision = decide(controller.controls,snapshot.as_perception(),timestamp_ns)
                stamp = (frame_id,controller.controls,controller.paused,controller.binding,state['message'],snapshot.signal)
                if stamp != display_stamp:
                    image = view.render(current,snapshot,controller.controls,decision,frame_id=frame_id,fps=fps,
                                        paused=controller.paused,binding=controller.binding,message=state['message'])
                    display_stamp = stamp
                if last_written_frame != frame_id or input_changed or action:
                    row = {'frame':frame_id,'source_s':frame_id/fps,'video':str(video),
                           'controls':asdict(controller.controls),'decision':asdict(decision),
                           'signal':asdict(snapshot.signal),'binding_source':snapshot.binding_source,
                           'main':snapshot.main,'specialist':snapshot.specialist,
                           'main_age_ms':snapshot.main_age_ms,'specialist_age_ms':snapshot.specialist_age_ms}
                    trace.write(json.dumps(row,ensure_ascii=False)+'\n');trace.flush()
                    if last_written_frame != frame_id:
                        processed += 1
                        colour_counts[snapshot.signal.light.value] += 1
                        action_counts[decision.maneuver.value] += 1
                        for alarm in decision.alarms:
                            alarm_counts[alarm] += 1
                        if writer is not None:
                            writer.write(image)
                        if frame_id in snapshot_frames:
                            cv2.imwrite(str(output/f'frame_{frame_id:05d}.png'),image)
                    last_written_frame = frame_id
                if not args.headless:
                    cv2.imshow(WINDOW,image)
                    try:
                        currently_visible = cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE) >= 1
                    except cv2.error:
                        currently_visible = False
                    visible = visible or currently_visible
                    if not currently_visible:
                        break
                    wait = max(1,round(max(0,1/fps-(time.perf_counter()-iteration))*1000))
                    key = cv2.waitKey(wait if not controller.paused else 30)
                    if key >= 0:
                        pending_keys.append(key & 255)
                if args.max_frames and processed >= args.max_frames:
                    break
                advance = (not controller.paused) or action == 'step'
                if advance:
                    frame_id += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    report = {'evidence':'PC_FP32_VIDEO_INFERENCE_WITH_EXPLICIT_SCENE_BINDING_NOT_RK_OR_FPGA',
              'video':str(video),'video_sha256':video_digest,'profile':profile,
              'main_model':str(args.main_model),'main_sha256':digest(args.main_model),
              'specialist_model':str(args.specialist_model),'specialist_sha256':digest(args.specialist_model),
              'source_fps':fps,'processed_frames':processed,'loops':loops,'wall_seconds':time.perf_counter()-started,
              'requested_source_time_hz':{'main':pipeline.main_hz,'specialist':pipeline.specialist_hz},
              'main_inference_updates':len(pipeline.main_timings),'specialist_inference_updates':len(pipeline.specialist_timings),
              'colour_display_frame_counts':dict(colour_counts),'decision_frame_counts':dict(action_counts),
              'alarm_frame_counts':dict(alarm_counts),'events':events,'native_window_was_visible':visible,
              'renderer_source':str(ROOT/'.deps/Tesla-BevCam-Clone/main.py'),
              'limitations':['Lamp direction/ego association is explicitly configured, not inferred automatically',
                             'Bound colour branch is a calibrated horizontal R/Y/G pixel classifier, separate from both detection models',
                             'No lane model, hazard-zone detector or navigation reader in this first video integration',
                             'No real vehicle motion, calibrated 3D distances, PCIe output or board benchmark']}
    (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output':str(output.resolve()),'frames':processed,'colours':dict(colour_counts),
                      'actions':dict(action_counts),'alarms':dict(alarm_counts),'window_visible':visible},ensure_ascii=False),flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--video',type=Path,default=DEFAULT_VIDEO)
    result.add_argument('--main-model',type=Path,default=DEFAULT_MAIN)
    result.add_argument('--specialist-model',type=Path,default=DEFAULT_SPECIALIST)
    result.add_argument('--profile',type=Path)
    result.add_argument('--upstream',type=Path)
    result.add_argument('--window-title')
    result.add_argument('--output',type=Path)
    result.add_argument('--start',type=float,default=0.)
    result.add_argument('--end',type=float)
    result.add_argument('--max-frames',type=int,default=0)
    result.add_argument('--paused',action='store_true')
    result.add_argument('--headless',action='store_true')
    result.add_argument('--record',action='store_true')
    result.add_argument('--scripted-controls',action='store_true')
    result.add_argument('--close-at-end',action='store_true')
    return result


if __name__ == '__main__':
    run(parser().parse_args())

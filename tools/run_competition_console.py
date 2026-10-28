"""Native competition cockpit with real-video inference and explicit simulation."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
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
from competition_ai.scenario import Lane, Mode
from competition_ai.scenario_policy import decide
from competition_io.scenario_hud import MockScenarioHud, project_scenario
from tools.competition_console.controls import Controller
from tools.competition_console.pipeline import FrontPipeline
from tools.competition_console.navigation_session import NavigationSession, apply_navigation
from tools.competition_console.view import NativeView, SIZE
from tools.competition_console.runtime import BoundedVideoWriter, LatestFrameWorker, PlaybackPacer, PrefetchedCapture
from tools.competition_console.pgv_capture import PgvLiveCapture
from tools.competition_console.navigation_reader import NavigationReader
from tools.competition_console.gtk_display import choose_display_backend
from tools.competition_console.fpga_perception import build_hud_perception
from tools.competition_console.hud_sender import Bar1MmapBus, HudV2Sender
from tools.competition_console.hud_diagnostics import HudDiagnostics, LegacyHudMmapBus

DEFAULT_VIDEO = Path('D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_113539.mp4')
DEFAULT_MAIN = Path('D:/Datasets/adas_edge_training/rect_models/yolo11n_512x288_split.onnx')
DEFAULT_SPECIALIST = Path('D:/Datasets/adas_edge_training/tl_specialist/tl_specialist_512x192_split.onnx')
WINDOW = 'Tesla-BevCam | Competition PC'


def digest(path):
    with Path(path).open('rb') as handle:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
        return checksum.hexdigest()


def _summarize_ms(values):
    if not values:
        return {"samples": 0, "mean": None, "median": None, "p95": None, "max": None}
    samples = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(samples),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
        "max": float(np.max(samples)),
    }


def source_capture(path):
    capture = cv2.VideoCapture(str(path))
    fps,total = float(capture.get(cv2.CAP_PROP_FPS)),int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or fps <= 0 or total <= 0:
        capture.release()
        raise ValueError(f'Cannot decode video: {path}')
    return PrefetchedCapture(capture),fps,total


def pgv_capture(path,timeout_s=5.,quadrant=None,navigation_quadrant=None):
    return PgvLiveCapture(path,timeout_s=timeout_s,quadrant=quadrant,
                          navigation_quadrant=navigation_quadrant),0.,None


def validate_hardware_hud_args(args):
    if args.fpga_hud and (
            args.pgv is None or args.profile is not None or args.scripted_controls
            or args.mock_hud or args.mode is not None or args.split_navigation):
        raise ValueError('FPGA HUD requires live PGV perception without simulation controls')
    if args.pgv is not None and (args.profile is not None or args.scripted_controls):
        raise ValueError('Live PGV does not accept manual profiles or scripted control inputs')


def dispatch_key(controller,key,*,hardware_live=False):
    if not hardware_live:
        return controller.key(key)
    if isinstance(key,int):
        if key == 27:
            return 'quit'
        if not 0 <= key < 128:
            return None
        key = chr(key)
    key = key.lower()
    if key in ('v','f','c','b'):
        return controller.key(key)
    return None


def dispatch_diagnostic(tester,key):
    if not isinstance(key,str) or not key.startswith('fpga_test_'):
        return False
    if tester is not None:
        command = key.removeprefix('fpga_test_')
        if command == 'enable':
            tester.set_enabled(not tester.enabled)
        elif command in ('left','right','off'):
            tester.request(command)
    return True


def bind_signal_roi(pipeline,frame,bbox,lane,frame_id,timestamp_ns):
    pipeline.bind(frame,bbox,[lane])
    signal,_ = pipeline.bindings[Lane(lane)]
    signal.update(frame,frame_id,timestamp_ns)


def _receipt_dict(receipt):
    return asdict(receipt) if hasattr(receipt,'__dataclass_fields__') else dict(vars(receipt))


def send_fpga_perception(sender,frame,snapshot):
    perception = build_hud_perception(frame,snapshot)
    pending = getattr(sender,'pending',None)
    if pending is not None:
        receipt = sender.query_pending(timeout_ms=0.0)
        return {'attempted':False,**_receipt_dict(receipt)}
    receipt = sender.commit(perception.ext_request(),time_enable=False,timeout_ms=0.0)
    return {'attempted':True,**_receipt_dict(receipt)}


def run(args):
    with ExitStack() as resources:
        return _run(args,resources)


def _run(args,resources):
    global WINDOW
    validate_hardware_hud_args(args)
    setup_started = time.perf_counter()
    WINDOW = args.window_title or WINDOW
    output = args.output or ROOT/'outputs'/('competition_pgv_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') if args.pgv else 'competition_live_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'trace.jsonl').exists():
        raise ValueError('Choose a fresh output directory; previous evidence is preserved')
    profile = json.loads(args.profile.read_text(encoding='utf-8')) if args.profile else {}
    live = bool(args.pgv)
    video = args.pgv.resolve() if live else args.video.resolve()
    video_digest = None if live else digest(video)
    if not live and profile and profile['video_sha256'] != video_digest:
        raise ValueError('This calibration profile belongs to a different video; use manual B binding')
    splitter, navigation_session = None,None
    if args.split_navigation and args.pgv:
        raise ValueError('--split-navigation is for calibrated file video; live PGV uses fixed quadrants')
    if args.split_navigation:
        from tools.competition_console.split_video import SplitVideo
        splitter = SplitVideo(profile)
        navigation_session = NavigationSession(NavigationReader())
    elif args.pgv is not None and args.pgv_navigation_quadrant is not None:
        navigation_session = NavigationSession(NavigationReader(),image_size=(960,540))
    detector_backend = getattr(args,'detector_backend','onnx')
    runtime_status = 'RK3566 RKNN INT8' if detector_backend == 'rknn' else 'PC ONNX FP32'
    if detector_backend == 'rknn':
        from tools.competition_console.rknn_detector import RknnConsoleDetector
        detector_type = RknnConsoleDetector
    else:
        detector_type = PcDetector
    main = detector_type(args.main_model,(512,288))
    specialist = detector_type(args.specialist_model,(512,192),specialist=True)
    for detector in (main,specialist):
        close = getattr(detector,'close',None)
        if close is not None:
            resources.callback(close)
    pipeline = FrontPipeline(main,specialist,hazards=args.hazards,hazard_path=profile.get('hazard_path'))
    controller,view = Controller(),NativeView(args.upstream)
    tester = HudDiagnostics(bus_factory=lambda: LegacyHudMmapBus(args.bar1_resource)) if live else None
    if tester is not None:
        resources.callback(tester.close)
    hud = MockScenarioHud() if args.mock_hud else None
    fpga_hud,fpga_hud_state = None,{'requested':bool(args.fpga_hud),'available':False,
                                   'phase':'DISABLED','reason':None,'attempted_transactions':0,
                                   'applied_transactions':0,'last_sequence':None,'last_commit':None}
    if args.fpga_hud:
        fpga_bus = None
        try:
            fpga_bus = Bar1MmapBus(args.bar1_resource)
            fpga_hud = HudV2Sender(fpga_bus)
            fpga_hud.probe()
            fpga_hud_state.update(available=True,phase='READY')
            resources.callback(fpga_hud.close)
        except (OSError,RuntimeError,ValueError) as exc:
            if fpga_hud is not None:
                fpga_hud.close()
            elif fpga_bus is not None:
                fpga_bus.close()
            fpga_hud = None
            fpga_hud_state.update(phase='UNAVAILABLE',reason=str(exc))
    hud_receipt,hud_checked = None,0
    if args.mode:
        controller.controls = replace(controller.controls,mode=Mode(args.mode))
    elif args.split_navigation:
        controller.controls = replace(controller.controls,mode=Mode.NAVIGATION)
    elif args.hazards:
        controller.controls = replace(controller.controls,mode=Mode.HAZARD)
    controller.paused = args.paused
    if live and args.record:
        raise ValueError('Recording a live PGV source needs an explicit frame rate; use video replay for recording')
    if live:
        capture,fps,total = pgv_capture(video,args.pgv_timeout,args.pgv_quadrant,
                                        args.pgv_navigation_quadrant)
        fps = 0.0
    else:
        capture,fps,total = source_capture(video)
    resources.callback(capture.release)
    if live:
        frame_id = 0
        end_frame = 10**12
        if args.paused:
            raise ValueError('Live PGV cannot start paused; use display-only controls while streaming')
    else:
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
    state = {'frame':None,'drag':None,'drag_box':None,'message':profile.get('binding_message',''),'ended':False}
    events,seen_events,colour_counts,action_counts,alarm_counts = [],set(),Counter(),Counter(),Counter()
    test_status = tester.status if tester is not None else None
    writer = BoundedVideoWriter(str(output/'native_mp4v.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,SIZE) if args.record else None
    if writer is not None:
        resources.callback(writer.release)
    if writer is not None and not writer.isOpened():
        raise RuntimeError('Cannot create recording')
    requested_display = getattr(args,'display_backend','auto')
    gui_backend = next((line.split(':',1)[1].strip() for line in cv2.getBuildInformation().splitlines()
                        if line.strip().startswith('GUI:')),'NONE')
    display_backend = choose_display_backend(requested_display,gui_backend=gui_backend)
    display = None
    if not args.headless:
        def mouse(event,x,y,flags,param):
            current = state['frame']
            if current is None:
                return
            key = view.hit(x,y)
            if key and event == cv2.EVENT_LBUTTONUP:
                pending_keys.append(key)
                state['drag'] = None
                state['drag_box'] = None
                return
            point = None if controller.diagnostics else view.video_point(x,y,current.shape)
            if controller.binding and point is not None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    state['drag'] = point
                elif event == cv2.EVENT_MOUSEMOVE and state['drag'] is not None:
                    a,b = state['drag'],point
                    state['drag_box'] = (min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1]))
                elif event == cv2.EVENT_LBUTTONUP and state['drag'] is not None:
                    a,b = state['drag'],point
                    pending_boxes.append([min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])])
                    state['drag'] = None
                    state['drag_box'] = None
        if display_backend == 'gtk':
            from tools.competition_console.gtk_display import GtkDisplay
            display_size = getattr(args,'display_size',SIZE)
            display = GtkDisplay(WINDOW,SIZE,display_size)
            display.set_mouse_callback(mouse)
            resources.callback(display.close)
        else:
            resources.callback(cv2.destroyAllWindows)
            cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW,1440,810)
            cv2.setMouseCallback(WINDOW,mouse)
    processed,loops,visible = 0,0,False
    current, navigation_image, split_metadata = None,None,None
    source_start_timestamp_ns = None
    advance = True
    last_written_frame = None
    last_emitted_ended = False
    display_stamp = None
    image_is_rgb = False
    capture_read_timings, pipeline_timings = [], []
    display_present_timings, display_wait_timings, loop_timings = [], [], []
    render_timings = []
    pipeline_worker = None
    cached_snapshot = None
    last_pipeline_submitted_frame = None
    pipeline_submitted_frames = 0
    pipeline_skipped_frames = 0
    if live and not args.headless:
        def advance_pipeline_async(frame, source_frame_id, source_timestamp_ns):
            pipeline_started = time.perf_counter()
            pipeline.advance(frame,source_frame_id,source_timestamp_ns)
            return (time.perf_counter()-pipeline_started)*1000
        pipeline_worker = LatestFrameWorker(advance_pipeline_async)
        resources.callback(pipeline_worker.close)

    def finish_pipeline(*, wait=False):
        if pipeline_worker is None:
            return False
        elapsed = pipeline_worker.wait() if wait else pipeline_worker.poll()
        if elapsed is None:
            return False
        pipeline_timings.append(elapsed)
        return True

    pacer = None if live else PlaybackPacer(fps)
    started = time.perf_counter()
    setup_seconds = started-setup_started
    try:
        with (output/'trace.jsonl').open('w',encoding='utf-8') as trace:
            while True:
                iteration = time.perf_counter()
                if advance:
                    if frame_id >= end_frame:
                        if live:
                            raise RuntimeError('Invalid live end frame')
                        if args.headless or args.close_at_end:
                            break
                        controller.paused = True
                        state['ended'] = True
                        advance = False
                        frame_id = end_frame-1
                    else:
                        read_started = time.perf_counter()
                        ok,source_frame = capture.read()
                        if live:
                            if not ok:
                                detail = f': {capture.error}' if capture.error is not None else ''
                                raise RuntimeError(f'PCIe frame timeout{detail}')
                            frame_id = int(capture.frame_id)
                            timestamp_ns = int(capture.timestamp_ns)
                            if source_start_timestamp_ns is None:
                                source_start_timestamp_ns = timestamp_ns
                        else:
                            if not ok:
                                raise RuntimeError(f'Early video end at frame {frame_id}')
                        capture_read_timings.append((time.perf_counter()-read_started)*1000)
                        if live:
                            current = source_frame
                            navigation_image = getattr(capture,'navigation_image',None)
                            if navigation_image is not None:
                                split_metadata = {
                                    'source_frame_id':frame_id,
                                    'road_source_frame_id':frame_id,
                                    'navigation_source_frame_id':frame_id,
                                    'road_quadrant':args.pgv_quadrant,
                                    'navigation_quadrant':args.pgv_navigation_quadrant,
                                    'source_timestamp_ns':timestamp_ns,
                                    'geometry_source':'PGV_FIXED_QUADRANT_ROUTE',
                                }
                        elif splitter is not None:
                            split = splitter.split(source_frame,frame_id)
                            current,navigation_image,split_metadata = split.road,split.navigation,split.metadata
                        else:
                            current,navigation_image,split_metadata = source_frame,None,None
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
                    if not dispatch_diagnostic(tester,key):
                        action = dispatch_key(controller,key,hardware_live=live) or action
                    events.append({'frame':frame_id,'source':'NATIVE_KEYBOARD_OR_MOUSE','key':key})
                pending_keys.clear()
                test_status = tester.poll() if tester is not None else None
                if not controller.binding:
                    state['drag'],state['drag_box'] = None,None
                if action == 'quit':
                    break
                if action in ('restart','open'):
                    if pipeline_worker is not None:
                        finish_pipeline(wait=True)
                        cached_snapshot = None
                        last_pipeline_submitted_frame = None
                    pending_boxes.clear()
                    state['drag'],state['drag_box'] = None,None
                    if action == 'open':
                        if writer is not None:
                            state['message'] = '正在录制：请结束本次录制后再打开另一段视频。'
                            advance = False
                            continue
                        if display is not None:
                            chosen = display.choose_video()
                        else:
                            import tkinter as tk
                            from tkinter import filedialog
                            root = tk.Tk();root.withdraw()
                            chosen = filedialog.askopenfilename(title='打开前视视频',filetypes=[('Video','*.mp4 *.avi *.mkv *.mov')])
                            root.destroy()
                        if not chosen:
                            advance = False
                            continue
                        try:
                            next_capture,next_fps,next_total = source_capture(Path(chosen))
                            resources.callback(next_capture.release)
                        except ValueError:
                            state['message'] = '视频无法解码，请重新选择可播放的视频文件。'
                            advance = False
                            continue
                        capture.release()
                        capture,fps,total = next_capture,next_fps,next_total
                        video = Path(chosen).resolve()
                        video_digest = digest(video)
                        live = False
                        pipeline_worker = None
                        pacer.reset(fps)
                        # Loading another video invalidates the old calibration.
                        binding_events,control_events,snapshot_frames = {},{},set()
                        profile = {}
                        # Screen geometry is tied to the calibrated source.
                        splitter,navigation_session,navigation_image,split_metadata = None,None,None,None
                        controller = Controller()
                        if args.mode:
                            controller.controls = replace(controller.controls,mode=Mode(args.mode))
                        elif args.hazards:
                            controller.controls = replace(controller.controls,mode=Mode.HAZARD)
                        # A path calibrated to another video cannot be reused.
                        if pipeline.hazard_engine is not None:
                            from competition_ai.hazard_events import HazardEvents
                            pipeline.hazard_engine = HazardEvents()
                        state['message'] = '新视频：请先按 B 绑定所选车道的灯组'
                    else:
                        loops += 1
                        capture.set(cv2.CAP_PROP_POS_FRAMES,0)
                        if args.mode:
                            controller.controls = replace(controller.controls,mode=Mode(args.mode))
                        elif splitter is not None:
                            controller.controls = replace(controller.controls,mode=Mode.NAVIGATION)
                        elif args.hazards:
                            controller.controls = replace(controller.controls,mode=Mode.HAZARD)
                    frame_id,end_frame,current,advance,last_written_frame = 0,total,None,True,None
                    state['ended'] = False
                    last_emitted_ended = False
                    display_stamp = None
                    pacer.reset(fps)
                    pipeline.reset();seen_events.clear()
                    cached_snapshot = None
                    if navigation_session is not None:
                        navigation_session.reset()
                    continue
                if pending_boxes:
                    if pipeline_worker is not None and pipeline_worker.busy:
                        finish_pipeline(wait=True)
                    for bbox in pending_boxes:
                        try:
                            bind_signal_roi(pipeline,current,bbox,controller.controls.lane,frame_id,timestamp_ns)
                            controller.controls = replace(controller.controls,signalized=True)
                            # One source image is measured once after binding;
                            # green still needs the next distinct frame.
                            events.append({'frame':frame_id,'source':'USER_ROI_BINDING','lane':controller.controls.lane.value,'bbox':bbox})
                            state['message'] = '已绑定当前车道的横向红/黄/绿灯组；可逐帧核对。'
                        except ValueError as exc:
                            state['message'] = str(exc)
                    pending_boxes.clear();controller.binding = False
                if not live:
                    timestamp_ns = round(frame_id/fps*1e9)
                if pipeline_worker is None:
                    pipeline_started = time.perf_counter()
                    pipeline.advance(current,frame_id,timestamp_ns)
                    pipeline_timings.append((time.perf_counter()-pipeline_started)*1000)
                    snapshot = pipeline.snapshot(controller.controls.lane,timestamp_ns)
                else:
                    finish_pipeline()
                    if pipeline_worker.busy:
                        snapshot = cached_snapshot.at(timestamp_ns)
                        if frame_id != last_pipeline_submitted_frame:
                            pipeline_skipped_frames += 1
                    else:
                        snapshot = pipeline.snapshot(controller.controls.lane,timestamp_ns)
                        if frame_id != last_pipeline_submitted_frame:
                            if pipeline_worker.submit(current.copy(),frame_id,timestamp_ns):
                                last_pipeline_submitted_frame = frame_id
                                pipeline_submitted_frames += 1
                    cached_snapshot = snapshot
                fpga_perception = build_hud_perception(current,snapshot).to_dict() if live else None
                perception = snapshot.as_perception()
                navigation = None
                if navigation_session is not None:
                    navigation_session.advance(navigation_image,frame_id,timestamp_ns)
                    navigation = navigation_session.current(timestamp_ns,ended=state['ended'])
                    controller.controls = apply_navigation(controller.controls,navigation)
                decision = decide(controller.controls,perception,timestamp_ns)
                should_emit = last_written_frame != frame_id or input_changed or action or state['ended'] != last_emitted_ended
                fpga_receipt = None
                if fpga_hud is not None and should_emit:
                    try:
                        fpga_receipt = send_fpga_perception(fpga_hud,current,snapshot)
                        fpga_hud_state['last_commit'] = fpga_receipt
                        fpga_hud_state['last_sequence'] = fpga_receipt.get('sequence')
                        fpga_hud_state['attempted_transactions'] += int(fpga_receipt.get('attempted',False))
                        fpga_hud_state['applied_transactions'] += int(fpga_receipt.get('committed',False))
                        fpga_hud_state['phase'] = fpga_receipt.get('phase','UNKNOWN')
                        fpga_hud_state['reason'] = fpga_receipt.get('reason')
                    except (OSError,RuntimeError,TimeoutError,ValueError) as exc:
                        fpga_hud_state.update(phase='ERROR',reason=str(exc))
                if hud is not None and should_emit:
                    # EOF is an inactive source. Explicitly project an expired
                    # envelope to clear direction in the software receiver.
                    projection = project_scenario(controller.controls,perception,decision,
                                                  now_ns=timestamp_ns+(750_000_000 if state['ended'] else 0),
                                                  source_frame_id=frame_id,source_timestamp_ns=timestamp_ns)
                    hud_receipt = hud.submit(projection)
                    if not hud_receipt['receiver_verified']:
                        raise RuntimeError('Software HUD receiver does not match the encoded decision')
                    hud_checked += 1
                stamp = (frame_id,controller.controls,controller.paused,controller.binding,state['message'],snapshot.signal,
                         controller.overlay,controller.diagnostics,controller.bev,state['ended'],state['drag_box'],
                         tuple(test_status.items()) if test_status else None)
                if live:
                    transport_status = (f'PCIe 帧 {frame_id}  龄期 {getattr(capture,"age_ms",0) or 0:.0f} ms'
                                        f'  丢帧 {capture.dropped_frames}')
                    if args.fpga_hud:
                        transport_status += f'  FPGA {fpga_hud_state["phase"]}'
                        if fpga_hud_state['reason']:
                            transport_status += ': ' + fpga_hud_state['reason'][:48]
                elif hud_receipt is not None:
                    transport_status = 'HUD 图标仿真通过'
                else:
                    transport_status = 'PCIe 未连接'
                if stamp != display_stamp:
                    render_start = time.perf_counter()
                    image_is_rgb = (display is not None and display_backend == 'gtk'
                                    and writer is None and frame_id not in snapshot_frames)
                    image = view.render(current,snapshot,controller.controls,decision,frame_id=frame_id,fps=fps,
                                        paused=controller.paused,binding=controller.binding,message=state['message'],
                                        overlay=controller.overlay,diagnostics=controller.diagnostics,bev=controller.bev,
                                        total_frames=total,source_name=str(video),ended=state['ended'],
                                        drag_box=state['drag_box'],
                                        navigation_image=navigation_image,navigation=navigation,
                                        runtime_status=runtime_status,
                                        transport_status=transport_status,
                                        live=live,
                                        fpga_perception=fpga_perception,test_status=test_status,
                                        source_timestamp_ns=timestamp_ns,
                                        source_start_timestamp_ns=source_start_timestamp_ns if live else 0,
                                        output_rgb=image_is_rgb)
                    render_timings.append((time.perf_counter()-render_start)*1000)
                    display_stamp = stamp
                if should_emit:
                    row = {'frame':frame_id,'source_s':frame_id/fps if fps > 0 else None,
                           'source_timestamp_ns':timestamp_ns,
                           'source_elapsed_s':max(0,timestamp_ns-source_start_timestamp_ns)/1e9
                           if live and source_start_timestamp_ns is not None else None,
                           'video':str(video),
                           'view':{'overlay':controller.overlay,'diagnostics':controller.diagnostics,
                                   'bev':controller.bev,'paused':controller.paused,'ended':state['ended']},
                           'controls':asdict(controller.controls),'decision':asdict(decision),
                           'signal':asdict(snapshot.signal),'binding_source':snapshot.binding_source,
                           'hazards':asdict(snapshot.hazards),
                           'main':snapshot.main,'specialist':snapshot.specialist,
                           'main_age_ms':snapshot.main_age_ms,'specialist_age_ms':snapshot.specialist_age_ms}
                    if navigation is not None:
                        row['navigation'] = navigation
                        row['same_frame_split'] = split_metadata
                        row['road_inference_frame_id'] = round(frame_id-(snapshot.main_age_ms or 0)*fps/1000)
                    if hud_receipt is not None:
                        row['hud_software_receipt'] = hud_receipt
                    if fpga_receipt is not None:
                        row['fpga_hud_receipt'] = fpga_receipt
                    if live:
                        row['fpga_perception'] = fpga_perception
                        row['fpga_test'] = test_status
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
                    last_emitted_ended = state['ended']
                if not args.headless:
                    present_started = time.perf_counter()
                    if display is not None:
                        display.present(image,input_rgb=image_is_rgb)
                        currently_visible = display.is_visible()
                    else:
                        cv2.imshow(WINDOW,image)
                        try:
                            currently_visible = cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE) >= 1
                        except cv2.error:
                            currently_visible = False
                    display_present_timings.append((time.perf_counter()-present_started)*1000)
                    visible = visible or currently_visible
                    if not currently_visible:
                        break
                    wait = 1 if live else pacer.delay_ms(frame_id,paused=controller.paused,unpaced=args.unpaced)
                    wait_started = time.perf_counter()
                    key = display.wait_key(wait) if display is not None else cv2.waitKey(wait)
                    display_wait_timings.append((time.perf_counter()-wait_started)*1000)
                    if key >= 0:
                        pending_keys.append(key & 255)
                if args.max_frames and processed >= args.max_frames:
                    break
                advance = (not controller.paused) or action == 'step'
                if advance:
                    frame_id += 1
                loop_timings.append((time.perf_counter()-iteration)*1000)
    finally:
        if pipeline_worker is not None and pipeline_worker.busy:
            finish_pipeline(wait=True)
        resources.close()
    providers = getattr(getattr(main,'session',None),'get_providers',None)
    inference_runtime = 'RKNN_LITE_INT8' if detector_backend == 'rknn' else 'ONNX_RUNTIME_CPU_FP32_NOT_RKNN'
    wall_seconds = time.perf_counter()-started
    report = {'evidence':('RK3566_RKNN_LITE_INT8_WITH_EXPLICIT_SCENE_BINDING_NOT_FPGA' if detector_backend == 'rknn'
                          else 'PC_FP32_VIDEO_INFERENCE_WITH_EXPLICIT_SCENE_BINDING_NOT_RK_OR_FPGA'),
                  'video':str(video),'video_sha256':video_digest,'profile':profile,
              'main_model':str(args.main_model),'main_sha256':digest(args.main_model),
              'specialist_model':str(args.specialist_model),'specialist_sha256':digest(args.specialist_model),
              'source_fps':None if live else fps,'source_clock':'PGV_MONOTONIC_NS' if live else 'VIDEO_FRAME_INDEX',
              'processed_frames':processed,'loops':loops,'wall_seconds':wall_seconds,
              'effective_fps':processed/wall_seconds if wall_seconds > 0 else 0.,
              'pipeline_frame_submissions':pipeline_submitted_frames,
              'pipeline_frames_skipped_while_busy':pipeline_skipped_frames,
              'setup_seconds':setup_seconds,
              'execution':{'headless':args.headless,'unpaced':args.unpaced,'record':args.record,
                           'opencv_threads':cv2.getNumThreads(),
                           'prefetch_capacity':2,'frame_skipping':pipeline_skipped_frames>0,
                           'pipeline_inference_async':pipeline_worker is not None,
                           'pipeline_timing_overlaps_main_loop':pipeline_worker is not None,
                           'display_backend':display_backend,
                           'display_size':getattr(args,'display_size',None),
                           'providers':providers() if providers is not None else None,
                           'inference_runtime':inference_runtime},
              'requested_source_time_hz':{'main':pipeline.main_hz,'specialist':pipeline.specialist_hz},
              'main_inference_updates':len(pipeline.main_timings),'specialist_inference_updates':len(pipeline.specialist_timings),
              'colour_display_frame_counts':dict(colour_counts),'decision_frame_counts':dict(action_counts),
              'alarm_frame_counts':dict(alarm_counts),'events':events,'native_window_was_visible':visible,
              'renderer_source':str(ROOT/'.deps/Tesla-BevCam-Clone/main.py'),
              'native_display_size':SIZE,'render_ms':{'median':float(np.median(render_timings)),
                  'p95':float(np.percentile(render_timings,95))} if render_timings else None,
              'stage_ms':{
                  'capture_read':_summarize_ms(capture_read_timings),
                  'pipeline_advance':_summarize_ms(pipeline_timings),
                  'main_rknn_inference':_summarize_ms(pipeline.main_timings),
                  'specialist_rknn_inference':_summarize_ms(pipeline.specialist_timings),
                  'render':_summarize_ms(render_timings),
                  'display_present':_summarize_ms(display_present_timings),
                  'display_wait':_summarize_ms(display_wait_timings),
                  'loop_work':_summarize_ms(loop_timings),
              },
              'pcie_source':{'enabled':args.pgv is not None,'path':str(video) if args.pgv is not None else None,
                             'road_quadrant':args.pgv_quadrant if args.pgv is not None else None,
                             'navigation_quadrant':args.pgv_navigation_quadrant if args.pgv is not None else None,
                             'latest_frame_id':getattr(capture,'frame_id',None) if args.pgv is not None else None,
                             'latest_age_ms':getattr(capture,'age_ms',None) if args.pgv is not None else None,
                             'dropped_frames':capture.dropped_frames if args.pgv is not None else None},
              'playback_lateness':pacer.summary() if pacer is not None else None,
              'recording':writer.summary() if writer is not None else None,
              'hud_software_checked_transactions':hud_checked,'fpga_hud':fpga_hud_state,
              'fpga_test':tester.status if tester is not None else None,
              'same_frame_navigation_enabled':navigation_session is not None,
              'navigation_hz':navigation_session.hz if navigation_session is not None else None,
              'navigation_inference_updates':len(navigation_session.timings) if navigation_session is not None else 0,
              'navigation_ms':{'median':float(np.median(navigation_session.timings)),
                               'p95':float(np.percentile(navigation_session.timings,95))}
                               if navigation_session is not None and navigation_session.timings else None,
              'hud_limit':'HUD1 carries icons only; complete maneuver, light, lanes and TTL remain in JSON sidecar',
              'hazards_enabled':args.hazards,
              'hazard_algorithm':'SOURCE_TIMED_IMAGE_TRAJECTORY_EXPERIMENTAL' if args.hazards else 'DISABLED',
              'hazard_postprocess_ms':{'updates':len(pipeline.hazard_engine.timings),
                                      'median':float(np.median(pipeline.hazard_engine.timings)),
                                      'p95':float(np.percentile(pipeline.hazard_engine.timings,95))}
                                      if pipeline.hazard_engine and pipeline.hazard_engine.timings else None,
              'limitations':['HUD V2 only receives directly observed traffic light/pedestrian fields; unavailable fields stay clear',
                             'Lamp direction/ego association is explicitly configured, not inferred automatically',
                             'Bound colour branch is a calibrated horizontal R/Y/G pixel classifier, separate from both detection models',
                             'Path polygon is an image-space assumption, not detected lanes or a calibrated vehicle trajectory',
                             'Sudden braking and pedestrian red-signal violations remain unconfirmed without their missing evidence',
                             'No lane model; navigation uses calibrated screen pixels with expiry' if splitter is not None else 'No lane model or navigation reader in this video integration',
                             'No real vehicle motion, calibrated 3D distances, PCIe output or full RK/FPGA competition benchmark']}
    (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output':str(output.resolve()),'frames':processed,'colours':dict(colour_counts),
                      'actions':dict(action_counts),'alarms':dict(alarm_counts),'window_visible':visible},ensure_ascii=False),flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--video',type=Path,default=DEFAULT_VIDEO)
    result.add_argument('--pgv',type=Path,help='Read newest frames from board-side PGV1 publisher instead of replaying a video')
    result.add_argument('--main-model',type=Path,default=DEFAULT_MAIN)
    result.add_argument('--specialist-model',type=Path,default=DEFAULT_SPECIALIST)
    result.add_argument('--profile',type=Path)
    result.add_argument('--split-navigation',action='store_true',help='Split calibrated road and navigation screen from the same source frame; recognize navigation pixels')
    result.add_argument('--upstream',type=Path)
    result.add_argument('--window-title')
    result.add_argument('--output',type=Path)
    result.add_argument('--start',type=float,default=0.)
    result.add_argument('--end',type=float)
    result.add_argument('--max-frames',type=int,default=0)
    result.add_argument('--paused',action='store_true')
    result.add_argument('--headless',action='store_true')
    result.add_argument('--detector-backend',choices=('onnx','rknn'),default='onnx')
    result.add_argument('--display-backend',choices=('auto','opencv','gtk'),default='auto')
    result.add_argument('--display-size',type=lambda value: tuple(map(int,value.lower().replace('×','x').split('x'))),default=SIZE,
                        help='GTK fullscreen logical size, e.g. 1024x576')
    result.add_argument('--unpaced',action='store_true',help='Measure visible-window throughput without playback waiting; no frame/inference skipping')
    result.add_argument('--pgv-timeout',type=float,default=5.,help='Maximum seconds to wait for each new PGV frame')
    result.add_argument('--pgv-quadrant',type=int,choices=range(4),default=3,
                        help='PGV quadrant for road detection: 0 TL, 1 TR, 2 BL, 3 BR')
    result.add_argument('--pgv-navigation-quadrant',type=int,choices=range(4),default=1,
                        help='PGV quadrant for navigation OCR: 0 TL, 1 TR, 2 BL, 3 BR')
    result.add_argument('--fpga-hud',action='store_true',help='Send only observed HUD V2 perception fields over BAR1')
    result.add_argument('--bar1-resource',type=Path,default=Path('/sys/bus/pci/devices/0000:01:00.0/resource1'))
    result.add_argument('--hazards',action='store_true',help='Enable experimental image-trajectory danger events')
    result.add_argument('--mock-hud',action='store_true',help='Verify HUD icon projection using an in-memory receiver; no hardware writes')
    result.add_argument('--mode',choices=[mode.value for mode in Mode],help='Initial displayed competition workflow')
    result.add_argument('--record',action='store_true')
    result.add_argument('--scripted-controls',action='store_true')
    result.add_argument('--close-at-end',action='store_true')
    return result


if __name__ == '__main__':
    run(parser().parse_args())

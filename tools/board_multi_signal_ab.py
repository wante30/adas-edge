"""RK3566 same-video A/B in an isolated directory with no live-process control.

Run only on the connected LubanCat. Uses private UDP ports and DRY_RUN. Never
edits services, autostart, control registers, or formal deployment files.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import cv2
import numpy as np


def cpu_sample():
    values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3]+values[4]


def require_idle_bench(proc_root=Path('/proc')):
    competing = []
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
            if ((argv and argv[0].endswith(b'/qt_gpu_probe'))
                    or any(part.endswith((b'/tools/isolated_ai.py', b'/ai/b05p_ai_bridge_launcher.py')) for part in argv)):
                competing.append(int(path.parent.name))
        except OSError:
            continue
    if competing:
        raise RuntimeError('Competing HMI/AI processes: '+str(competing)
                           +'. Prepare an idle bench through the normal lifecycle workflow; this tool never pauses live services.')


def stop_child(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(8)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(3)


def make_replay(video, target):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot open real video')
    count = 0
    with target.open('xb') as output:
        while count < 106:
            ok, frame = cap.read()
            if not ok:
                break
            # Real Q4 pixels. Other cameras are blank in this performance-only
            # fixture; the original real video is retained as image evidence.
            quad = np.zeros((720, 1280, 3), np.uint8)
            quad[360:, 640:] = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            b, g, r = cv2.split(quad.astype(np.uint16))
            packed = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
            output.write(packed.astype('<u2').tobytes())
            count += 1
    cap.release()
    if count == 0:
        raise ValueError('No real video frames')
    return count


def switch_model(ai_path, model):
    sys.path.insert(0, str(ai_path))
    from model_profiles import encode_model_switch_request, decode_model_switch_reply
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(.3)
        request = encode_model_switch_request(12345, model)
        deadline = time.monotonic()+30
        last_sent = 0.
        while time.monotonic() < deadline:
            if time.monotonic()-last_sent >= 1:
                sock.sendto(request, ('127.0.0.1', 38360))
                last_sent = time.monotonic()
            try:
                reply = decode_model_switch_reply(sock.recv(4096))
                if reply and reply.get('state') == 'APPLIED':
                    return reply
                if reply and reply.get('state') in ('FAILED', 'REJECTED'):
                    raise RuntimeError(str(reply))
            except socket.timeout:
                pass
        raise RuntimeError('Model switch timeout')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--model', default='FAST-Q4')
    p.add_argument('--label', default='ab')
    p.add_argument('--rate', type=float, default=2.)
    p.add_argument('--debug-smoke', action='store_true', help='One enabled run with Qt debug screenshot; no A/B claim')
    p.add_argument('--start-view', type=int, choices=(0, 1, 2), default=2)
    args = p.parse_args()
    # SIGSTOP/SIGCONT is unsafe for shell-supervised Qt: wait() can report a
    # stopped child as completed, causing duplicate launches after resume.
    # Require an idle test bench and leave service lifecycle to its maintainer.
    require_idle_bench()
    root = args.root.resolve()
    run = root/(args.label+'_'+str(time.monotonic_ns())); run.mkdir()
    raw = Path('/dev/shm')/(run.name+'.rgb')
    raw_frames = make_replay(args.video, raw)
    qml = root/'hmi/main.qml'
    replay_qml = qml.read_text().replace('property bool autoHudSend: true', 'property bool autoHudSend: false')
    if args.debug_smoke:
        replay_qml = replay_qml.replace('property bool signalDebugOpen: false', 'property bool signalDebugOpen: true')
    # Keep relative assets/components in the existing new development directory.
    replay_qml_path = root/'hmi'/('test_'+run.name+'.qml')
    replay_qml_path.write_text(replay_qml)
    results = []
    ai = hmi = None
    streams = []
    try:
        for enabled in ((True,) if args.debug_smoke else (False, True)):
            label = 'B_on' if enabled else 'A_off'
            shared = Path('/dev/shm')/(run.name+'.mfr1')
            env = dict(os.environ, DISPLAY=':0', XAUTHORITY='/home/cat/.Xauthority',
                       QT_QPA_PLATFORM='xcb', QT_XCB_GL_INTEGRATION='xcb_egl',
                       QT_FRAME_METADATA_HZ='15', QT_QML_FILE=str(replay_qml_path),
                       ADAS_EDGE_ROOT=str(root/'ai'), ADAS_CONTROL_MODE='DRY_RUN',
                       ADAS_CONTROL_AUTHORIZED='0', ADAS_AUTO_BRAKE='0',
                       ADAS_AI_REQUEST_PORT='38360', ADAS_AI_RESULT_PORT='38361',
                       ADAS_DISABLE_ROAD_SCENE='1', ADAS_LEGACY_HAZARD='1',
                       ADAS_EVENT_TIMELINE=str(run/(label+'_events.jsonl')),
                       QT_AI_TIMING_LOG=str(run/(label+'_qt_apply.jsonl')))
            ai_args = ['/usr/bin/python3', '-u', str(root/'ai/b05p_ai_bridge_launcher.py'),
                       '--live-frame-file', str(shared), '--source', 'LOCAL_REPLAY',
                       '--layout', 'quad', '--display-view', 'front', '--road-quadrant', '3',
                       '--forward-quadrant', '3', '--navigation-quadrant', '1',
                       '--request-port', '38360', '--result-port', '38361',
                       '--main-model', '/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn',
                       '--main-hz', '10', '--hazard-hz', '5', '--idle-timeout', '0',
                       '--timing-log', str(run/(label+'_ai.jsonl')), '--summary', str(run/(label+'_ai_summary.json'))]
            if enabled:
                ai_args += ['--multi-signals', '--signal-attribute-hz', str(args.rate),
                            '--signal-jsonl', str(run/(label+'_attributes.jsonl'))]
            ai_log = (run/(label+'_ai.log')).open('wb'); streams.append(ai_log)
            ai = subprocess.Popen(ai_args, cwd=root/'ai', env=env, stdout=ai_log, stderr=ai_log)
            switch_model(root/'ai', args.model)
            hmi_log = (run/(label+'_hmi.log')).open('wb'); streams.append(hmi_log)
            hmi_args = [str(root/'build/qt_gpu_probe'), '--fps', '30', '--duration', str(args.seconds+8),
                        '--start-mode', str(args.start_view), '--raw-replay', str(raw), '--pgv-layout', 'quad',
                        '--road-quadrant', '3', '--forward-quadrant', '3', '--ai-frame-file', str(shared), '--ai-hz', '0',
                        '--output-json', str(run/(label+'_hmi_summary.json')),
                        '--metrics-jsonl', str(run/(label+'_hmi.jsonl'))]
            hmi = subprocess.Popen(hmi_args, cwd=root, env=env, stdout=hmi_log, stderr=hmi_log)
            # Warm-up before the measured interval; the source clip is identical.
            time.sleep(6)
            if args.debug_smoke:
                subprocess.run(['xfce4-screenshooter', '--fullscreen', '--save', str(run/'qt_signal_debug.png')],
                               env=env, check=True, timeout=20)
            start_ns = time.monotonic_ns(); total0, idle0 = cpu_sample()
            samples = []
            deadline = time.monotonic()+args.seconds
            while time.monotonic() < deadline:
                if hmi.poll() is not None or ai.poll() is not None:
                    raise RuntimeError('A/B child exited; inspect '+str(run))
                time.sleep(.5)
                total, idle = cpu_sample()
                samples.append(dict(timestamp_ns=time.monotonic_ns(), cpu_busy_percent=100*(1-(idle-idle0)/max(1, total-total0))))
                total0, idle0 = total, idle
            end_ns = time.monotonic_ns()
            hmi.wait(8); stop_child(ai); ai = None; hmi = None
            def rows(suffix):
                path = run/(label+suffix)
                return [r for r in map(json.loads, path.read_text().splitlines())
                        if start_ns <= r.get('result_publish_timestamp_ns', r.get('timestamp_ns', 0)) <= end_ns]
            ai_rows = rows('_ai.jsonl')
            hm_rows = [r for r in map(json.loads, (run/(label+'_hmi.jsonl')).read_text().splitlines())
                       if 6000 <= r.get('process_elapsed_ms', 0) <= (args.seconds+6)*1000]
            ages = [r['multi_signal']['signals'][0]['age_ms'] for r in ai_rows
                    if (r.get('multi_signal') or {}).get('signals')]
            attr_ms = [(r.get('multi_signal') or {}).get('metrics', {}).get('attribute_ms') for r in ai_rows]
            attr_ms = [v for v in attr_ms if isinstance(v, (int, float))]
            measurement = dict(label=label, model=args.model, enabled=enabled,
                               seconds=(end_ns-start_ns)/1e9, ai_results=len(ai_rows),
                               ai_fps=len(ai_rows)/((end_ns-start_ns)/1e9),
                               cpu_busy_percent=float(np.mean([s['cpu_busy_percent'] for s in samples])),
                               qt_fps=float(np.mean([r['scene_graph_fps'] for r in hm_rows])) if hm_rows else None,
                               attribute_ms_mean=float(np.mean(attr_ms)) if attr_ms else None,
                               signal_age_ms_p95=float(np.percentile(ages, 95)) if ages else None,
                               hmi_samples=hm_rows, system_samples=samples)
            results.append(measurement)
            (run/'measurements.json').write_text(json.dumps(dict(video=str(args.video), raw_frames=raw_frames,
                rate=args.rate, results=results, note='Short replay with blank non-Q4 cameras; not full competition acceptance'), indent=2))
            print(json.dumps({k: v for k, v in measurement.items() if k not in ('hmi_samples', 'system_samples')}), flush=True)
    finally:
        stop_child(hmi); stop_child(ai)
        for stream in streams:
            stream.close()
        # Remove only the temporary replay file created by this invocation.
        if raw.exists():
            raw.unlink()
        (run/'restored.json').write_text(json.dumps(dict(completed=time.monotonic_ns(),
            live_processes_paused=False, formal_services_modified=False)))
        print('EVIDENCE_DIRECTORY '+str(run), flush=True)


if __name__ == '__main__':
    main()

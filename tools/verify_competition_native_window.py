"""Windows integration check: send keys only to the child test window's PID."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as wt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def is_descendant(pid, root_pid):
    # A Windows venv python.exe may launch the real interpreter as its child.
    class Entry(ctypes.Structure):
        _fields_ = [('size',wt.DWORD),('usage',wt.DWORD),('pid',wt.DWORD),
                    ('heap',ctypes.c_size_t),('module',wt.DWORD),('threads',wt.DWORD),
                    ('parent',wt.DWORD),('priority',wt.LONG),('flags',wt.DWORD),('name',wt.WCHAR*260)]
    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
    kernel.Process32FirstW.argtypes = [wt.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [wt.HANDLE,ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2,0)
    parents = {}
    try:
        entry = Entry();entry.size = ctypes.sizeof(entry)
        ok = kernel.Process32FirstW(snapshot,ctypes.byref(entry))
        while ok:
            parents[entry.pid] = entry.parent
            ok = kernel.Process32NextW(snapshot,ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    visited = set()
    while pid and pid not in visited:
        if pid == root_pid:return True
        visited.add(pid);pid = parents.get(pid,0)
    return False


def wait_for(predicate, timeout=20.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise TimeoutError('Native UI integration condition did not complete')


def capture_client(hwnd, path):
    user,gdi = ctypes.windll.user32,ctypes.windll.gdi32
    user.GetDC.restype = wt.HDC
    gdi.CreateCompatibleDC.restype = wt.HDC
    gdi.CreateCompatibleBitmap.restype = wt.HBITMAP
    gdi.SelectObject.restype = wt.HANDLE
    gdi.CreateCompatibleDC.argtypes = [wt.HDC]
    gdi.CreateCompatibleBitmap.argtypes = [wt.HDC,ctypes.c_int,ctypes.c_int]
    gdi.SelectObject.argtypes = [wt.HDC,wt.HANDLE]
    user.GetDC.argtypes = [wt.HWND]
    user.PrintWindow.argtypes = [wt.HWND,wt.HDC,wt.UINT]
    gdi.GetDIBits.argtypes = [wt.HDC,wt.HBITMAP,wt.UINT,wt.UINT,ctypes.c_void_p,ctypes.c_void_p,wt.UINT]
    gdi.DeleteObject.argtypes = [wt.HANDLE]
    gdi.DeleteDC.argtypes = [wt.HDC]
    user.ReleaseDC.argtypes = [wt.HWND,wt.HDC]
    rect = wt.RECT()
    user.GetClientRect(hwnd,ctypes.byref(rect))
    width,height = rect.right,rect.bottom
    if min(width,height) <= 0:
        raise RuntimeError('Native client rectangle is empty')
    dc = user.GetDC(hwnd)
    memory = gdi.CreateCompatibleDC(dc)
    bitmap = gdi.CreateCompatibleBitmap(dc,width,height)
    old = gdi.SelectObject(memory,bitmap)
    try:
        if not user.PrintWindow(hwnd,memory,1):
            raise OSError('PrintWindow did not capture the test client')
        # BITMAPINFOHEADER: uncompressed top-down 32-bit pixels.
        import struct
        header = struct.pack('<IiiHHIIiiII',40,width,-height,1,32,0,width*height*4,0,0,0,0)
        info = ctypes.create_string_buffer(header+b'\0'*16)
        pixels = ctypes.create_string_buffer(width*height*4)
        if gdi.GetDIBits(memory,bitmap,0,height,pixels,info,0) != height:
            raise OSError('Cannot read captured native client pixels')
        image = Image.frombuffer('RGB',(width,height),pixels.raw,'raw','BGRX',0,1)
        image.save(path)
    finally:
        gdi.SelectObject(memory,old)
        gdi.DeleteObject(bitmap);gdi.DeleteDC(memory);user.ReleaseDC(hwnd,dc)


def verify(output, *, hazards=False):
    if os.name != 'nt':
        raise RuntimeError('This native event check requires Windows')
    output.mkdir(parents=True,exist_ok=True)
    app_output = output/'app'
    title = f'Tesla-BevCam Competition Integration Check {os.getpid()}'
    command = [sys.executable,str(ROOT/'tools/run_competition_console.py'),'--start','12','--paused',
               '--profile',str(ROOT/'configs/competition_native_official.json'),
               '--output',str(app_output),'--window-title',title]
    if hazards:
        command.append('--hazards')
    env = os.environ.copy();env['PYTHONIOENCODING'] = 'utf-8'
    with (output/'process.txt').open('w',encoding='utf-8') as process_log:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=process_log,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            user = ctypes.windll.user32
            user.FindWindowW.restype = wt.HWND
            user.FindWindowW.argtypes = [wt.LPCWSTR,wt.LPCWSTR]
            user.GetWindowThreadProcessId.argtypes = [wt.HWND,ctypes.POINTER(wt.DWORD)]
            user.PostMessageW.argtypes = [wt.HWND,wt.UINT,wt.WPARAM,wt.LPARAM]
            user.GetClassNameW.argtypes = [wt.HWND,wt.LPWSTR,ctypes.c_int]
            callback_type = ctypes.WINFUNCTYPE(wt.BOOL,wt.HWND,wt.LPARAM)
            user.EnumChildWindows.argtypes = [wt.HWND,callback_type,wt.LPARAM]
            hwnd = wait_for(lambda:user.FindWindowW(None,title))
            owner = wt.DWORD();user.GetWindowThreadProcessId(hwnd,ctypes.byref(owner))
            if not is_descendant(owner.value,process.pid):
                raise RuntimeError(f'Refusing input outside child process tree: owner={owner.value}, root={process.pid}')
            children = []
            @callback_type
            def visit(child,param):
                name = ctypes.create_unicode_buffer(128)
                user.GetClassNameW(child,name,128)
                children.append((child,name.value))
                return True
            user.EnumChildWindows(hwnd,visit,0)
            target = next((child for child,name in children if name == 'HighGUI class'),hwnd)
            def last():
                path = app_output/'trace.jsonl'
                if not path.exists():return None
                lines = path.read_text(encoding='utf-8').splitlines()
                if not lines:return None
                try:return json.loads(lines[-1])
                except json.JSONDecodeError:return None
            def send(key):
                if not user.PostMessageW(target,0x0102,ord(key) if isinstance(key,str) else key,0):
                    raise OSError('Cannot post a key to the child HighGUI window')
            initial = wait_for(lambda:last() and last()['signal']['light']=='RED' and last())
            if hazards:
                assert initial['hazards']['reason'] != 'disabled'
            send('g')
            forced = wait_for(lambda:last() and 'RED_LIGHT' in last()['decision']['alarms'] and last())
            assert forced['decision']['maneuver']=='stop' and forced['decision']['simulated_motion']=='go'
            time.sleep(.15)
            capture_client(hwnd,output/'native_client.png')
            send('s')
            stopped = wait_for(lambda:last() and last()['controls']['motion']=='stop' and last())
            assert stopped['decision']['alarms']==[]
            send('D')
            wait_for(lambda:last() and last()['controls']['mode']=='road')
            send('G')
            road = wait_for(lambda:last() and last()['controls']['mode']=='road' and 'RED_LIGHT' in last()['decision']['alarms'] and last())
            assert road['controls']['signalized'] and road['decision']['maneuver']=='stop'
            send('I');send('S')
            wait_for(lambda:last() and last()['controls']['mode']=='intersection' and last()['controls']['motion']=='stop')
            send('1')
            wait_for(lambda:last() and last()['controls']['lane']=='left')
            send('.')
            stepped = wait_for(lambda:last() and last()['frame']>initial['frame'] and last())
            assert stepped['frame']==initial['frame']+1
            send('r')
            restarted = wait_for(lambda:last() and last()['frame']<initial['frame'] and last())
            assert restarted['controls']['motion']=='auto' and restarted['signal']['light']=='UNKNOWN'
            send(27)
            process.wait(timeout=15)
            if process.returncode:
                raise RuntimeError(f'Native child exited with {process.returncode}')
            summary = json.loads((app_output/'summary.json').read_text(encoding='utf-8'))
            assert summary['native_window_was_visible']
            report = {'evidence':'WINDOWS_NATIVE_CHILD_GUI_EVENT_TEST_NOT_BOARD', 'passed':True,
                      'window_pid_verified':True,'native_window_visible':True,
                      'forced_red_go_alarm':True,'stop_clears_red_alarm':True,'lane_key_changes_state':True,
                      'bound_signal_gate_survives_scene_switch':True,'uppercase_shortcuts_work':True,
                      'single_frame_step':True,'restart_invalidates_binding_and_forced_go':True,
                      'escape_exits_cleanly':True,'client_capture':str((output/'native_client.png').resolve()),
                      'hazards_enabled':hazards,
                      'input_source':'SCRIPTED_WIN32_MESSAGES_TO_THIS_TEST_CHILD_ONLY','command':command}
            (output/'gui_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2))
            return report
        finally:
            if process.poll() is None:
                process.terminate();process.wait(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hazards',action='store_true')
    args = parser.parse_args()
    verify(args.output.resolve(),hazards=args.hazards)

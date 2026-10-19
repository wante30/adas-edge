#!/usr/bin/env python3
"""Click the probe's three QML mode buttons and capture the board screen."""

import ctypes
import subprocess
import sys
import time
from pathlib import Path


def main(output_dir):
    x11 = ctypes.CDLL("libX11.so.6")
    xtst = ctypes.CDLL("libXtst.so.6")
    window_t = ctypes.c_ulong
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XDefaultRootWindow.restype = window_t
    x11.XQueryTree.argtypes = [
        ctypes.c_void_p, window_t, ctypes.POINTER(window_t), ctypes.POINTER(window_t),
        ctypes.POINTER(ctypes.POINTER(window_t)), ctypes.POINTER(ctypes.c_uint)
    ]
    x11.XFetchName.argtypes = [ctypes.c_void_p, window_t, ctypes.POINTER(ctypes.c_char_p)]
    x11.XMapRaised.argtypes = [ctypes.c_void_p, window_t]
    x11.XSetInputFocus.argtypes = [ctypes.c_void_p, window_t, ctypes.c_int, ctypes.c_ulong]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XFree.argtypes = [ctypes.c_void_p]
    xtst.XTestFakeMotionEvent.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong
    ]
    xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("could not connect to the current X11 display")

    def find_probe_window(window, depth=0):
        if depth > 8:
            return None
        root_return = window_t()
        parent_return = window_t()
        children = ctypes.POINTER(window_t)()
        child_count = ctypes.c_uint()
        if not x11.XQueryTree(display, window, ctypes.byref(root_return),
                              ctypes.byref(parent_return), ctypes.byref(children),
                              ctypes.byref(child_count)):
            return None
        try:
            for child in children[:child_count.value]:
                title = ctypes.c_char_p()
                if x11.XFetchName(display, child, ctypes.byref(title)) and title.value:
                    matches = b"Qt Quick GPU Probe" in title.value
                    x11.XFree(title)
                    if matches:
                        return child
                nested = find_probe_window(child, depth + 1)
                if nested:
                    return nested
        finally:
            if children:
                x11.XFree(children)
        return None

    probe_window = find_probe_window(x11.XDefaultRootWindow(display))
    if not probe_window:
        raise RuntimeError("could not find the Qt Quick GPU Probe window")
    x11.XMapRaised(display, probe_window)
    x11.XSetInputFocus(display, probe_window, 2, 0)
    x11.XFlush(display)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, x in (("single", 496), ("quad", 618), ("spatial", 740), ("single_return", 496)):
        xtst.XTestFakeMotionEvent(display, -1, x, 46, 0)
        xtst.XTestFakeButtonEvent(display, 1, 1, 0)
        xtst.XTestFakeButtonEvent(display, 1, 0, 0)
        x11.XFlush(display)
        time.sleep(1.0)
        capture = output_dir / f"qt_mode_{name}.xwd"
        subprocess.run(["xwd", "-root", "-silent", "-out", str(capture)], check=True)

    x11.XCloseDisplay(display)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_DIR")
    main(sys.argv[1])

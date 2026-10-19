#!/usr/bin/env python3
"""Click one HUD mode tab without closing the running app."""
import ctypes
import sys

x11 = ctypes.CDLL("libX11.so.6")
xtst = ctypes.CDLL("libXtst.so.6")
x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XFlush.argtypes = [ctypes.c_void_p]
x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

display = x11.XOpenDisplay(None)
if not display:
    raise SystemExit("cannot open DISPLAY")
x = int(sys.argv[1]) if len(sys.argv) > 1 else 483
y = int(sys.argv[2]) if len(sys.argv) > 2 else 32
xtst.XTestFakeMotionEvent(display, -1, x, y, 0)
xtst.XTestFakeButtonEvent(display, 1, 1, 0)
xtst.XTestFakeButtonEvent(display, 1, 0, 0)
x11.XFlush(display)
x11.XCloseDisplay(display)
print(f"clicked {x},{y}")

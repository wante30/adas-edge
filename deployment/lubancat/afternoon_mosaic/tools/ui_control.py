"""Operate only the isolated 1280x720 HMI through its existing visible controls."""
import argparse
import ctypes as C
import os

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XAUTHORITY", "/home/cat/.Xauthority")
x = C.CDLL("libX11.so.6")
t = C.CDLL("libXtst.so.6")
x.XOpenDisplay.argtypes = [C.c_char_p]
x.XOpenDisplay.restype = C.c_void_p
x.XFlush.argtypes = [C.c_void_p]
x.XCloseDisplay.argtypes = [C.c_void_p]
t.XTestFakeMotionEvent.argtypes = [C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_ulong]
t.XTestFakeButtonEvent.argtypes = [C.c_void_p,C.c_uint,C.c_int,C.c_ulong]
p = argparse.ArgumentParser()
p.add_argument("x",type=int); p.add_argument("y",type=int)
a = p.parse_args()
assert 0 <= a.x < 1280 and 0 <= a.y < 720
d = x.XOpenDisplay(None)
assert d, "X display unavailable"
t.XTestFakeMotionEvent(d,-1,a.x,a.y,0)
t.XTestFakeButtonEvent(d,1,1,0)
t.XTestFakeButtonEvent(d,1,0,0)
x.XFlush(d)
x.XCloseDisplay(d)

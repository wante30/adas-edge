"""GTK3 display backend for Linux OpenCV builds without HighGUI."""
from __future__ import annotations

from collections import deque
import sys
import time


def choose_display_backend(requested="auto", platform_name=None, gui_backend=None):
    if requested not in ("auto", "opencv", "gtk"):
        raise ValueError(f"Unsupported display backend: {requested}")
    if requested != "auto":
        return requested
    platform_name = platform_name or sys.platform
    if gui_backend is None:
        import cv2

        gui_backend = next(
            (line.split(":", 1)[1].strip() for line in cv2.getBuildInformation().splitlines()
             if line.strip().startswith("GUI:")),
            "NONE",
        )
    if platform_name.startswith("linux") and gui_backend.strip().upper() == "NONE":
        return "gtk"
    return "opencv"


def fit_canvas(canvas_size, widget_size):
    canvas_width, canvas_height = canvas_size
    widget_width, widget_height = widget_size
    if min(canvas_width, canvas_height, widget_width, widget_height) <= 0:
        raise ValueError("Canvas and widget dimensions must be positive")
    scale = min(widget_width / canvas_width, widget_height / canvas_height)
    width, height = canvas_width * scale, canvas_height * scale
    return scale, (widget_width - width) / 2, (widget_height - height) / 2, width, height


def screen_to_canvas(x, y, transform):
    scale, left, top, width, height = transform
    if scale <= 0 or x < left or y < top or x >= left + width or y >= top + height:
        return None
    return round((x - left) / scale), round((y - top) / scale)


def prepare_display_rgb(image, display_size, *, input_rgb=False):
    import cv2

    width, height = display_size
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image if input_rgb else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class GtkDisplay:
    def __init__(self, title, canvas_size, window_size):
        import gi

        gi.require_version("Gdk", "3.0")
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gdk, GdkPixbuf, GLib, Gtk

        self.Gdk = Gdk
        self.GdkPixbuf = GdkPixbuf
        self.GLib = GLib
        self.Gtk = Gtk
        self.canvas_size = tuple(canvas_size)
        self.window_size = tuple(window_size)
        self.transform = fit_canvas(self.canvas_size, self.window_size)
        self.key_queue = deque(maxlen=32)
        self.mouse_callback = None
        self.pixbuf = None
        self.closed = False
        self.context = GLib.MainContext.default()

        self.window = Gtk.Window(title=title)
        self.window.set_decorated(False)
        self.window.set_default_size(*self.window_size)
        self.window.connect("key-press-event", self._on_key_press)
        self.window.connect("destroy", self._on_destroy)
        self.area = Gtk.DrawingArea()
        self.area.set_can_focus(True)
        self.area.set_hexpand(True)
        self.area.set_vexpand(True)
        self.area.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
            | Gdk.EventMask.BUTTON1_MOTION_MASK
        )
        self.area.connect("draw", self._on_draw)
        self.area.connect("button-press-event", self._on_button_press)
        self.area.connect("button-release-event", self._on_button_release)
        self.area.connect("motion-notify-event", self._on_motion)
        self.window.add(self.area)
        self.window.fullscreen()
        self.window.show_all()
        self.area.grab_focus()
        self._pump_events()

    def set_mouse_callback(self, callback):
        self.mouse_callback = callback

    def _on_destroy(self, *_):
        self.closed = True

    def _on_draw(self, widget, context):
        allocation = widget.get_allocation()
        self.transform = fit_canvas(self.canvas_size, (allocation.width, allocation.height))
        context.set_source_rgb(0.02, 0.04, 0.06)
        context.paint()
        if self.pixbuf is not None:
            _, left, top, _, _ = self.transform
            context.save()
            context.translate(left, top)
            self.Gdk.cairo_set_source_pixbuf(context, self.pixbuf, 0, 0)
            context.paint()
            context.restore()
        return False

    def _on_key_press(self, _widget, event):
        name = self.Gdk.keyval_name(event.keyval)
        if name == "Escape":
            key = 27
        elif name == "space":
            key = 32
        elif name == "period":
            key = ord(".")
        else:
            codepoint = self.Gdk.keyval_to_unicode(event.keyval)
            key = codepoint if 0 < codepoint < 128 else None
        if key is not None:
            self.key_queue.append(key)
        return key is not None

    def _dispatch_mouse(self, event_code, event, flags=0):
        if self.mouse_callback is None:
            return False
        point = screen_to_canvas(event.x, event.y, self.transform)
        if point is None:
            return False
        self.mouse_callback(event_code, point[0], point[1], flags, None)
        return True

    def _on_button_press(self, _widget, event):
        if event.button != 1:
            return False
        import cv2

        return self._dispatch_mouse(cv2.EVENT_LBUTTONDOWN, event, 1)

    def _on_button_release(self, _widget, event):
        if event.button != 1:
            return False
        import cv2

        return self._dispatch_mouse(cv2.EVENT_LBUTTONUP, event)

    def _on_motion(self, _widget, event):
        import cv2

        flags = int(bool(event.state & self.Gdk.ModifierType.BUTTON1_MASK))
        return self._dispatch_mouse(cv2.EVENT_MOUSEMOVE, event, flags)

    def present(self, image, *, input_rgb=False):
        allocation = self.area.get_allocation()
        widget_size = (allocation.width, allocation.height)
        if min(widget_size) <= 0:
            widget_size = self.window_size
        self.transform = fit_canvas(self.canvas_size, widget_size)
        _, _, _, display_width, display_height = self.transform
        display_width, display_height = max(1, round(display_width)), max(1, round(display_height))
        rgb = prepare_display_rgb(image, (display_width, display_height), input_rgb=input_rgb)
        height, width = rgb.shape[:2]
        payload = self.GLib.Bytes.new(rgb.tobytes())
        self.pixbuf = self.GdkPixbuf.Pixbuf.new_from_bytes(
            payload, self.GdkPixbuf.Colorspace.RGB, False, 8, width, height, width * 3
        )
        self.area.queue_draw()

    def _pump_events(self):
        while self.context.pending():
            self.context.iteration(False)

    def is_visible(self):
        return not self.closed and self.window.get_visible()

    def wait_key(self, timeout_ms):
        deadline = time.monotonic() + max(0, timeout_ms) / 1000
        while True:
            self._pump_events()
            if self.key_queue:
                return self.key_queue.popleft()
            if self.closed:
                return -1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return -1
            time.sleep(min(0.003, remaining))

    def choose_video(self):
        Gtk = self.Gtk
        dialog = Gtk.FileChooserDialog(
            title="打开前视视频",
            parent=self.window,
            action=Gtk.FileChooserAction.OPEN,
            buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                     Gtk.STOCK_OPEN, Gtk.ResponseType.OK),
        )
        file_filter = Gtk.FileFilter()
        file_filter.set_name("Video files")
        for pattern in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
            file_filter.add_pattern(pattern)
        dialog.add_filter(file_filter)
        response = dialog.run()
        filename = dialog.get_filename() if response == Gtk.ResponseType.OK else None
        dialog.destroy()
        self._pump_events()
        return filename

    def close(self):
        if not self.closed:
            self.window.destroy()
            self._pump_events()

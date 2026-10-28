"""Create the one-time RGBA overlay used by the isolated DESIGN_PREVIEW mode.

The file is intentionally a transparent UI layer. Video thumbnails are sampled by
the EGL renderer underneath it, so this script is never part of the frame loop.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1280, 720
THUMBNAILS = [(16, 548, 204, 150), (230, 548, 204, 150), (444, 548, 204, 150), (658, 548, 204, 150)]
SPATIAL = (872, 548, 392, 150)

BG = (5, 15, 27, 232)
PANEL = (7, 25, 43, 205)
PANEL_SOFT = (8, 31, 53, 178)
CYAN = (74, 202, 255, 235)
CYAN_SOFT = (53, 151, 205, 185)
TEXT = (235, 245, 250, 255)
MUTED = (157, 185, 205, 235)
GREEN = (41, 235, 153, 255)
AMBER = (255, 194, 83, 255)
RED = (255, 91, 101, 255)


def load_font(path: str, size: int, bold: bool = False):
    if path:
        return ImageFont.truetype(path, size, index=0)
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/Noto Sans SC Bold (TrueType).otf" if bold else "C:/Windows/Fonts/Noto Sans SC (TrueType).otf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size, index=0)
    return ImageFont.load_default()


def rounded(draw, rect, radius=14, fill=PANEL, outline=CYAN_SOFT, width=1):
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)
        return
    # Pillow on the board may predate rounded_rectangle. Keep the same geometry
    # with a rectangular fallback; the visual hierarchy remains unchanged.
    draw.rectangle(rect, fill=fill, outline=outline, width=width)


def label(draw, xy, value, font, fill=TEXT, anchor=None):
    draw.text(xy, value, font=font, fill=fill, anchor=anchor)


def line(draw, points, fill, width=2):
    draw.line(points, fill=fill, width=width, joint="curve")


def draw_header(draw, fonts):
    title, body, small, tiny = fonts
    draw.rectangle((0, 0, WIDTH, 58), fill=(5, 16, 29, 245))
    line(draw, ((0, 57), (WIDTH, 57)), (55, 155, 213, 190), 1)

    # Minimal car mark keeps the title legible without turning the UI into a logo wall.
    line(draw, ((22, 38), (28, 22), (39, 18), (53, 18), (63, 22), (69, 38)), CYAN, 3)
    line(draw, ((29, 29), (59, 29)), CYAN, 2)
    draw.ellipse((25, 36, 32, 43), fill=CYAN)
    draw.ellipse((57, 36, 64, 43), fill=CYAN)
    label(draw, (81, 11), "ADAS", title, fill=(221, 244, 255, 255))
    label(draw, (161, 15), "智能驾驶感知系统", body, fill=TEXT)

    modes = [(360, "单路检测", True), (456, "四路检测", False), (552, "空间视图", False)]
    for x, value, active in modes:
        rounded(draw, (x, 10, x + 88, 46), 22,
                fill=(23, 112, 190, 205) if active else (11, 34, 57, 188),
                outline=(91, 218, 255, 240) if active else (65, 128, 169, 170), width=1)
        label(draw, (x + 44, 28), value, small, fill=TEXT if active else MUTED, anchor="mm")

    rounded(draw, (650, 10, 914, 46), 12, fill=(7, 24, 42, 190), outline=CYAN_SOFT)
    label(draw, (665, 16), "当前视角", tiny, fill=MUTED)
    label(draw, (724, 14), "前视（HDMI-IN）", small, fill=TEXT)
    label(draw, (877, 14), "⌄", body, fill=CYAN)

    stats = [(932, "PCIe", "已连接", GREEN), (1004, "视频", "58.2 FPS", TEXT),
             (1080, "AI", "8.1 FPS", TEXT), (1152, "NPU", "100%", TEXT), (1212, "GPU", "47%", TEXT)]
    for x, heading, value, color in stats:
        label(draw, (x, 10), heading, tiny, fill=MUTED)
        label(draw, (x, 29), value, small, fill=color)
    label(draw, (1258, 10), "15:27", body, fill=TEXT, anchor="ra")
    label(draw, (1258, 36), "设计预览", tiny, fill=AMBER, anchor="ra")


def draw_signal_card(draw, fonts):
    title, body, small, tiny = fonts
    x, y, w, h = 24, 78, 300, 182
    rounded(draw, (x, y, x + w, y + h), 15, fill=PANEL, outline=CYAN_SOFT, width=1)
    # Traffic light icon.
    rounded(draw, (44, 106, 79, 167), 12, fill=(8, 44, 57, 230), outline=(66, 185, 164, 220))
    for cy, color in ((117, RED), (136, AMBER), (155, GREEN)):
        draw.ellipse((53, cy - 7, 70, cy + 10), fill=color if color is GREEN else (25, 75, 74, 220), outline=color, width=2)
    label(draw, (96, 100), "信号灯：", body, fill=TEXT)
    label(draw, (174, 100), "绿灯", body, fill=GREEN)
    label(draw, (96, 134), "状态正常  |  置信度 0.93", small, fill=MUTED)
    line(draw, ((40, 184), (308, 184)), (62, 120, 158, 170), 1)
    stats = [(51, "▰", "车辆", "12"), (145, "●", "行人", "3"), (237, "◌", "非机动车", "1")]
    for px, icon, name, value in stats:
        label(draw, (px, 195), icon, body, fill=CYAN)
        label(draw, (px + 22, 195), name, tiny, fill=MUTED)
        label(draw, (px + 22, 214), value, body, fill=TEXT)
    label(draw, (42, 242), "当前感知对象", tiny, fill=MUTED)
    label(draw, (286, 242), "DESIGN PREVIEW", tiny, fill=AMBER, anchor="ra")


def draw_nav_card(draw, fonts):
    title, body, small, tiny = fonts
    x, y, w, h = 1004, 78, 252, 136
    rounded(draw, (x, y, x + w, y + h), 15, fill=PANEL_SOFT, outline=CYAN_SOFT, width=1)
    # Direction arrow.
    line(draw, ((1036, 144), (1036, 112), (1050, 98), (1075, 98)), (40, 185, 255, 255), 12)
    line(draw, ((1068, 91), (1091, 101), (1068, 114)), (40, 185, 255, 255), 8)
    label(draw, (1112, 99), "前方", tiny, fill=MUTED)
    label(draw, (1112, 118), "50 m 右转", title, fill=TEXT)
    label(draw, (1112, 163), "科技园路", body, fill=TEXT)
    label(draw, (1112, 195), "导航识别  |  置信度 0.89", tiny, fill=MUTED)
    label(draw, (1242, 96), "设计预览", tiny, fill=AMBER, anchor="ra")


def draw_hazard_card(draw, fonts):
    title, body, small, tiny = fonts
    x, y, w, h = 1004, 224, 252, 148
    rounded(draw, (x, y, x + w, y + h), 15, fill=(48, 15, 28, 198), outline=(225, 71, 87, 225), width=1)
    label(draw, (1020, 239), "⚠  危险事件", body, fill=RED)
    rows = [("▰", "前车距离较近", "0.8s"), ("●", "右侧行人注意", "0.6s"), ("╱", "无效变道风险", "2.1s")]
    for i, (icon, text, age) in enumerate(rows):
        yy = 274 + i * 30
        label(draw, (1022, yy), icon, body, fill=RED)
        label(draw, (1050, yy + 1), text, small, fill=TEXT)
        label(draw, (1242, yy + 1), age, small, fill=RED, anchor="ra")
        if i != len(rows) - 1:
            line(draw, ((1020, yy + 25), (1242, yy + 25)), (132, 45, 61, 155), 1)
    label(draw, (1242, 360), "设计预览数据", tiny, fill=AMBER, anchor="ra")


def draw_lane_card(draw, fonts):
    title, body, small, tiny = fonts
    x, y, w, h = 1004, 382, 252, 148
    rounded(draw, (x, y, x + w, y + h), 15, fill=PANEL_SOFT, outline=CYAN_SOFT, width=1)
    label(draw, (1020, 397), "▣  车道状态", body, fill=TEXT)
    # Lane glyph.
    line(draw, ((1034, 522), (1054, 436)), (117, 190, 247, 220), 2)
    line(draw, ((1084, 522), (1068, 436)), (117, 190, 247, 220), 2)
    line(draw, ((1059, 522), (1059, 436)), AMBER, 2)
    label(draw, (1100, 436), "当前车道：", small, fill=MUTED)
    label(draw, (1180, 436), "直行", small, fill=GREEN)
    label(draw, (1100, 462), "左侧车道：", small, fill=MUTED)
    label(draw, (1180, 462), "虚线", small, fill=AMBER)
    label(draw, (1100, 488), "右侧车道：", small, fill=MUTED)
    label(draw, (1180, 488), "实线", small, fill=RED)
    label(draw, (1242, 516), "FPGA 元数据", tiny, fill=MUTED, anchor="ra")


def draw_thumbnails(draw, fonts):
    title, body, small, tiny = fonts
    names = [("1", "前视", "HDMI-IN"), ("2", "左视", "ETHERNET"), ("3", "后视", "OV5640"), ("4", "导航", "HSST")]
    for index, (x, y, w, h) in enumerate(THUMBNAILS):
        rounded(draw, (x, y, x + w, y + h), 13, fill=(8, 22, 38, 205), outline=(40, 125, 168, 215), width=1)
        # Transparent viewport: the EGL quad texture is composited underneath this layer.
        draw.rectangle((x + 6, y + 6, x + w - 6, y + 106), fill=(0, 0, 0, 0))
        line(draw, ((x + 6, y + 6), (x + w - 6, y + 6)), (93, 193, 236, 175), 1)
        line(draw, ((x + 6, y + 106), (x + w - 6, y + 106)), (93, 193, 236, 175), 1)
        line(draw, ((x + 6, y + 6), (x + 6, y + 106)), (93, 193, 236, 175), 1)
        line(draw, ((x + w - 6, y + 6), (x + w - 6, y + 106)), (93, 193, 236, 175), 1)
        if index == 0:
            rounded(draw, (x + 2, y + 2, x + w - 2, y + h - 2), 12, fill=(0, 0, 0, 0), outline=(53, 206, 255, 255), width=3)
        draw.ellipse((x + 10, y + 117, x + 32, y + 139), fill=(86, 201, 255, 235))
        label(draw, (x + 21, y + 128), names[index][0], body, fill=(4, 32, 54, 255), anchor="mm")
        label(draw, (x + 42, y + 116), names[index][1], small, fill=TEXT)
        label(draw, (x + 42, y + 137), names[index][2], tiny, fill=MUTED)


def draw_spatial(draw, fonts):
    title, body, small, tiny = fonts
    x, y, w, h = SPATIAL
    rounded(draw, (x, y, x + w, y + h), 15, fill=(8, 24, 42, 225), outline=CYAN_SOFT, width=1)
    label(draw, (x + 16, y + 11), "ADAS 空间示意", body, fill=TEXT)
    label(draw, (x + w - 14, y + 13), "设计预览", tiny, fill=AMBER, anchor="ra")
    # Perspective lane and vehicles; explicitly a schematic, not a calibrated distance view.
    line(draw, ((x + 144, y + h - 8), (x + 184, y + 49)), (52, 143, 196, 180), 2)
    line(draw, ((x + 284, y + h - 8), (x + 228, y + 49)), (52, 143, 196, 180), 2)
    line(draw, ((x + 210, y + h - 8), (x + 210, y + 58)), AMBER, 2)
    line(draw, ((x + 166, y + h - 8), (x + 190, y + 74)), (50, 178, 236, 165), 2)
    line(draw, ((x + 254, y + h - 8), (x + 232, y + 74)), (50, 178, 236, 165), 2)
    # Rear-view self car.
    rounded(draw, (x + 194, y + 103, x + 235, y + 139), radius=7,
            fill=(235, 241, 245, 245), outline=(121, 193, 226, 255), width=2)
    draw.rectangle((x + 201, y + 109, x + 228, y + 119), fill=(45, 80, 103, 240))
    for cx, cy, color in ((x + 150, y + 87, (26, 132, 242, 240)), (x + 269, y + 84, (26, 132, 242, 240)), (x + 216, y + 61, (239, 62, 79, 245))):
        rounded(draw, (cx - 14, cy - 9, cx + 14, cy + 9), radius=5,
                fill=color, outline=(102, 214, 255, 220), width=1)
    draw.polygon(((x + 343, y + 72), (x + 351, y + 87), (x + 335, y + 87)), fill=(83, 212, 255, 235))
    label(draw, (x + 16, y + 117), "车辆 12   行人 3   非机动车 1", tiny, fill=MUTED)
    label(draw, (x + 16, y + 136), "空间示意 · 未标定距离", tiny, fill=AMBER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--font", default="")
    args = parser.parse_args()

    title = load_font(args.font, 25, bold=True)
    body = load_font(args.font, 18, bold=True)
    small = load_font(args.font, 14)
    tiny = load_font(args.font, 11)
    fonts = (title, body, small, tiny)

    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_header(draw, fonts)
    draw_signal_card(draw, fonts)
    draw_nav_card(draw, fonts)
    draw_hazard_card(draw, fonts)
    draw_lane_card(draw, fonts)
    draw_thumbnails(draw, fonts)
    draw_spatial(draw, fonts)

    # Keep thumbnail image rectangles transparent; the main road viewport is transparent by default.
    for x, y, w, h in THUMBNAILS:
        draw.rectangle((x + 7, y + 7, x + w - 7, y + 105), fill=(0, 0, 0, 0))

    rgba = image.tobytes()
    header = struct.pack("<4sHHIQI", b"UI1\0", WIDTH, HEIGHT, 0, time.time_ns(), len(rgba))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + rgba)
    print(f"wrote {output} ({WIDTH}x{HEIGHT}, {len(rgba)} RGBA bytes)")


if __name__ == "__main__":
    main()

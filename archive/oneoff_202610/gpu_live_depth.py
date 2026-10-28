from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
old='''            const float tx = std::clamp(float(-1.75 + (cx - .5) * 14.0), -6.8f, 6.8f);
            const float tz = std::clamp(float(3.8 + (1.0 - bottom) * 51.0), 3.8f, 55.0f);'''
new='''            const float tx = std::clamp(float(-1.75 + (cx - .5) * 14.0), -6.8f, 6.8f);
            // Non-calibrated relative depth: bbox width is a scale cue, not a
            // metric distance. It separates near/far detections much better
            // than a constant bottom-to-depth line and never yields metres.
            const float widthDepth = 2600.0f / std::max(24.0f, float(b.w));
            const float bottomDepth = 3.8f + float(1.0 - bottom) * 42.0f;
            const float tz = std::clamp(0.72f * widthDepth + 0.28f * bottomDepth, 4.0f, 62.0f);'''
if old not in s: raise SystemExit('live map marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('live depth patched')

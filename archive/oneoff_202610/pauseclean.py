from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
old='''        if payload is None or time.time() - last_payload > 1.5:
            scene.clear()
        else:
            tracked = scene.update(payload)
            try:
                DEBUG.write_text(json.dumps({
                    "source": scene.source, "tracks": tracked,
                    "source_frame_id": payload.get("source_frame_id"),
                    "timestamp_ns": payload.get("timestamp_ns"),
                }, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            img = render(up, tracked)'''
new='''        if payload is None or time.time() - last_payload > 1.5:
            scene.clear()
            tracked = []
        else:
            tracked = scene.update(payload)
            try:
                DEBUG.write_text(json.dumps({
                    "source": scene.source, "tracks": tracked,
                    "source_frame_id": payload.get("source_frame_id"),
                    "timestamp_ns": payload.get("timestamp_ns"),
                }, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        # Always emit a frame, including the ego-only cleanup frame; otherwise
        # the last QML image would freeze with stale detections after AI stops.
        img = render(up, tracked)'''
if old not in s: raise SystemExit('loop marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('pause cleanup patched')

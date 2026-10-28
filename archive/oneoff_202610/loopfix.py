from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
start=s.index('        if payload is None or time.time() - last_payload > 1.5:')
end=s.index('\n\nif __name__ == "__main__":', start)
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
        # Always emit, including the ego-only cleanup frame after AI stops.
        img = render(up, tracked)
        ok, encoded = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if ok:
            for target in (OUT_A, OUT_B):
                tmp = target.with_name(target.name + ".tmp")
                tmp.write_bytes(encoded.tobytes())
                os.replace(tmp, target)
            tick += 1
'''
s=s[:start]+new+s[end:]
p.write_text(s,encoding='utf-8')
print('loop replaced')

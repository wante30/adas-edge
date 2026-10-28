from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''            tmp = target.with_suffix(".tmp")
            cv2.imwrite(str(tmp), img)
            os.replace(tmp, target)''','''            ok, encoded = cv2.imencode(".bmp", img)
            if ok:
                tmp = target.with_name(target.name + ".tmp")
                tmp.write_bytes(encoded.tobytes())
                os.replace(tmp, target)''')
p.write_text(s,encoding='utf-8')
print('patched')

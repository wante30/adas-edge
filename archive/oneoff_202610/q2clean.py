from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\q2_navigation_reader.py')
s=p.read_text(encoding='utf-8')
old='''    def _clean(text):
        text = str(text).strip()
        for wrong, right in OCR_FIXES:'''
new='''    def _clean(text):
        text = str(text).strip()
        # Recognition of a tight crop can append a star/period that the line
        # detector does not; it is not part of the distance or road name.
        text = text.strip(" *·•,，。")
        for wrong, right in OCR_FIXES:'''
if old not in s: raise SystemExit('clean marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('clean patched')

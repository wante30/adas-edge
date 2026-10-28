from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\hud_bridge.py')
s=p.read_text(encoding='utf-8')
old='''        applied = self.rd(OFF_EXT_APPLIED)
        result = {
            "type": "ack",
            "ok": bool(not (status & 1) and applied == self.ext_seq),'''
new='''        applied = self.rd(OFF_EXT_APPLIED)
        # BAR1 applies the committed word on the following ready edge. The
        # accepted sequence is therefore allowed to be one behind the current
        # commit; this is pipeline latency, not a rejected write.
        lag = (self.ext_seq - applied) & 0xFFFF
        result = {
            "type": "ack",
            "ok": bool(not (status & 1) and (applied == self.ext_seq or lag == 1)),
            "pipeline_lag": lag,'''
if old not in s: raise SystemExit('bridge ack marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('bridge ack patched')

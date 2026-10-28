from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\hud_bridge.py')
s=p.read_text(encoding='utf-8')
old='''        self.ext_seq = 0
        self.legacy_seq = 0
        if self.rd(OFF_MAGIC) != MAGIC:
            raise SystemExit("HUD magic mismatch; refuse to write BAR1")'''
new='''        if self.rd(OFF_MAGIC) != MAGIC:
            raise SystemExit("HUD magic mismatch; refuse to write BAR1")
        # BAR1 persists the applied sequence across a bridge restart. Resume
        # from it; starting at 0 makes every new commit look stale to the FPGA.
        self.ext_seq = self.rd(OFF_EXT_APPLIED) & 0xFFFF
        self.legacy_seq = self.ext_seq & 0xFFFF'''
if old not in s: raise SystemExit('init marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('bridge resume patched')

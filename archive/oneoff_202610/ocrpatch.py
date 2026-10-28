from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
a=s.index('    def _nav_ocr_worker(')
b=s.index('    def _poll_navigation_ocr(',a)
s=s[:a]+s[b:]
old='''        self.result_socket.close()
        if self.timing_stream is not None:
'''
new='''        self.result_socket.close()
        if self.nav_process is not None:
            try:
                self.nav_request.put_nowait(None)
            except queue.Full:
                pass
            self.nav_process.join(timeout=2.0)
            if self.nav_process.is_alive():
                self.nav_process.terminate()
                self.nav_process.join(timeout=1.0)
            self.nav_process = None
        if self.timing_stream is not None:
'''
if old not in s: raise SystemExit('close marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('removed thread, added process close')

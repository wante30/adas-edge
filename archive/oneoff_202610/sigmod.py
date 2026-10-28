from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
s=s.replace('import resource\nimport socket','import resource\nimport signal\nimport socket')
old='''    bridge = AiReplayBridge(args)
    bridge.run()
'''
new='''    bridge = AiReplayBridge(args)

    def _terminate(signum, _frame):
        try:
            bridge.close()
        finally:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    bridge.run()
'''
if old not in s: raise SystemExit('main marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('signal handler added')

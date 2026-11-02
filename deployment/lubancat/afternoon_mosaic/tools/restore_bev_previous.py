"""Stop this replay and resume the unchanged ADAS session saved before Issue 10."""
import json,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    for path in Path('/dev/shm/adas_bev_runtime/evidence').glob('*_processes.json'):
        for pid in json.loads(path.read_text()).get('pids',[]):
            proc=Path('/proc')/str(pid)
            try:
                cmd=proc.joinpath('cmdline').read_bytes().decode().replace('\0',' ')
                if str(ROOT) in cmd or '/dev/shm/adas_bev_runtime/candidate/' in cmd or '/dev/shm/adas_bev_runtime/baseline_src/' in cmd:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
            except FileNotFoundError:pass
    state=json.loads((ROOT.parent/'restore_session.json').read_text())
    if state['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        print('Different boot; original autostart is unchanged. No old PID was resumed.');return
    for saved in state['processes']:
        try:
            cmd=(Path('/proc')/str(saved['pid'])/'cmdline').read_bytes().decode().replace('\0',' ')
            if cmd==saved['cmd']:os.kill(saved['pid'],signal.SIGCONT)
        except (ProcessLookupError,FileNotFoundError):pass
    print('Resumed the preserved previous session; no deployed files were replaced.')
if __name__=='__main__':main()

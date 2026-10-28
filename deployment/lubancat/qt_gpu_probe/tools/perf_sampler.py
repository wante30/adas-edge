import argparse, json, time
from collections import defaultdict
import psutil

p = argparse.ArgumentParser()
p.add_argument('--duration', type=float, default=35)
p.add_argument('--out', required=True)
a = p.parse_args()

roles = defaultdict(lambda: {'cpu': [], 'rss': [], 'threads': []})
cores = []
samples = []
end = time.monotonic() + a.duration
while time.monotonic() < end:
    t0 = time.monotonic()
    per_core = psutil.cpu_percent(interval=None, percpu=True)
    cores.append(per_core)
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_times', 'memory_info', 'num_threads']):
        try:
            cmd = ' '.join(proc.info.get('cmdline') or ())
            if 'pgv_local_publisher.py' in cmd:
                role = 'publisher'
            elif 'ai_bridge.py' in cmd:
                role = 'ai'
            elif 'spawn_main' in cmd:
                role = 'ocr'
            elif 'tesla_scene_renderer.py' in cmd:
                role = 'scene'
            elif 'qt_gpu_probe' in cmd:
                role = 'qt'
            else:
                continue
            times = proc.info['cpu_times']
            total = times.user + times.system
            # delta CPU percent since previous sample
            key = (role, proc.info['pid'])
            samples.append((t0, key, total, proc.info['memory_info'].rss, proc.info['num_threads']))
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue
    time.sleep(max(0.0, 0.5 - (time.monotonic() - t0)))

# aggregate deltas by role
by_pid = {}
for t, key, total, rss, threads in samples:
    by_pid.setdefault(key, []).append((t, total, rss, threads))
summary = defaultdict(lambda: {'cpu_percent': [], 'rss_mb': [], 'threads': []})
for (role, pid), rows in by_pid.items():
    for (t0, c0, r0, n0), (t1, c1, r1, n1) in zip(rows, rows[1:]):
        dt = t1 - t0
        if dt > 0:
            summary[role]['cpu_percent'].append(max(0.0, (c1-c0)/dt*100.0))
            summary[role]['rss_mb'].append(r1/1024/1024)
            summary[role]['threads'].append(n1)

def agg(values):
    if not values:
        return None
    values = sorted(values)
    return {'n': len(values), 'mean': sum(values)/len(values),
            'p50': values[len(values)//2], 'p95': values[min(len(values)-1, int(len(values)*.95))],
            'max': values[-1]}

out = {'duration_s': a.duration,
       'cores_mean_percent': [sum(row[i] for row in cores)/len(cores) for i in range(len(cores[0]))] if cores else [],
       'roles': {role: {'cpu_percent': agg(rows['cpu_percent']),
                        'rss_mb': agg(rows['rss_mb']),
                        'threads': agg(rows['threads'])} for role, rows in summary.items()}}
with open(a.out, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

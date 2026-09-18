#!/usr/bin/env python3
"""Lightweight board telemetry sampler (stdlib only, Python 3.8).

Polls SoC sensors and an optional tracked process every ``--interval`` seconds
and appends one CSV row per sample. Runs alongside a benchmark so thermal or
frequency throttling can be ruled out (or confirmed) after the fact.

Sampled: thermal zones, per-policy CPU frequency, NPU/DDR/VDEC devfreq
cur_freq, overall CPU busy percent (from /proc/stat deltas) and the RSS of
``--pid`` when given.
"""
import argparse
import csv
import glob
import sys
import time


def read_int(path):
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def cpu_busy_percent(prev_stat):
    with open("/proc/stat") as handle:
        fields = handle.readline().split()
    values = [int(v) for v in fields[1:8]]  # user nice system idle iowait irq softirq
    total = sum(values)
    idle = values[3] + values[4]
    prev_total, prev_idle = prev_stat
    busy_pct = None
    if total > prev_total:
        busy_pct = 100.0 * (1.0 - (idle - prev_idle) / float(total - prev_total))
    return (total, idle), busy_pct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--duration", type=float, default=1260.0)
    parser.add_argument("--pid", type=int, default=None, help="process whose RSS to track")
    args = parser.parse_args()

    zones = sorted(glob.glob("/sys/class/thermal/thermal_zone*"))
    zone_names = {}
    for zone in zones:
        try:
            with open(zone + "/type") as handle:
                zone_names[zone] = handle.read().strip()
        except OSError:
            zone_names[zone] = zone.rsplit("/", 1)[-1]

    cpu_paths = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))
    devfreq = {
        "npu_mhz": "/sys/class/devfreq/fde40000.npu/cur_freq",
        "ddr_mhz": "/sys/class/devfreq/dmc/cur_freq",
        "vdec_mhz": "/sys/class/devfreq/fdf80200.rkvdec/cur_freq",
        "gpu_mhz": "/sys/class/devfreq/fde60000.gpu/cur_freq",
    }

    stat = (0, 0)
    started = time.time()
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["t_s", "cpu_busy_pct"]
        header += ["%s_C" % zone_names[z] for z in zones]
        header += ["cpu%d_mhz" % i for i in range(len(cpu_paths))]
        header += list(devfreq)
        if args.pid:
            header += ["tracked_rss_mb"]
        writer.writerow(header)

        while time.time() - started < args.duration:
            row = [round(time.time() - started, 1)]
            stat, busy = cpu_busy_percent(stat)
            row.append(round(busy, 1) if busy is not None else "")
            row += [read_int(z + "/temp") for z in zones]  # milli-Celsius
            row += [read_int(p) for p in cpu_paths]        # kHz
            row += [read_int(p) for p in devfreq.values()]  # Hz
            if args.pid:
                rss_kb = read_int("/proc/%d/status" % args.pid) if False else None
                try:
                    with open("/proc/%d/status" % args.pid) as status:
                        for line in status:
                            if line.startswith("VmRSS:"):
                                rss_kb = int(line.split()[1])
                                break
                except OSError:
                    rss_kb = None
                row.append(round(rss_kb / 1024.0, 1) if rss_kb else "")
            writer.writerow(row)
            handle.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())

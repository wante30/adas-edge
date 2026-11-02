"""Click the actual Qt model menu; record each applied backend and fresh results."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
LABEL = os.environ.get("ADAS_VALIDATION_LABEL", "stage2_menu")

def metric():
    with (EVIDENCE / (LABEL + "_hmi.jsonl")).open("rb") as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 12000))
        return json.loads(f.read().splitlines()[-1])

def wait(predicate, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            row = metric()
            if predicate(row): return row
        except (FileNotFoundError, IndexError): pass
        time.sleep(.2)
    raise RuntimeError("Menu acceptance timeout: " + json.dumps(metric()))

def click(x, y):
    subprocess.run(["python3", str(ROOT / "tools/ui_control.py"), str(x), str(y)], check=True)

def main():
    events = []
    wait(lambda m: m["ai_online"] and m["ai_results"] > 10)
    for index, (model, mode, y) in enumerate([
        ("B", "ROAD_Q1", 172),
        ("MTL-FULL", "ROAD_Q1", 260),
        ("FAST-MOSAIC", "MOSAIC_FULL", 348),
        ("MTL-FAST", "ROAD_Q1", 304),
        ("FAST-MOSAIC", "MOSAIC_FULL", 348),
        ("B-capi", "ROAD_Q1", 216),
        ("MTL-FAST", "ROAD_Q1", 304),
    ]):
        before = metric()["ai_results"]
        click(1200, 40); time.sleep(.3); click(1000, y)
        row = wait(lambda m: m["model_id"] == model and m["inference_mode"] == mode
                   and m["inference_ready"] and m["ai_online"] and m["ai_results"] > before + 20)
        # Require real post-selection completions from this actual backend/epoch.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            row = metric()
            with (EVIDENCE / (LABEL + "_ai.jsonl")).open("rb") as stream:
                stream.seek(0, 2); stream.seek(max(0, stream.tell() - 300000))
                data = stream.read().splitlines()
            completions = []
            parsed = []
            for raw in data:
                try: item = json.loads(raw)
                except ValueError: continue
                parsed.append(item)
                if (item.get("model_id") == model and item["inference"]["mode"] == mode
                        and int(item["inference"]["epoch"]) == int(row.get("inference_epoch", -1))):
                    completions.append(item)
            # Epoch is recorded in AI; metrics from older builds omit it.
            if "inference_epoch" not in row:
                completions = parsed
                completions = [item for item in completions if item.get("model_id") == model and item["inference"]["mode"] == mode]
                latest_epoch = completions[-1]["inference"]["epoch"] if completions else None
                completions = [item for item in completions if item["inference"]["epoch"] == latest_epoch]
            if len(completions) >= 12: break
            time.sleep(.2)
        assert len(completions) >= 12, "Missing fresh actual model completions"
        row["actual_completions"] = len(completions)
        row["actual_last_result"] = {k:completions[-1].get(k) for k in ("model_id", "inference", "main_timing", "pipeline_ms")}
        row["selection"] = model; events.append(row)
        subprocess.run(["xwd", "-display", ":0", "-name","Qt Quick GPU Probe - RK3566", "-silent", "-out",
                        str(EVIDENCE / (LABEL + "_menu_" + str(index) + ".xwd"))],
                       env=dict(os.environ, XAUTHORITY="/home/cat/.Xauthority"), check=True)
        print(json.dumps(row), flush=True)
    state = json.loads((EVIDENCE / "integration_session.json").read_text())
    report = {"passed": True, "input": state.get("input_kind", "LOCAL_REPLAY"),
              "control_mode": state.get("control_mode", "DRY_RUN"), "events": events}
    (EVIDENCE / (LABEL + "_model_menu_acceptance.json")).write_text(json.dumps(report, indent=2))
    print("MODEL_MENU_ACCEPTANCE_COMPLETE", flush=True)

if __name__ == "__main__": main()

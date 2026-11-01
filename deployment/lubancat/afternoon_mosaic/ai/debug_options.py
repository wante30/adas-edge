import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
BRIDGE_DIR = Path("/home/cat/qt_gpu_probe_ai_20261019")
os.environ["ADAS_EDGE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BRIDGE_DIR))
spec = importlib.util.spec_from_file_location("ai_dbg", BRIDGE_DIR / "ai_bridge.py")
ai = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ai; spec.loader.exec_module(ai)
comp_spec = importlib.util.spec_from_file_location("comp_dbg", BRIDGE_DIR / "b05p_competition_bridge.py")
comp = importlib.util.module_from_spec(comp_spec); sys.modules[comp_spec.name] = comp; comp_spec.loader.exec_module(comp)
comp._install_raw_head_adapter(ai, "B")
argv = ["--source", "SYNTHETIC", "--layout", "quad", "--display-view", "front",
        "--road-quadrant", "3", "--navigation-quadrant", "1",
        "--main-model", "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn",
        "--main-hz", "10", "--hazard-hz", "5", "--specialist-hz", "1",
        "--request-port", "39561", "--result-port", "39562", "--idle-timeout", "60",
        "--file", "/home/cat/b05n_mtl_b2a_bench_20261031/replay60/frame_000.rgb565"]
args = ai.parser().parse_args(argv)
b = ai.AiReplayBridge(args)
print("profiles", len(b.model_profiles), [p.id for p in b.model_profiles.values()])
print("has method", hasattr(b, "_model_selection_status"))
st = b._model_selection_status()
print("options", len(st["options"]), [(x.get("id"), x.get("available")) for x in st["options"]])
b.close()

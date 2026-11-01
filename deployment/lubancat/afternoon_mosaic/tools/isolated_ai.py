"""Run the original brake decision logic against a UDP simulator in tests."""
import os
from pathlib import Path
import runpy
import sys

root = Path(os.environ["ADAS_EDGE_ROOT"])
sys.path.insert(0, str(root))
import auto_brake

original_init = auto_brake.AutoBrakeController.__init__


def isolated_init(self, **kwargs):
    kwargs["port"] = int(os.environ.get("ADAS_TEST_ACTION_PORT", "38362"))
    original_init(self, **kwargs)


auto_brake.AutoBrakeController.__init__ = isolated_init
runpy.run_path(str(root / "b05p_ai_bridge_launcher.py"), run_name="__main__")

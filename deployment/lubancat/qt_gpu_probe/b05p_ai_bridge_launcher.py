"""Launch the live AI bridge with the frozen B05P raw-head adapter."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


BRIDGE_DIR = Path(__file__).resolve().parent
ADAPTER_PATH = BRIDGE_DIR / "b05p_competition_bridge.py"


def main() -> None:
    root = os.environ.get("ADAS_EDGE_ROOT")
    if not root:
        raise RuntimeError("ADAS_EDGE_ROOT must point to the competition bridge package")
    sys.path.insert(0, root)
    import ai_bridge

    spec = importlib.util.spec_from_file_location("b05p_live_adapter", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load B05P adapter: {ADAPTER_PATH}")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    adapter._install_raw_head_adapter(ai_bridge, "B")
    ai_bridge.main()


if __name__ == "__main__":
    main()

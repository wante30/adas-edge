import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('signal_bench', Path(__file__).parents[1]/'tools/board_multi_signal_ab.py')
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


@pytest.mark.parametrize('argv', [b'/old/build/qt_gpu_probe\0--fps\060',
    b'/usr/bin/python3\0-u\0/old/src/tools/isolated_ai.py\0',
    b'/usr/bin/python3\0/old/ai/b05p_ai_bridge_launcher.py\0'])
def test_live_bench_is_rejected_before_any_run_or_process_changes(tmp_path, argv):
    process = tmp_path/'123'
    process.mkdir()
    (process/'cmdline').write_bytes(argv)
    with pytest.raises(RuntimeError, match='Competing HMI/AI'):
        BENCH.require_idle_bench(tmp_path)
    assert (process/'cmdline').read_bytes() == argv


def test_idle_bench_does_not_modify_other_processes(tmp_path):
    process = tmp_path/'124'
    process.mkdir()
    (process/'cmdline').write_bytes(b'/usr/bin/ssh\0')
    BENCH.require_idle_bench(tmp_path)
    assert (process/'cmdline').read_bytes() == b'/usr/bin/ssh\0'

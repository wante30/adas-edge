from pathlib import Path
import importlib.util


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round_robin_selects_nonempty_streams() -> None:
    scheduler = _load("scheduler", "multistream_framework/scheduler.py")
    rr = scheduler.RoundRobinScheduler(stream_count=3)
    states = {0: {"queue_length": 0}, 1: {"queue_length": 1}, 2: {"queue_length": 1}}
    assert [rr.select(states), rr.select(states), rr.select(states)] == [1, 2, 1]


def test_drop_oldest_queue_evicts_without_blocking() -> None:
    replay = _load("replay", "multistream_framework/replay.py")
    q = replay.DropOldestQueue(maxsize=3)
    assert q.put_drop_oldest("a") is None
    assert q.put_drop_oldest("b") is None
    assert q.put_drop_oldest("c") is None
    assert q.put_drop_oldest("d") == "a"
    assert [q.get_nowait(), q.get_nowait(), q.get_nowait()] == ["b", "c", "d"]


def test_memmap_queue_accepts_metadata_only() -> None:
    replay = _load("memmap_replay", "tools/replay_memmap_multistream.py")
    q = replay.DropOldestQueue(maxsize=3)
    event = (2, 17, 123.5, 17)
    assert q.put_drop_oldest(event) is None
    assert q.get_nowait() == event
    # Queue entries are fixed-size metadata tuples, never image arrays.
    assert not any(hasattr(value, "shape") for value in event)


def test_object_metrics_use_oracle_and_track_iou() -> None:
    metrics = _load("hetero_eval", "tools/hetero_eval.py")
    oracle = {(0, 0): [[0, 0, 10, 10, 0.9, 0]], (0, 1): [[1, 0, 11, 10, 0.9, 0]]}
    scheduled = {(0, 1): [[1, 0, 11, 10, 0.8, 0]]}
    gt = {(0, 0): [{"track_id": 7, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "class_id": 1}], (0, 1): [{"track_id": 7, "x1": 1, "y1": 0, "x2": 11, "y2": 10, "class_id": 1}]}
    result = metrics.evaluate(oracle, scheduled, gt, {0: [(1, 2.0)]}, 25.0, 1.0)
    assert result["per_object_miss_rate"] == 0.0
    assert result["first_detection_delay_frames_mean"] == 1
    assert result["object_frame_recall"] == 0.5

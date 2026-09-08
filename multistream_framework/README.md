# Multistream Replay Framework

This directory provides a measurement-only multi-stream replay harness for the
RK3566 RKNN runtime.

- `scheduler.py` defines `Scheduler.select(stream_states)` and implements only
  `RoundRobinScheduler`.
- `replay.py` starts one timestamp-paced decode thread per stream, a bounded
  depth-3 drop-oldest queue per stream, one serial RKNN worker, and one
  postprocess thread per stream.
- The decode loop follows each input video's FPS using `time.perf_counter()`;
  a full queue evicts the oldest frame and never blocks the producer.

Example on the board:

```bash
python3 multistream_framework/replay.py \
  --video traffic_unannotated_source.mp4 \
  --model yolo11n_320_raw_int8.rknn \
  --streams 2 4 6 8 --duration 60 \
  --output-dir multistream_output
```

Each configuration writes a per-frame CSV with
`stream_id,frame_id,arrival_ts,selected_ts,infer_done_ts,dropped`, a resource
sampling CSV, a per-stream summary CSV and a markdown summary. No content
analysis, multi-NPU execution or scheduler other than round-robin is included.

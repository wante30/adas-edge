from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
needle='''    report["source_to_upload_ms_last"] = video->sourceToUploadMs();'''
rep='''    report["source_to_upload_ms_last"] = video->sourceToUploadMs();
    report["source_to_upload_p50_ms"] = video->sourceToUploadP50Ms();
    report["source_to_upload_p95_ms"] = video->sourceToUploadP95Ms();
    report["display_queue_dropped_on_take"] = static_cast<qint64>(video->queueDroppedOnTake());
    report["display_queue_overwritten"] = static_cast<qint64>(video->queueOverwritten());
    report["display_queue_max_depth"] = static_cast<qint64>(video->queueMaxDepth());
    report["display_queue_last_take_depth"] = static_cast<qint64>(video->queueLastTakeDepth());'''
if needle not in s: raise SystemExit('report marker')
s=s.replace(needle,rep,1)
p.write_text(s,encoding='utf-8')
print('main report fields')

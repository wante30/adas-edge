"""Create performance/epoch summaries from unmodified archived board evidence."""
import collections,csv,hashlib,io,json,statistics,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'validation/issue10'
WINDOWS=[(10,28),(33,38),(44,48),(54,68),(74,88)]

def lines(z,name):return [json.loads(line) for line in z.read(name).decode().splitlines() if line.strip()]

def main():
    report={}
    with zipfile.ZipFile(ROOT/'raw_evidence.zip') as z:
        for label in ['perf_baseline','perf_final','mtl_baseline','mtl_final']:
            hmi=lines(z,label+'_hmi.jsonl');ai=lines(z,label+'_ai.jsonl');groups=collections.defaultdict(lambda:{'weighted_qt':0,'duration':0,'ai_intervals':0,'ai_seconds':0,'hmi_cpu':[]})
            for left,right in ([(10,50)] if label.startswith('mtl_') else WINDOWS):
                rs=[r for r in hmi if left<r['process_elapsed_ms']/1000<right and r['ai_online'] and r['inference_ready']]
                key=rs[-1]['model_id'];g=groups[key]
                for a,b in zip(rs,rs[1:]):
                    dt=(b['process_elapsed_ms']-a['process_elapsed_ms'])/1000;g['duration']+=dt;g['weighted_qt']+=a['scene_graph_fps']*dt
                g['hmi_cpu'] += [r['cpu_percent_of_all_cores'] for r in rs]
                selected=[r for r in ai if r['model_id']==key and rs[0]['ai_frame_id']<=r['frame_id']<=rs[-1]['ai_frame_id']]
                for a,b in zip(selected,selected[1:]):
                    if a['inference']['epoch']==b['inference']['epoch']:
                        dt=(b['infer_end_ns']-a['infer_end_ns'])/1e9
                        if dt>0:g['ai_intervals']+=1;g['ai_seconds']+=dt
            summary={key:{'qt_time_weighted_fps':g['weighted_qt']/g['duration'],'ai_completion_fps':g['ai_intervals']/g['ai_seconds'],'hmi_cpu_all_four_cores_percent':statistics.mean(g['hmi_cpu']),'window_seconds':g['duration']} for key,g in groups.items()}
            cpu=json.loads(z.read(label+'_cpu.json'));summary['whole_system_cpu_all_four_cores_percent']=statistics.mean(r['cpu_all_four_cores_percent'] for r in cpu if 10<r['elapsed_s']<(50 if label.startswith('mtl_') else 1000))
            report[label]=summary
        measurements=lines(z,'baseline_q4_v1_measurements.jsonl');ai=lines(z,'baseline_q4_v1_ai.jsonl')
        epoch_by_time={(r['capture_timestamp_ns'],r['model_id']):r['inference']['epoch'] for r in ai}
        lookup=collections.defaultdict(set)
        for row in measurements:
            for b in row['boxes']:
                key=(b.get('sourceId',''),b['trackId'],tuple(round(b[k],6) for k in ('x','y','width','height')))
                lookup[key].add(int(b['detectTimestampNs']))
        output=io.StringIO();writer=csv.writer(output);writer.writerow(['variant','original_capture_timestamp_ns','replay_capture_timestamp_ns','serverTrackId','sourceId','original_ai_epoch','replay_epoch','bbox_x','bbox_y','bbox_width','bbox_height','bev_raw_x','bev_raw_z','bev_filtered_x','bev_filtered_z','birth_count','visual_frame_interval_s','measurement_age_s'])
        offsets={}
        for label in ['exact_baseline','exact_final']:
            rows=lines(z,label+'_trace.jsonl');counts=collections.Counter()
            for r in rows:
                key=(r['source_id'],r['track_id'],tuple(round(v,6) for v in r['bbox']))
                if len(lookup[key])==1:counts[r['capture_timestamp_ns']-next(iter(lookup[key]))]+=1
            offset,n=counts.most_common(1)[0];assert len(counts)==1,counts
            offsets[label]={'replay_timestamp_offset_ns':offset,'unique_box_matches':n}
            for r in rows:
                original=r['capture_timestamp_ns']-offset;model=r['source_key'].split(':')[-1]
                epoch=epoch_by_time.get((original,model),'UNAVAILABLE')
                writer.writerow([label,original,r['capture_timestamp_ns'],r['server_track_id'],r['source_id'],epoch,r['epoch'],*r['bbox'],r['raw_x'],r['raw_z'],r['filtered_x'],r['filtered_z'],r['birth_count'],r['visual_dt'],r['measurement_age_s']])
        report['timestamp_alignment']=offsets
    (ROOT/'performance_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'trajectory_csv.zip','w',zipfile.ZIP_DEFLATED) as archive:archive.writestr('bev_trajectories.csv',output.getvalue().encode('utf-8'))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

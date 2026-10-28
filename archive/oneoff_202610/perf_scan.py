import json,glob,os
roots=[r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\results',r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\mali_video\results']
rows=[]
for root in roots:
 for p in glob.glob(root+r'\*.json'):
  try: d=json.load(open(p,encoding='utf-8'))
  except Exception: continue
  keys=['unique_source_fps_average','unique_frame_id_upload_fps_average','unique_source_submit_fps','unique_replay_content_fps_average','consumed_fps_last_window','input_fps_average','actual_input_fps','ai_status_last']
  vals={k:d.get(k) for k in keys if k in d}
  if vals:
   rows.append((os.path.basename(p),vals))
for n,v in rows:
 print(n,v)

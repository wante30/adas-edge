from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\competition_ai\hazard_events.py')
s=p.read_text(encoding='utf-8')
old='''            if trusted and len(history) >= 6:
                lateral, lateral_fit = trend(history, 1, corrected=True)'''
new='''            if (trusted and track.name == 'person' and info['in_assumed_path']
                    and track.box[3] / height > .60):
                # Image-space path presence is a real detection-backed
                # reminder, not a collision/TTC estimate.
                events.append({'kind': 'PEDESTRIAN_IN_PATH', 'track_id': track.track_id,
                               'bbox': track.box, 'source': 'IMAGE_SPACE_PATH_PRESENCE'})
            if trusted and len(history) >= 6:
                lateral, lateral_fit = trend(history, 1, corrected=True)'''
if old not in s: raise SystemExit('marker')
p.write_text(s.replace(old,new),encoding='utf-8')
print('patched')

from pathlib import Path
import json, cv2, numpy as np

ROOT = Path(r'D:\Datasets\adas_edge_training\b05h_auto_annotation_20261024')
P = ROOT / 'pseudo'
rows = json.loads((ROOT / 'meta' / 'pseudo_labels.json').read_text(encoding='utf-8'))
B = [r for r in rows if r['review'] == 'B']

def split(r):
    return 'test' if r['video'].startswith('video_20260929_113421') else ('val' if r['video'].startswith('video_20260929_112739') else 'train')

for page in range(0, len(B), 15):
    items = B[page:page+15]
    canvas = np.full((3*300, 5*300, 3), 25, np.uint8)
    for j, r in enumerate(items):
        img = cv2.imread(str(P / 'images' / split(r) / r['image']))
        if img is None:
            continue
        x1,y1,x2,y2 = map(int, r['bbox_xyxy'])
        cx, cy = (x1+x2)//2, (y1+y2)//2
        half = max(60, int(2.2*max(x2-x1, y2-y1)))
        a1,b1 = max(0,cx-half), max(0,cy-half)
        a2,b2 = min(img.shape[1],cx+half), min(img.shape[0],cy+half)
        crop = img[b1:b2,a1:a2].copy()
        crop = cv2.resize(crop,(300,300),interpolation=cv2.INTER_LINEAR)
        # draw relative bbox
        sx = 300/(a2-a1); sy=300/(b2-b1)
        cv2.rectangle(crop,(int((x1-a1)*sx),int((y1-b1)*sy)),(int((x2-a1)*sx),int((y2-b1)*sy)),(0,255,255),2)
        cv2.putText(crop, f'B{page+j} {r["color"]} {r["confidence"]:.2f}',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
        row,col = divmod(j,5)
        canvas[row*300:(row+1)*300,col*300:(col+1)*300]=crop
    cv2.imwrite(str(ROOT/'meta'/f'B_review_{page//15+1}.jpg'), canvas)
print('pages', (len(B)+14)//15)

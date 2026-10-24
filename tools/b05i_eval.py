import json
from pathlib import Path
from ultralytics import YOLO

def main():
    models={
     'source': r'D:\Datasets\adas_edge_training\tl_specialist\tl-512x192\weights\best.pt',
     'baseline': r'D:\Datasets\adas_edge_training\b05i_runs\baseline\weights\best.pt',
     'candidate': r'D:\Datasets\adas_edge_training\b05i_runs\candidate_clean\weights\best.pt',
    }
    out={}
    for name,path in models.items():
        m=YOLO(path)
        r=m.val(data=r'D:\Datasets\adas_edge_training\b05i_train_20261024\bdd50.yaml', imgsz=512, rect=True, batch=16, device=0, workers=0, conf=0.25, iou=0.5, plots=False, verbose=False)
        out[name]={'metrics': {k: float(v) for k,v in r.results_dict.items()}, 'speed': r.speed}
    print(json.dumps(out, indent=2))
    Path(r'C:\Users\JingyuWang\adas-edge\outputs\b05i_eval.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()




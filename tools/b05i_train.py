import sys
from ultralytics import YOLO

def main():
    name=sys.argv[1]; data=sys.argv[2]
    m=YOLO(r"D:\Datasets\adas_edge_training\tl_specialist\tl-512x192\weights\best.pt")
    r=m.train(data=data, epochs=1, imgsz=512, rect=True, batch=16, device=0,
              workers=2, project=r"D:\Datasets\adas_edge_training\b05i_runs",
              name=name, seed=0, exist_ok=True, plots=True, verbose=True)
    print("DONE", r.save_dir, r.results_dict)

if __name__ == "__main__":
    main()

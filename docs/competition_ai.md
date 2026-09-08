# Competition AI implementation

`competition_ai.CompetitionDetector` supports the generated 320/416 split-output RKNN models and historical six-output raw models. It performs letterbox preprocessing, RKNN Lite inference, DFL/raw or split decode, confidence filtering, class-wise NMS and source-coordinate restoration. Results carry stream/frame/timestamp metadata and remain independent of PCIe and MOT17.

The COCO checkpoint supports person, bicycle, car, motorcycle, bus, truck, traffic light and stop sign among the competition focus classes. `competition_ai.traffic_light` adds a conservative HSV ROI adapter with `RED`, `YELLOW`, `GREEN` and `UNKNOWN`; it never invents a color when the ROI is too small, dim, low-saturation, ambiguous or outside the frame. Lane type, zebra crossing and road-obstacle semantics remain scene-adapter responsibilities. No fine-tuning is currently justified.

The detector and scene adapters are exercised by unit tests and a software replay tool. A replay result is evidence for the Python pipeline only; it is not a claim of FPGA/PCIe or road-video acceptance until the real capture path and approved clips are available.

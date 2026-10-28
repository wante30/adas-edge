import numpy as np
from competition_ai import CompetitionDetector, Detection, FrameResult

def test_serialization_and_metadata():
    d=Detection(2,'car',.8,1,2,3,4); r=FrameResult('front',7,99,1.5,[d])
    assert d.to_dict()['class_name']=='car'; assert r.to_dict()['stream_id']=='front'; assert r.to_dict()['frame_id']==7; assert r.to_dict()['timestamp_ns']==99

def test_preprocess_shapes_320_416():
    det=object.__new__(CompetitionDetector)
    frame=np.zeros((480,640,3),dtype=np.uint8)
    for n in (320,416):
        det.input_size=n; image,scale,px,py=det.preprocess(frame); assert image.shape==(n,n,3); assert image.dtype==np.uint8; assert scale>0; assert px>=0 and py>=0

def test_model_class_profiles_keep_explicit_adapter_arguments_compatible():
    det=CompetitionDetector("model.rknn", runner=object(), class_profile="bdd10")
    assert det.class_names[0] == "pedestrian"
    assert det.class_names[-1] == "traffic sign"
    custom=CompetitionDetector("model.rknn", runner=object(), class_names=("vehicle",))
    assert custom.class_names == ("vehicle",)

def test_empty_and_multiple_detections_and_mapping():
    class Runner:
        def inference(self,**kwargs): return []
    det=object.__new__(CompetitionDetector); det.input_size=320; det.conf_threshold=.25; det.nms_threshold=.45; det._runner=Runner()
    det.preprocess=lambda f:(f,1.0,0,0)
    import competition_ai.detector as mod
    old=mod.decode_yolo11_raw if hasattr(mod,'decode_yolo11_raw') else None
    # Exercise metadata path with an empty decoder by patching the imported module.
    import tools.profile_rknn_single_stream as prof
    saved=prof.decode_yolo11_raw; prof.decode_yolo11_raw=lambda *a: []
    try:
        assert det.infer(np.zeros((2,2,3),dtype=np.uint8),2,3,4).detections==[]
        assert set(det.last_timing) == {
            "preprocess_ms", "rknn_call_ms", "decode_nms_ms", "total_ms",
            "output_count", "detection_count", "preprocess_backend",
        }
    finally: prof.decode_yolo11_raw=saved

def test_split_decode_multiple_boxes_coordinate_restore():
    class Runner:
        def inference(self, **kwargs): return [np.zeros((1,4,2),np.float32), np.array([[[.9,.8]]],np.float32)]
    det=object.__new__(CompetitionDetector); det.input_size=320; det.conf_threshold=.25; det.nms_threshold=.45; det._runner=Runner()
    det.preprocess=lambda f:(f,2.0,10,20)
    import deployment.rknn.decode_yolo_output as dec
    saved=dec.decode_outputs; dec.decode_outputs=lambda *a: [{'bbox':[10,20,30,40],'confidence':.9,'class_id':0},{'bbox':[20,30,40,50],'confidence':.8,'class_id':2}]
    try:
        r=det.infer(np.zeros((2,2,3),np.uint8),stream_id='s',frame_id=4,timestamp_ns=5)
        assert [d.class_name for d in r.detections]==['person','car']; assert r.detections[0].x1==0 and r.detections[0].y1==0
    finally: dec.decode_outputs=saved

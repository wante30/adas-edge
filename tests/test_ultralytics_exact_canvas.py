from types import SimpleNamespace
import numpy as np

from adas_edge.contracts import Frame, PixelFormat
from adas_edge.perception.ultralytics_detector import UltralyticsDetector


def test_exact_canvas_keeps_bgr_for_predict_and_restores_source_box():
    seen={}
    def predict(**kwargs):
        seen.update(kwargs)
        box=SimpleNamespace(cls=np.array([9]),conf=np.array([.9]),
                            xyxy=np.array([[160,48,256,96]],dtype=float))
        return [SimpleNamespace(boxes=[box],names={9:'traffic light'})]
    d=object.__new__(UltralyticsDetector)
    d._model=SimpleNamespace(predict=predict);d._confidence=.25;d._device='cpu'
    d._input_width,d._input_height=512,192;d._input_size=(192,512)
    d._exact_input_canvas=True;d._allowed_classes={'traffic light'}
    source=np.full((200,400,3),(10,20,30),np.uint8)
    frame=Frame('front',0,None,source,PixelFormat.BGR888)
    result=d.detect(frame)
    assert seen['source'].shape==(192,512,3)
    assert seen['source'][96,256].tolist()==[10,20,30]
    assert seen['rect'] is False
    assert np.allclose(result[0].bbox_xyxy,(100.,50.,200.,100.))

import torch
import pytest
from experiments.training.fpga_road_seg.models import build_model
from experiments.training.fpga_road_seg.engine import masked_loss, Metrics
from experiments.training.fpga_road_seg.profile import profile_model

torch.set_num_threads(4)


def test_missing_channel_and_pixel_have_zero_gradient():
    x=torch.zeros((1,4,8,8),requires_grad=True)
    y=torch.ones_like(x); v=torch.ones_like(x); v[:,3]=0; v[:,:,0]=0
    masked_loss(x,y,v,[1,5,5,3]).backward()
    assert x.grad[:,3].abs().sum()==0 and x.grad[:,:,0].abs().sum()==0
    assert x.grad[:,1,1:].abs().sum()>0
    z=torch.zeros_like(v)
    assert masked_loss(x,y,z).item()==0


@pytest.mark.parametrize('name,channels',[('twin-small',2),('twin-nano',2),('twin-four',4),('candidate',4),('candidate-plus',4)])
def test_real_forward_backward(name,channels):
    model=build_model(name)
    y=model(torch.rand(2,3,192,320))
    assert y.shape==(2,channels,192,320)
    loss=masked_loss(y,torch.zeros_like(y),torch.ones_like(y))
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_original_binary_adapter_matches_softmax():
    model=build_model('twin-small').eval(); x=torch.rand(1,3,192,320)
    with torch.no_grad():
        da,ll=model.network(x)
        adapted=model(x).sigmoid()
    torch.testing.assert_close(adapted[:,0],da.softmax(1)[:,1])
    torch.testing.assert_close(adapted[:,1],ll.softmax(1)[:,1])


def test_original_384x216_fails_explicitly():
    with pytest.raises(RuntimeError):
        build_model('twin-small').eval()(torch.zeros(1,3,216,384))
    assert build_model('candidate').eval()(torch.zeros(1,3,216,384)).shape==(1,4,216,384)


def test_metrics_ignore_unlabelled_channel_and_no_positive_recall():
    m=Metrics(2); p=torch.ones(1,2,2,2); y=torch.ones_like(p); v=torch.ones_like(p); v[:,1]=0
    m.update(p,y,v)
    r=m.report(['known','missing'])
    assert r['known']['iou']==1 and r['missing']['iou'] is None and r['missing']['recall'] is None


def test_profile_counts_tiny_conv_exactly():
    r=profile_model(torch.nn.Conv2d(3,4,3,bias=False),(1,3,8,8))
    assert r['parameters']==108 and r['macs']==4*6*6*3*3*3


def test_candidate_parameter_targets():
    small = build_model('candidate')
    plus = build_model('candidate-plus')
    assert sum(p.numel() for p in small.parameters()) == 8508
    assert 30000 <= sum(p.numel() for p in plus.parameters()) <= 80000

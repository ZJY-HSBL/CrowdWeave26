import pytest
torch = pytest.importorskip("torch")
from crowdweave.research import SCADDownsample, gotp_loss


def test_scad_output_and_backward():
    layer = SCADDownsample(8, 16)
    x = torch.randn(2, 8, 33, 35, requires_grad=True)
    out, gate = layer(x)
    assert out.shape == (2, 16, 17, 18)
    assert gate.shape == (2, 1, 17, 18)
    assert gate.min() >= 0 and gate.max() <= 1
    out.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_gotp_tolerates_real_overlap_and_propagates_gradient():
    gt = torch.tensor([[0., 0., 10., 10.], [5., 0., 15., 10.]])
    p = gt.clone().requires_grad_(True)
    assert gotp_loss(p, gt).item() == pytest.approx(0.)
    over = torch.tensor([[0., 0., 10., 10.], [1., 0., 11., 10.]], requires_grad=True)
    loss = gotp_loss(over, gt)
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(over.grad).all()


def test_singleton_and_shape_rejection():
    gt = torch.tensor([[0., 0., 10., 10.]])
    assert gotp_loss(gt, gt).item() == 0
    with pytest.raises(ValueError):
        gotp_loss(gt, gt[:, :3])

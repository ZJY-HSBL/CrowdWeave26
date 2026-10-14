"""Tests for standalone research components; not detection benchmark scores."""
import pytest
torch=pytest.importorskip("torch")
from crowdweave26 import (SCADDownsample, CFSRNeck, gotp_pairwise_loss,
                          conflict_supervision_loss, rasterize_overlap)


def test_scad_shape_and_gradient():
    module=SCADDownsample(8,16)
    x=torch.randn(2,8,33,35,requires_grad=True)
    y,logits=module(x,return_conflict_logits=True)
    assert y.shape==(2,16,17,18) and logits.shape==(2,1,17,18)
    y.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_scad_invalid_channels():
    with pytest.raises(ValueError):
        SCADDownsample(0,8)


def test_cfsr_fusion_gradient():
    neck=CFSRNeck((8,16,32),24)
    inputs=(torch.randn(2,8,32,32,requires_grad=True),
            torch.randn(2,16,16,16,requires_grad=True),
            torch.randn(2,32,8,8,requires_grad=True))
    output,logits,weights=neck(inputs,return_aux=True)
    assert output.shape==(2,24,16,16)
    assert logits.shape==(2,1,16,16)
    assert torch.allclose(weights.sum(1),torch.ones_like(weights[:,0]),atol=1e-6)
    output.mean().backward()
    assert all(t.grad is not None for t in inputs)


def test_raster_overlap():
    boxes=torch.tensor([[0.,0.,8.,8.],[4.,4.,12.,12.]])
    overlap=rasterize_overlap(boxes,(16,16),(16,16))
    assert overlap.sum().item()==16


def test_empty_raster():
    assert rasterize_overlap(torch.zeros((0,4)),(16,16),(8,8)).sum().item()==0


def test_gotp_true_overlap_zero():
    gt=torch.tensor([[0.,0.,10.,10.],[5.,0.,15.,10.]])
    pred=gt.clone().requires_grad_(True)
    assert gotp_pairwise_loss(pred,gt).item()==pytest.approx(0.)


def test_gotp_excess_overlap_gradient():
    gt=torch.tensor([[0.,0.,10.,10.],[5.,0.,15.,10.]])
    pred=torch.tensor([[0.,0.,10.,10.],[1.,0.,11.,10.]],requires_grad=True)
    loss=gotp_pairwise_loss(pred,gt)
    assert loss.item()>0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_aux_loss_backprop():
    logits=torch.randn(2,1,4,4,requires_grad=True)
    target=torch.zeros_like(logits)
    loss=conflict_supervision_loss(logits,target)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

"""Unique GT selection and differentiable GOTP target tests."""
import torch
import pytest
from crowdweave26.gotp import representative_pair_loss


def _example():
    gt = torch.tensor([[[0.,0.,5.,10.],[0.,0.,5.,10.],
                        [6.,0.,11.,10.],[6.,0.,11.,10.]]])
    pred = torch.tensor([[[0.,0.,8.,10.],[0.,0.,8.,10.],
                          [3.,0.,11.,10.],[3.,0.,11.,10.]]], requires_grad=True)
    fg = torch.tensor([[True,True,True,True]])
    ids = torch.tensor([[0,0,1,1]])
    quality = torch.tensor([[[.8],[.3],[.9],[.2]]])
    return pred,gt,fg,ids,quality


def test_unique_gt_pairs_produce_nonzero_loss_and_gradients():
    pred,gt,fg,idx,q=_example()
    loss=representative_pair_loss(pred,gt,fg,idx,q)
    assert loss > 0
    loss.backward()
    assert pred.grad.abs().sum() > 0
    assert pred.grad[0,1].abs().sum() == 0
    assert pred.grad[0,3].abs().sum() == 0


def test_single_person_many_anchors_has_no_false_collision():
    pred,gt,fg,idx,q=_example()
    idx[:]=0
    loss=representative_pair_loss(pred,gt,fg,idx,q)
    assert loss == 0


def test_gt_overlap_tolerance_eliminates_penalty_when_prediction_matches():
    pred,gt,fg,idx,q=_example()
    same=gt.clone().requires_grad_()
    loss=representative_pair_loss(same,gt,fg,idx,q)
    assert loss.item() == pytest.approx(0,abs=1e-8)
    loss.backward()
    assert same.grad is not None


def test_empty_and_unmatched_outputs_are_zero_with_gradient_path():
    pred,gt,fg,idx,q=_example()
    fg[:]=False
    loss=representative_pair_loss(pred,gt,fg,idx,q)
    assert loss.item()==0
    loss.backward()
    assert pred.grad is not None


def test_batch_isolation():
    pred,gt,fg,idx,q=_example()
    batch_pred=torch.cat([pred.detach(),pred.detach()],0).requires_grad_()
    batch_gt=torch.cat([gt,gt],0)
    batch_fg=torch.cat([fg,torch.zeros_like(fg)],0)
    loss=representative_pair_loss(batch_pred,batch_gt,batch_fg,
                                   torch.cat([idx,idx]),torch.cat([q,q]))
    assert loss > 0
    loss.backward()
    assert batch_pred.grad[1].abs().sum() == 0


def test_malformed_assignment_dimensions_rejected():
    pred,gt,fg,idx,q=_example()
    with pytest.raises(ValueError):
        representative_pair_loss(pred,gt,fg,idx[:,:-1],q)

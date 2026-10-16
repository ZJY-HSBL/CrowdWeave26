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



def test_max_gt_subsampling_is_stable_and_finite():
    pred,gt,fg,idx,q=_example()
    loss=representative_pair_loss(pred,gt,fg,idx,q,max_gt=2)
    assert torch.isfinite(loss)



def test_pair_counters_report_violation_instead_of_just_zero_loss():
    pred,gt,fg,idx,q=_example()
    loss, numbers=representative_pair_loss(
        pred,gt,fg,idx,q,return_stats=True
    )
    assert torch.allclose(loss,representative_pair_loss(pred,gt,fg,idx,q))
    # Unique GT=2, one eligible neighboring pair, one violation, one image.
    assert numbers.tolist() == [2.,1.,1.,1.]


def test_zero_loss_distinguishes_satisfied_pair_from_absent_pair():
    pred,gt,fg,idx,q=_example()
    same = gt.clone()
    zero, stats = representative_pair_loss(
        same,gt,fg,idx,q,return_stats=True
    )
    assert zero == 0 and stats.tolist() == [2.,1.,0.,1.]
    fg.zero_()
    no_pairs, absent = representative_pair_loss(
        same,gt,fg,idx,q,return_stats=True
    )
    assert no_pairs == 0 and absent.tolist() == [0.,0.,0.,0.]


def test_empty_minibatch_statistics_preserve_zero_gradient():
    pred=torch.empty(0,0,4,requires_grad=True)
    gt=pred.detach()
    loss, stats=representative_pair_loss(
        pred,gt,torch.zeros(0,0,dtype=torch.bool),
        torch.zeros(0,0,dtype=torch.long),
        torch.zeros(0,0,1),return_stats=True
    )
    assert stats.tolist() == [0.,0.,0.,0.]
    assert loss.item() == 0.
    loss.backward()
    assert pred.grad is not None

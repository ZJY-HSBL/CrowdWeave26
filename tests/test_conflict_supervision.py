"""Standalone overlap target and gradient tests."""
import pytest
import torch
from crowdweave26.supervision import overlap_targets, conflict_bce


def batch(boxes, inds=None, bs=1):
    if inds is None:
        inds = [0] * len(boxes)
    return {
        "img": torch.zeros(bs, 3, 32, 32),
        "batch_idx": torch.tensor(inds, dtype=torch.float32),
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    }


def test_no_overlap():
    assert overlap_targets(batch([[.2,.2,.2,.2],[.8,.8,.2,.2]]), (8,8)).sum() == 0


def test_overlap_map_shape_and_location():
    m = overlap_targets(batch([[.5,.5,.5,.5],[.5,.5,.5,.5]]), (8,8))
    assert m.shape == (1,1,8,8)
    assert m[0,0,4,4] == 1 and m[0,0,0,0] == 0


def test_independent_batch_images():
    m = overlap_targets(batch([[.5,.5,.5,.5]] * 3, [0,0,1], 2), (8,8))
    assert m[0].sum() > 0 and m[1].sum() == 0


def test_augmented_boundary_is_clipped():
    assert overlap_targets(batch([[.9,.5,.4,.8],[.9,.5,.4,.8]]), (8,8)).sum() > 0


def test_empty_batch():
    assert overlap_targets(batch([]), (7,11)).sum() == 0


def test_aux_loss_gradient():
    logits = torch.zeros(1,1,8,8,requires_grad=True)
    loss = conflict_bce(logits, batch([[.5,.5,.5,.5],[.5,.5,.5,.5]]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,0,4,4] < 0 and logits.grad[0,0,0,0] > 0


def test_bad_indices_fail():
    with pytest.raises(ValueError):
        overlap_targets(batch([[.5,.5,.5,.5]], [4]), (8,8))


def test_wrong_batch_size_fails():
    with pytest.raises(ValueError):
        conflict_bce(torch.zeros(2,1,8,8), batch([]))

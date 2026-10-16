"""The synthetic microbenchmark reports measurements, not AP claims."""
import math
import torch
from scripts.benchmark_gotp import run_case, synthetic_batch


def test_synthetic_crowd_geometry_and_output_contract():
    pred, gt, fg, idx, q = synthetic_batch(4,2,torch.device("cpu"))
    assert pred.shape == gt.shape == (1,8,4)
    assert fg.shape == idx.shape == (1,8)
    assert q.shape == (1,8,1)
    assert torch.unique(idx).numel()==4
    result=run_case(4,2,2,0,torch.device("cpu"))
    assert result["total_positive_anchors"]==8
    assert result["mean_close_pairs"]>0
    assert result["mean_step_ms"]>=0
    assert 0 <= result["mean_violating_fraction"] <= 1
    assert result["nonzero_gradient_step_fraction"] > 0


def test_microbenchmark_rejects_unusable_population():
    import pytest
    with pytest.raises(ValueError):
        synthetic_batch(1,2,torch.device("cpu"))

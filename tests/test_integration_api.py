"""Pinned-upstream contract tests, skipped when Ultralytics is unavailable."""
import pytest
pytest.importorskip("ultralytics", reason="Pinned YOLO26 is optional in local unit environment")
from scripts.smoke_integration import check


@pytest.mark.parametrize("mode", ("none", "scad", "cfsr", "both"))
def test_official_yolo26_contract(mode):
    result = check(mode)
    assert result["grad_ok"]
    assert result["params"] > 0

def test_real_yolo26_training_criterion_has_finite_gradients():
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from crowdweave26.integration import install_crowdweave

    model = DetectionModel("yolo26n-p2.yaml", nc=1, verbose=False)
    install_crowdweave(model, "both")
    model.args = get_cfg()
    model.train()
    batch = {
        "img": torch.rand(1, 3, 96, 96),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.30, 0.50]]),
    }
    loss, items = model(batch)
    assert bool(torch.isfinite(loss).all())
    loss.sum().backward()
    assert model.model[3].detail[0].weight.grad is not None
    assert model.model[-1].crowd_cfsr_gain.grad is not None

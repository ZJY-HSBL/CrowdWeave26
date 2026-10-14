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



def test_supervised_conflict_loss_backpropagates_to_cfsr_logits():
    import torch
    from ultralytics.cfg import get_cfg
    from crowdweave26.integration import CrowdSupervisedDetectionModel, install_crowdweave
    model = CrowdSupervisedDetectionModel("yolo26n-p2.yaml", nc=1, verbose=False)
    install_crowdweave(model, "both")
    model.args = get_cfg()
    model.crowd_conflict_weight = 0.25
    model.train()
    batch = {
        "img": torch.rand(1, 3, 96, 96),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.50, 0.50, 0.45, 0.60], [0.55, 0.52, 0.45, 0.60]]),
    }
    loss, items = model(batch)
    assert loss.numel() == 4 and "conflict_loss" in items
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert model.model[-1].crowd_cfsr.conflict.weight.grad is not None
    assert model.model[-1].crowd_cfsr.conflict.weight.grad.abs().sum() > 0
    assert hasattr(model.criterion, "update")
    model.criterion.update()


def test_cra_real_tal_recovers_collided_person():
    import torch
    from ultralytics.utils.tal import TaskAlignedAssigner
    from crowdweave26.cra import upgrade_assigner
    base = TaskAlignedAssigner(topk=2, topk2=1, num_classes=1)
    cra = upgrade_assigner(base)
    candidates = torch.tensor([[[1, 1, 0], [1, 0, 0]]], dtype=torch.int8)
    quality = torch.tensor([[[0.90, 0.50, 0.0], [0.80, 0.0, 0.0]]])
    baseline = base.select_highest_overlaps(candidates.clone(), quality.clone(), 2, quality.clone())
    revised = cra.select_highest_overlaps(candidates.clone(), quality.clone(), 2, quality.clone())
    assert int(baseline[1].sum()) == 1
    assert int(revised[1].sum()) == 2
    assert revised[2][0, 0, 1] == 1 and revised[2][0, 1, 0] == 1
    assert cra.last_cra_stats["recovered"] == 1


def test_cra_only_real_yolo26_loss_and_gradient():
    import torch
    from ultralytics.cfg import get_cfg
    from crowdweave26.integration import CrowdSupervisedDetectionModel
    from crowdweave26.cra import CrowdReservedAssigner
    model = CrowdSupervisedDetectionModel("yolo26n-p2.yaml", nc=1, verbose=False)
    model.args = get_cfg()
    model.crowd_cra_enabled = True
    model.train()
    batch = {
        "img": torch.rand(1, 3, 96, 96),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.50, 0.50, 0.40, 0.60], [0.52, 0.50, 0.40, 0.60]]),
    }
    loss, items = model(batch)
    assert loss.numel() == 3
    assert torch.isfinite(loss).all()
    assert isinstance(model.criterion.one2many.assigner, CrowdReservedAssigner)
    assert isinstance(model.criterion.one2one.assigner, CrowdReservedAssigner)
    assert len(items) == 3
    loss.sum().backward()
    assert model.model[-1].cv2[0][0].conv.weight.grad is not None


def test_cra_and_cfsr_can_train_together():
    import torch
    from ultralytics.cfg import get_cfg
    from crowdweave26.integration import CrowdSupervisedDetectionModel, install_crowdweave
    model = CrowdSupervisedDetectionModel("yolo26n-p2.yaml", nc=1, verbose=False)
    install_crowdweave(model, "both")
    model.args = get_cfg()
    model.crowd_cra_enabled = True
    model.crowd_conflict_weight = 0.25
    model.train()
    batch = {
        "img": torch.rand(1, 3, 96, 96),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.50, 0.50, 0.45, 0.60], [0.55, 0.52, 0.45, 0.60]]),
    }
    loss, items = model(batch)
    assert loss.numel() == 4 and "conflict_loss" in items
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert model.model[-1].crowd_cfsr.conflict.weight.grad is not None


def test_trainer_constructs_cra_graph(monkeypatch):
    from ultralytics.cfg import get_cfg
    from crowdweave26.integration import CrowdWeaveTrainer, ENV_MODE, ENV_CRA_ENABLED
    monkeypatch.setenv(ENV_MODE, "none")
    monkeypatch.setenv(ENV_CRA_ENABLED, "1")
    trainer = object.__new__(CrowdWeaveTrainer)
    trainer.data = {"nc": 1, "channels": 3, "names": {0: "person"}}
    trainer.args = get_cfg()
    model = trainer.get_model(cfg="yolo26n-p2.yaml", verbose=False)
    assert model.crowd_cra_enabled is True
    assert model.model[-1].nl == 4


def test_cfsr_validation_uses_zero_auxiliary_loss():
    import torch
    from ultralytics.cfg import get_cfg
    from crowdweave26.integration import CrowdSupervisedDetectionModel, install_crowdweave
    model = CrowdSupervisedDetectionModel("yolo26n-p2.yaml", nc=1, verbose=False)
    install_crowdweave(model, "both")
    model.args = get_cfg()
    model.crowd_conflict_weight = 0.25
    model.crowd_cra_enabled = True
    model.train()
    batch = {
        "img": torch.rand(1, 3, 96, 96),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.50, 0.50, 0.45, 0.60], [0.55, 0.52, 0.45, 0.60]]),
    }
    model.eval()
    with torch.no_grad():
        preds = model(batch["img"])
        assert isinstance(preds, tuple)
        loss, items = model.loss(batch, preds)
    assert loss.numel() == 4 and torch.isfinite(loss).all()
    assert "conflict_loss" in items and items["conflict_loss"] == 0



def test_cra_counts_and_validation_exclusion():
    import torch
    from ultralytics.cfg import get_cfg
    from crowdweave26.integration import CrowdSupervisedDetectionModel
    model = CrowdSupervisedDetectionModel("yolo26n-p2.yaml", nc=1, verbose=False)
    model.args = get_cfg()
    model.crowd_cra_enabled = True
    batch = {
        "img": torch.rand(1, 3, 96, 96),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, .4, .6], [.52, .5, .4, .6]]),
    }
    model.train()
    model(batch)
    assigner = model.criterion.one2many.assigner
    before = dict(assigner.cumulative_cra_stats)
    assert before["calls"] > 0
    model.eval()
    with torch.no_grad():
        model.loss(batch, model(batch["img"]))
    assert assigner.cumulative_cra_stats == before

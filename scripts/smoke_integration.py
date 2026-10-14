"""Real pinned-YOLO26 forward/backward graph contract test."""
from __future__ import annotations

import argparse
import torch
from ultralytics.nn.tasks import DetectionModel
from crowdweave26.integration import install_crowdweave


def check(mode: str, imgsz: int = 96) -> dict:
    torch.manual_seed(123)
    model = DetectionModel("yolo26n-p2.yaml", nc=1, verbose=False)
    install_crowdweave(model, mode)
    model.train()
    outputs = model(torch.randn(1, 3, imgsz, imgsz))
    expected = {"one2many", "one2one"}
    if mode in ("cfsr", "both"):
        expected.add("crowd_conflict_logits")
        logits = outputs["crowd_conflict_logits"]
        assert logits.shape == (1, 1, imgsz // 8, imgsz // 8)
        assert torch.isfinite(logits).all()
    assert set(outputs) == expected
    feats = outputs["one2many"]["feats"]
    assert len(feats) == 4
    assert [int(f.shape[-1]) for f in feats] == [imgsz // s for s in (4, 8, 16, 32)]
    objective = sum(
        outputs[k]["boxes"].square().mean() + outputs[k]["scores"].square().mean()
        for k in ("one2many", "one2one")
    )
    assert bool(torch.isfinite(objective))
    objective.backward()
    if mode in ("scad", "both"):
        assert model.model[3].detail[0].weight.grad is not None
        assert bool(torch.isfinite(model.model[3].detail[0].weight.grad).all())
    if mode in ("cfsr", "both"):
        assert model.model[-1].crowd_cfsr_gain.grad is not None
        assert model.model[-1].crowd_cfsr.projections[0][0].weight.grad is not None
    return {"mode": mode, "params": sum(p.numel() for p in model.parameters()), "grad_ok": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("none", "scad", "cfsr", "both"), default="both")
    p.add_argument("--imgsz", type=int, default=96)
    args = p.parse_args()
    print(check(args.mode, args.imgsz))

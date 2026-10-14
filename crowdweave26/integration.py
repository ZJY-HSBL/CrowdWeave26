"""Opt-in SCAD/CFSR integration for the pinned official YOLO26-P2 graph.

This modifies no vendored Ultralytics source. It preserves the original
P2/P3/P4/P5 Detect outputs and the One-to-Many/One-to-One loss contract.
"""
from __future__ import annotations

import os

import torch
from torch import nn
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.head import Detect

from .modules import CFSRNeck, SCADDownsample

MODES = ("none", "scad", "cfsr", "both")
ENV_MODE = "CROWDWEAVE26_MODE"


class CrowdDetect(Detect):
    """Keep both original detector branches; refine P3 using P2/P3/P4."""

    def forward(self, x: list[torch.Tensor]):
        if len(x) != 4:
            raise ValueError("CFSR requires YOLO26 P2/P3/P4/P5 inputs")
        update = self.crowd_cfsr((x[0], x[1], x[2]))
        x = list(x)
        x[1] = x[1] + self.crowd_cfsr_gain * update
        return super().forward(x)


def install_crowdweave(model: nn.Module, mode: str = "both") -> nn.Module:
    """Install in a freshly built official YOLO26-P2 model, never twice."""
    if mode not in MODES:
        raise ValueError(f"Unknown mechanism mode: {mode}")
    if mode == "none":
        return model
    layers = getattr(model, "model", None)
    if not isinstance(layers, nn.Sequential) or len(layers) != 30:
        raise ValueError("Expected pinned YOLO26 P2 graph of 30 layers")
    original, head = layers[3], layers[-1]
    if not isinstance(head, Detect) or head.nl != 4 or list(head.f) != [19, 22, 25, 28]:
        raise ValueError("Expected YOLO26 P2 Detect inputs 19/22/25/28")
    if isinstance(original, SCADDownsample) or isinstance(head, CrowdDetect):
        raise ValueError("Mechanisms already installed")
    if mode in ("scad", "both"):
        conv, bn = getattr(original, "conv", None), getattr(original, "bn", None)
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            raise ValueError("Layer 3 must be official Conv+BN")
        if conv.kernel_size != (3, 3) or conv.stride != (2, 2):
            raise ValueError("Layer 3 requires stride-2 3x3 Conv")
        new = SCADDownsample(conv.in_channels, conv.out_channels).to(
            device=conv.weight.device, dtype=conv.weight.dtype
        )
        with torch.no_grad():
            new.detail[0].weight.copy_(conv.weight)
            new.detail[1].load_state_dict(bn.state_dict())
        new.i, new.f, new.type = original.i, original.f, "SCADDownsample"
        new.np = sum(p.numel() for p in new.parameters())
        layers[3] = new
    if mode in ("cfsr", "both"):
        channels = [head.cv2[i][0].conv.in_channels for i in range(head.nl)]
        weight = head.cv2[0][0].conv.weight
        head.__class__ = CrowdDetect  # Original trained Detect towers are kept.
        head.crowd_cfsr = CFSRNeck(tuple(channels[:3]), channels[1]).to(
            device=weight.device, dtype=weight.dtype
        )
        head.crowd_cfsr_gain = nn.Parameter(weight.new_tensor(0.05))
        head.type = "CrowdDetect"
        head.np = sum(p.numel() for p in head.parameters())
    return model


class CrowdWeaveTrainer(DetectionTrainer):
    """Install custom graph BEFORE loading pretrained weights.

    The environment variable makes the mode visible to Ultralytics DDP
    workers; multi-GPU behavior must still be independently validated.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        mode = os.environ.get(ENV_MODE, "both")
        install_crowdweave(model, mode)
        if weights is not None:
            model.load(weights)
            if mode in ("scad", "both"):
                source = getattr(weights, "model", [None] * 4)[3]
                conv, bn = getattr(source, "conv", None), getattr(source, "bn", None)
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    with torch.no_grad():
                        model.model[3].detail[0].weight.copy_(conv.weight)
                        model.model[3].detail[1].load_state_dict(bn.state_dict())
        return model

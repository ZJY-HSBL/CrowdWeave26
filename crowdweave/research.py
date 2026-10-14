"""SCAD/GOTP research PROTOTYPES: not integrated with Ultralytics training.

These prototypes do not establish novelty or benchmark improvements.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class SCADDownsample(nn.Module):
    """Local gated blend of blurred and detail-preserving stride-2 paths.

    Future: supervise spatial gate using a GT-defined conflict field. The gate
    currently learns only from local features without special crowd labels.
    """

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        if min(channels_in, channels_out) < 1:
            raise ValueError("Channel counts must be positive")
        self.blur = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.detail = nn.Conv2d(channels_in, channels_in, kernel_size=3, stride=2,
                                padding=1, groups=channels_in, bias=False)
        self.gate = nn.Conv2d(channels_in, 1, kernel_size=3, stride=2, padding=1)
        self.project = nn.Sequential(nn.Conv2d(channels_in, channels_out, 1, bias=False),
                                     nn.BatchNorm2d(channels_out), nn.SiLU())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        alpha = self.gate(x).sigmoid()
        blended = alpha * self.detail(x) + (1 - alpha) * self.blur(x)
        return self.project(blended), alpha


def pairwise_iou(boxes: Tensor, other: Tensor) -> Tensor:
    """Pairwise xyxy box IoU, differentiable in predicted box coordinates."""
    lt = torch.maximum(boxes[:, None, :2], other[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], other[None, :, 2:])
    intersection = (rb - lt).clamp(min=0).prod(dim=-1)
    area_a = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=-1)
    area_b = (other[:, 2:] - other[:, :2]).clamp(min=0).prod(dim=-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=1e-9)


def gotp_loss(pred_xyxy: Tensor, gt_xyxy: Tensor, margin: float = 0.05) -> Tensor:
    """Penalize predicted inter-instance overlap above GT overlap plus tolerance.

    Each predicted row must match the corresponding GT *identity*. O(N^2)
    reference only; production loss needs sparse pairs and robust assignment.
    """
    if pred_xyxy.shape != gt_xyxy.shape or pred_xyxy.ndim != 2 or pred_xyxy.shape[1] != 4:
        raise ValueError("pred_xyxy and gt_xyxy must both have shape (N, 4)")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    n = pred_xyxy.shape[0]
    if n < 2:
        return pred_xyxy.sum() * 0.0
    i, j = torch.triu_indices(n, n, offset=1, device=pred_xyxy.device)
    genuine = pairwise_iou(gt_xyxy.detach(), gt_xyxy.detach())[i, j]
    predicted = pairwise_iou(pred_xyxy, pred_xyxy)[i, j]
    return (predicted - genuine - margin).clamp(min=0).mean()

"""Live augmented YOLO batch supervision for dense person box overlaps.

Targets represent >=2 non-ignored, trainable full-body rectangles per feature cell,
NOT semantic crowd density or visible-area occlusion. Ignore masks are NOT applied.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .conflict import rasterize_overlap


def overlap_targets(batch: dict, output_hw: tuple[int, int], *, device=None) -> torch.Tensor:
    """Compute [B,1,H,W] overlap map from current augmented normalized xywh."""
    if "img" not in batch or batch["img"].ndim != 4:
        raise ValueError("batch[img] must be BCHW")
    bs = int(batch["img"].shape[0])
    h, w = (int(output_hw[0]), int(output_hw[1]))
    if min(h, w) < 1:
        raise ValueError("output dimensions must be positive")
    device = device if device is not None else batch["img"].device
    inds = batch["batch_idx"].reshape(-1).to(device)
    boxes = batch["bboxes"].reshape(-1, 4).to(device).float()
    if inds.numel() != boxes.shape[0]:
        raise ValueError("batch_idx/box mismatch")
    if inds.numel() and (not bool(torch.isfinite(inds).all()) or
                          not bool((inds == inds.round()).all()) or
                          not bool(((inds >= 0) & (inds < bs)).all())):
        raise ValueError("invalid batch indices")
    boxes = torch.nan_to_num(boxes, nan=-1, posinf=-1, neginf=-1)
    valid = (boxes[:, 2:] > 0).all(1) & (boxes[:, :2] >= 0).all(1) & (boxes[:, :2] <= 1).all(1)
    out = []
    for i in range(bs):
        local = boxes[valid & (inds == i)]
        if local.shape[0]:
            xyxy = torch.stack((local[:, 0] - local[:, 2] * 0.5,
                                local[:, 1] - local[:, 3] * 0.5,
                                local[:, 0] + local[:, 2] * 0.5,
                                local[:, 1] + local[:, 3] * 0.5), -1)
            xyxy[:, [0, 2]] *= w
            xyxy[:, [1, 3]] *= h
            mask = rasterize_overlap(xyxy, (h, w), (h, w))
        else:
            mask = torch.zeros((h, w), device=device, dtype=torch.float32)
        out.append(mask)
    return torch.stack(out)[:, None]


def conflict_bce(logits: torch.Tensor, batch: dict, *, max_pos_weight: float = 20.0) -> torch.Tensor:
    """Positive-weighted BCE with bounded amplification for rare overlaps."""
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape B,1,H,W")
    target = overlap_targets(batch, logits.shape[-2:], device=logits.device).to(logits.dtype)
    if target.shape != logits.shape:
        raise ValueError("logits and batch shape differ")
    pos = target.sum()
    neg = target.numel() - pos
    weight = (neg / pos.clamp_min(1)).clamp(min=1, max=max_pos_weight).detach()
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)

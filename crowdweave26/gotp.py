"""GT-Overlap-Tolerant Pairwise (GOTP) YOLO26 training criterion.

One representative positive anchor per unique (image, GT) is selected by
the detached TAL target score. Comparisons therefore never treat two
positives from the SAME pedestrian as distinct objects.

The auxiliary penalty is a research hypothesis, NOT an accuracy claim.
"""
from __future__ import annotations

import torch
from .losses import gotp_pairwise_loss


def representative_pair_loss(
    pred_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    fg_mask: torch.Tensor,
    target_gt_idx: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    margin: float = 0.03,
    max_gt: int = 128,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Compute per-image mean over unique-GT matched predictions.

    Args:
        pred_xyxy, gt_xyxy: [B,A,4] boxes in SAME pixel coordinate system.
        fg_mask, target_gt_idx: [B,A] TAL assignment mask and local GT index.
        target_scores: [B,A,C] TAL target scores for deterministic selection.
        max_gt: cap representative count by highest detached confidence.
    """
    if pred_xyxy.ndim != 3 or pred_xyxy.shape[-1] != 4 or gt_xyxy.shape != pred_xyxy.shape:
        raise ValueError("Pred/GT tensors must both be B,A,4")
    bs, anchors, _ = pred_xyxy.shape
    if fg_mask.shape != (bs, anchors) or target_gt_idx.shape != (bs, anchors):
        raise ValueError("TAL foreground/GT indices incompatible with predicted boxes")
    if target_scores.ndim != 3 or target_scores.shape[:2] != (bs, anchors):
        raise ValueError("TAL target score dimensions incompatible")
    if margin < 0 or max_gt < 2 or max_pairs < 1:
        raise ValueError("Require margin >= 0, max_gt >= 2, max_pairs >= 1")
    values = []
    for b in range(bs):
        available = fg_mask[b].bool().nonzero(as_tuple=False).flatten()
        if available.numel() < 2:
            values.append(pred_xyxy[b].sum() * 0.0)
            continue
        gt_index = target_gt_idx[b, available].long()
        if bool((gt_index < 0).any()):
            raise ValueError("Matched TAL GT index must be nonnegative")
        quality = target_scores[b, available].detach().float().sum(-1)
        representatives = []
        # Unique people, not positive anchors. Ties select the first anchor.
        for gt in torch.unique(gt_index, sorted=True):
            owned = gt_index == gt
            anchor = available[owned][torch.argmax(quality[owned])]
            representatives.append(anchor)
        if len(representatives) < 2:
            values.append(pred_xyxy[b].sum() * 0.0)
            continue
        selected = torch.stack(representatives)
        if selected.numel() > max_gt:
            q = target_scores[b, selected].detach().float().sum(-1)
            # torch.topk is deterministic for unequal scores; this is
            # explicit subset selection, not a claim to preserve all GTs.
            selected = selected[q.topk(max_gt).indices]
        values.append(gotp_pairwise_loss(
            pred_xyxy[b, selected], gt_xyxy[b, selected],
            margin=margin, max_pairs=max_pairs, only_close_gt=True,
        ))
    return torch.stack(values).mean() if values else pred_xyxy.sum() * 0.0


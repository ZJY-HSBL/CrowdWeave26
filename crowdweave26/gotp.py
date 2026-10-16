"""GT-Overlap-Tolerant Pairwise (GOTP) YOLO26 training criterion.

One representative positive anchor per unique (image, GT) is selected by
the detached TAL target score. Comparisons therefore never treat two
positives from the SAME pedestrian as distinct objects.

The auxiliary penalty is a research hypothesis, NOT an accuracy claim.
"""
from __future__ import annotations

import torch
from ultralytics.utils.loss import v8DetectionLoss
from .ignore_loss import IgnoreAwareDetectionLoss
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


class GOTPAwareDetectionLoss(IgnoreAwareDetectionLoss):
    """Existing official classification/box/DFL + independently logged GOTP."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.ignore_aware = bool(getattr(model, "crowd_ignore_aware", False))
        self.gotp_weight = float(getattr(model, "crowd_gotp_weight", 0))
        self.gotp_margin = float(getattr(model, "crowd_gotp_margin", 0.03))
        self.gotp_max_gt = int(getattr(model, "crowd_gotp_max_gt", 128))
        if not 0 <= self.gotp_weight <= 10:
            raise ValueError("GOTP weight must lie in [0,10]")
        if not 0 <= self.gotp_margin <= 1:
            raise ValueError("GOTP margin must lie in [0,1]")

    def get_assigned_targets_and_loss(self, preds, batch):
        if self.ignore_aware:
            return IgnoreAwareDetectionLoss.get_assigned_targets_and_loss(self, preds, batch)
        return v8DetectionLoss.get_assigned_targets_and_loss(self, preds, batch)

    def loss(self, preds, batch):
        matches, base_loss, items = self.get_assigned_targets_and_loss(preds, batch)
        fg_mask, idx, target_bboxes, anchor_points, stride_tensor = matches
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        decoded = self.bbox_decode(anchor_points, pred_distri)
        predicted_pixels = decoded * stride_tensor
        extra = representative_pair_loss(
            predicted_pixels, target_bboxes, fg_mask, idx,
            # TAL target scores aren't exposed by the official return
            # contract; recompute quality from target scores by re-assigner?
            # Instead use differentiable-free IoU of assigned GT and
            # detached decoded boxes as representative selection quality.
            _representative_quality(predicted_pixels.detach(), target_bboxes, fg_mask),
            margin=self.gotp_margin, max_gt=self.gotp_max_gt
        ) * self.gotp_weight
        loss = torch.cat((base_loss, extra.reshape(1)))
        return loss * preds["boxes"].shape[0], {
            **items, "gotp_loss": extra.detach()
        }


def _representative_quality(pred, gt, fg_mask):
    """GT IoU is a stable fallback quality for selecting one TAL-positive anchor."""
    from .losses import _pair_iou
    b, anchors, _ = pred.shape
    quality = _pair_iou(pred.reshape(-1, 4), gt.reshape(-1, 4)).reshape(b, anchors)
    return (quality * fg_mask).unsqueeze(-1)

"""Pinned YOLO26 GOTP criterion, isolated from standalone research primitives."""
from __future__ import annotations
import torch
from ultralytics.utils.loss import v8DetectionLoss
from .ignore_loss import IgnoreAwareDetectionLoss
from .gotp import representative_pair_loss

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

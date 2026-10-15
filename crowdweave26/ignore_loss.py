"""Ignore-aware extension of the pinned YOLO26 classification loss.

The only deliberate difference from the official detection criterion is that
unassigned anchors inside transformed ignored regions cannot act as background
classification negatives. Positive TAL targets and box/DFL losses are retained.
"""
from __future__ import annotations
import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from .ignore import ignored_anchor_mask


class IgnoreAwareDetectionLoss(v8DetectionLoss):
    """Pinned-upstream v8DetectionLoss with background BCE ignore masking."""

    def get_assigned_targets_and_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous()
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1),
                             batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_(min=1)
        bce = self.bce(pred_scores, target_scores.to(dtype))
        # A matched real person remains positive even when its anchor center
        # falls inside another object's ignore region.
        ignored = ignored_anchor_mask(batch, anchor_points, stride_tensor,
                                       device=self.device)
        suppress = ignored & ~fg_mask
        bce = bce.masked_fill(suppress.unsqueeze(-1), 0)
        if self.class_weights is not None:
            bce *= self.class_weights
        loss[1] = bce.sum() / target_scores_sum
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points,
            target_bboxes / stride_tensor, target_scores, target_scores_sum,
            fg_mask, imgsz, stride_tensor,
        )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )

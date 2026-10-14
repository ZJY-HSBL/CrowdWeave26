"""Standalone differentiable auxiliary loss research prototypes."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def conflict_supervision_loss(logits: torch.Tensor, overlap_target: torch.Tensor,
                              *, pos_weight: float = 1.0) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape != overlap_target.shape:
        raise ValueError("logits and target must match Bx1xHxW")
    if pos_weight <= 0:
        raise ValueError("pos_weight must be positive")
    return F.binary_cross_entropy_with_logits(
        logits, overlap_target.to(dtype=logits.dtype),
        pos_weight=torch.as_tensor(pos_weight,dtype=logits.dtype,device=logits.device))


def _pair_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lo = torch.maximum(a[:,:2],b[:,:2])
    hi = torch.minimum(a[:,2:],b[:,2:])
    inter = (hi-lo).clamp_min(0).prod(-1)
    area_a = (a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    area_b = (b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return inter/(area_a+area_b-inter).clamp_min(1e-8)


def gotp_pairwise_loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, *,
                       margin: float = 0.03, max_pairs: int = 10000,
                       only_close_gt: bool = True) -> torch.Tensor:
    """Penalize predicted overlap exceeding actual GT overlap + margin.

    Requires predictions already uniquely assigned to DIFFERENT ground-truth
    persons. This function does not perform matching. No accuracy is claimed.
    """
    if pred_xyxy.ndim != 2 or pred_xyxy.shape[1] != 4 or gt_xyxy.shape != pred_xyxy.shape:
        raise ValueError("pred and GT must be corresponding Nx4 xyxy")
    if margin < 0 or max_pairs < 1:
        raise ValueError("margin>=0 and max_pairs>=1 required")
    n = pred_xyxy.shape[0]
    if n < 2:
        return pred_xyxy.sum()*0.
    gt = gt_xyxy.detach().to(device=pred_xyxy.device,dtype=pred_xyxy.dtype)
    a,b = torch.triu_indices(n,n,1,device=pred_xyxy.device)
    if only_close_gt:
        centers=(gt[:,:2]+gt[:,2:])*.5
        sides=(gt[:,2:]-gt[:,:2]).clamp_min(0)
        length=torch.maximum(sides[a].amax(1),sides[b].amax(1))
        close=(centers[a]-centers[b]).norm(dim=1)<=1.5*length
        a,b=a[close],b[close]
    if a.numel() == 0:
        return pred_xyxy.sum()*0.
    a,b=a[:max_pairs],b[:max_pairs]
    return F.relu(_pair_iou(pred_xyxy[a],pred_xyxy[b]) -
                  _pair_iou(gt[a],gt[b]) - margin).mean()

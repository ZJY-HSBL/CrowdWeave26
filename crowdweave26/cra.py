"""Opt-in Collision-Reserved Assignment (CRA) for pinned YOLO26 TAL.

Only modifies positive assignment during training. The CPU sparse alternating
path step may add synchronization overhead on CUDA; benchmark throughput.
"""
from __future__ import annotations
import math
import torch
from ultralytics.utils.tal import TaskAlignedAssigner
from .assignment import repair_coverage


class CrowdReservedAssigner(TaskAlignedAssigner):
    """Recover feasible zero-positive GTs without stealing a donor's last positive."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_cra_stats = True
        self.cumulative_cra_stats = {
            "calls": 0, "orphans_before": 0, "orphans_after": 0,
            "recovered": 0, "changed_anchors": 0,
        }

    def _record_cra_stats(self):
        if not self.collect_cra_stats:
            return
        self.cumulative_cra_stats["calls"] += 1
        for key in ("orphans_before", "orphans_after", "recovered", "changed_anchors"):
            self.cumulative_cra_stats[key] += self.last_cra_stats[key]

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        candidate_mask = mask_pos.bool()
        gt_idx, fg_mask, selected = super().select_highest_overlaps(
            mask_pos.clone(), overlaps, n_max_boxes, align_metric
        )
        self.last_cra_stats = dict(orphans_before=0, orphans_after=0, recovered=0, changed_anchors=0)
        bs, ng, na = candidate_mask.shape
        if not ng or not na:
            self._record_cra_stats()
            return gt_idx, fg_mask, selected

        counts = torch.zeros((bs, ng), dtype=torch.long, device=gt_idx.device)
        counts.scatter_add_(1, gt_idx.long(), fg_mask.long())
        if not bool(((counts == 0) & candidate_mask.any(-1)).any()):
            self._record_cra_stats()
            return gt_idx, fg_mask, selected

        k = min(self.topk, na)
        scores, anchors = align_metric.masked_fill(~candidate_mask, -float("inf")).topk(
            k, dim=-1
        )
        # Transfer only sparse top-k candidates and occupied anchors, not the
        # full dense GT-by-anchor matrix. This remains an experimental CPU sync.
        for b in range(bs):
            if not bool(((counts[b] == 0) & candidate_mask[b].any(-1)).any()):
                continue
            active = fg_mask[b].bool().nonzero(as_tuple=False).flatten()
            old_owners = [-1] * na
            for a, g in zip(active.cpu().tolist(), gt_idx[b, active].long().cpu().tolist()):
                old_owners[a] = g
            top_scores = scores[b].float().cpu().tolist()
            top_anchors = anchors[b].long().cpu().tolist()
            choices = []
            for g in range(ng):
                pairs = sorted(
                    ((float(score), int(a)) for score, a in zip(top_scores[g], top_anchors[g])
                     if math.isfinite(score) and score > 0.0),
                    key=lambda pair: (-pair[0], pair[1]),
                )
                choices.append([a for _, a in pairs])
            new_owners, stats = repair_coverage(old_owners, choices)
            for key, value in stats.items():
                self.last_cra_stats[key] += value
            changed = [(a, old, new) for a, (old, new) in
                       enumerate(zip(old_owners, new_owners)) if old != new]
            if not changed:
                continue
            indices = torch.tensor([t[0] for t in changed], device=gt_idx.device)
            old = torch.tensor([t[1] for t in changed], device=gt_idx.device)
            new = torch.tensor([t[2] for t in changed], device=gt_idx.device)
            had_old = old >= 0
            if bool(had_old.any()):
                selected[b, old[had_old], indices[had_old]] = 0
            selected[b, new, indices] = 1
            gt_idx[b, indices] = new.to(gt_idx.dtype)
            fg_mask[b, indices] = 1
        self._record_cra_stats()
        return gt_idx, fg_mask, selected


def upgrade_assigner(original: TaskAlignedAssigner) -> CrowdReservedAssigner:
    """Copy official TAL hyperparameters exactly; only collision repair changes."""
    return CrowdReservedAssigner(
        topk=original.topk, topk2=original.topk2,
        num_classes=original.num_classes, alpha=original.alpha,
        beta=original.beta, stride=list(original.stride), eps=original.eps,
    )

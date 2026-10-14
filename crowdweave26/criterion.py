"""Opt-in GT-overlap supervision with the pinned official YOLO26 E2E loss."""
from __future__ import annotations
import torch
from ultralytics.utils.loss import E2ELoss
from .supervision import conflict_bce
from .cra import upgrade_assigner


class CrowdConflictE2ELoss(E2ELoss):
    """Preserve O2M/O2O loss decay and add a separately logged fourth component."""

    def __init__(self, model, conflict_weight: float = 0.25, cra_enabled: bool = False):
        super().__init__(model)
        if cra_enabled:
            self.one2many.assigner = upgrade_assigner(self.one2many.assigner)
            self.one2one.assigner = upgrade_assigner(self.one2one.assigner)
        if conflict_weight < 0:
            raise ValueError("conflict_weight must be nonnegative")
        self.conflict_weight = float(conflict_weight)

    def __call__(self, preds, batch):
        raw = preds[1] if isinstance(preds, tuple) else preds
        standard_loss, items = super().__call__(preds, batch)
        if self.conflict_weight == 0:
            return standard_loss, items
        logits = raw.get("crowd_conflict_logits")
        if logits is None:
            # The pinned YOLO26 validator evaluates an inference-mode tuple,
            # which deliberately does not expose auxiliary training logits.
            # Keep the same four loss columns, but do not penalize this branch.
            if not isinstance(preds, tuple):
                raise ValueError("CFSR logits absent from training output")
            zero = standard_loss.new_zeros(())
            return torch.cat((standard_loss.flatten(), zero.reshape(1))), {
                **items, "conflict_loss": zero
            }
        auxiliary = conflict_bce(logits, batch) * self.conflict_weight * logits.shape[0]
        return torch.cat((standard_loss.flatten(), auxiliary.reshape(1))), {
            **items, "conflict_loss": auxiliary.detach() / logits.shape[0]
        }

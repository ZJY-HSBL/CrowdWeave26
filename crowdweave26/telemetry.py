"""Per-epoch CRA coverage telemetry for the real YOLO26 training loop.

The counters are process-local, not aggregated across DDP ranks, and only
measure assignment coverage, not pedestrian detection accuracy.
"""
from __future__ import annotations
import csv
from pathlib import Path

FIELDS = ("epoch", "branch", "calls", "orphans_before", "orphans_after",
          "recovered", "changed_anchors")


def cra_totals(model) -> dict[str, dict[str, int]]:
    """Read cumulative training-only CRA counters for both detector branches."""
    model = getattr(model, "module", model)
    criterion = getattr(model, "criterion", None)
    result = {}
    for branch in ("one2many", "one2one"):
        sub = getattr(criterion, branch, None)
        assigner = getattr(sub, "assigner", None)
        stats = getattr(assigner, "cumulative_cra_stats", None)
        if stats is not None:
            result[branch] = {key: int(stats[key]) for key in FIELDS[2:]}
    return result


def record_cra_epoch(trainer):
    """Append per-epoch deltas to cra_coverage.csv at the end of a train epoch."""
    if getattr(trainer, "rank", -1) not in (-1, 0):
        return
    current = cra_totals(trainer.model)
    if not current:
        return
    previous = getattr(trainer, "_crowd_cra_previous", {})
    output = Path(trainer.save_dir) / "cra_coverage.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    first = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if first:
            writer.writeheader()
        for branch in ("one2many", "one2one"):
            if branch not in current:
                continue
            now = current[branch]
            before = previous.get(branch, {})
            delta = {key: now[key] - before.get(key, 0) for key in FIELDS[2:]}
            if any(value < 0 for value in delta.values()):
                raise ValueError("CRA cumulative counters unexpectedly decreased")
            writer.writerow({"epoch": int(trainer.epoch) + 1,
                             "branch": branch, **delta})
    trainer._crowd_cra_previous = current

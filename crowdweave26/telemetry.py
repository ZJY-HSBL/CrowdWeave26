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


# Unlike a raw GOTP loss of zero, pair statistics distinguish no-pair batches
# from candidates which already satisfy the desired overlap tolerance.
GOTP_FIELDS = ("calls", "matched_people", "nearby_pairs", "violating_pairs",
               "images_with_pairs", "unweighted_loss_sum")


def gotp_totals(model):
    """Read detached branch accumulators with one GPU-to-host transfer/epoch."""
    model = getattr(model, "module", model)
    criterion = getattr(model, "criterion", None)
    result = {}
    for branch in ("one2many", "one2one"):
        loss = getattr(criterion, branch, None)
        counter = getattr(loss, "gotp_running", None)
        if counter is not None:
            result[branch] = dict(zip(GOTP_FIELDS, counter.detach().cpu().tolist()))
    return result


def record_gotp_epoch(trainer):
    """Write training-only GOTP pair activity as branch-wise per-epoch deltas."""
    if getattr(trainer, "rank", -1) not in (-1, 0):
        return
    current = gotp_totals(trainer.model)
    if not current:
        return
    previous = getattr(trainer, "_crowd_gotp_previous", {})
    path = Path(trainer.save_dir) / "gotp_pairs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    first = not path.is_file()
    columns = ("epoch", "branch", *GOTP_FIELDS, "violating_pair_fraction",
               "mean_raw_loss_per_call")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if first:
            writer.writeheader()
        for branch in ("one2many", "one2one"):
            if branch not in current:
                continue
            now = current[branch]
            old = previous.get(branch, {})
            delta = {key: now[key] - old.get(key, 0.0) for key in GOTP_FIELDS}
            if any(value < -1e-7 for value in delta.values()):
                raise ValueError("GOTP running counters decreased unexpectedly")
            # Ratios are undefined without a corresponding denominator.
            count = delta["nearby_pairs"]
            calls = delta["calls"]
            writer.writerow({
                "epoch": int(trainer.epoch) + 1,
                "branch": branch,
                **{key: int(value) if key != "unweighted_loss_sum"
                   else float(value) for key, value in delta.items()},
                "violating_pair_fraction": float(delta["violating_pairs"]/count)
                    if count else "",
                "mean_raw_loss_per_call": float(delta["unweighted_loss_sum"]/calls)
                    if calls else "",
            })
    trainer._crowd_gotp_previous = current

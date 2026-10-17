"""Paired multi-seed comparison of *actual* externally supplied CrowdHuman metrics.

Rejects unpaired seeds and mismatched annotation, inference, evaluator or
training settings; this deliberately does not invent missing experiments.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
from pathlib import Path

METRICS = ("AP","mMR","JI")
REQUIRED_COMMON = ("annotation_sha256","evaluator_commit","upstream_commit",
                   "inference","training_protocol")
INFERENCE_KEYS = ("imgsz","conf","iou","max_det","nms_free")
TRAINING_KEYS = ("epochs","batch","imgsz","optimizer","augmentations")


def load_reports(path: Path) -> dict[tuple[str,int],dict]:
    reports = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_no,line in enumerate(handle,1):
            if not line.strip():
                continue
            row=json.loads(line)
            try:
                experiment=row["experiment"]
                seed=row["seed"]
                if not isinstance(experiment,str) or not experiment or type(seed) is not int or seed < 0:
                    raise ValueError("experiment/seed invalid")
                for key in REQUIRED_COMMON:
                    if key not in row:
                        raise ValueError(f"{key} missing")
                for key in ("inference","training_protocol"):
                    if not isinstance(row[key],dict):
                        raise ValueError(f"{key} must be object")
                for key in INFERENCE_KEYS:
                    if key not in row["inference"]:
                        raise ValueError(f"inference.{key} missing")
                for key in TRAINING_KEYS:
                    if key not in row["training_protocol"]:
                        raise ValueError(f"training_protocol.{key} missing")
                for metric in METRICS:
                    value=row[metric]
                    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"{metric} must be finite in [0,1]")
                key=(experiment,seed)
                if key in reports:
                    raise ValueError(f"duplicate record {key}")
            except (KeyError,ValueError,TypeError) as exc:
                raise ValueError(f"Invalid metrics record on line {line_no}: {exc}") from exc
            reports[key]=row
    if not reports:
        raise ValueError("No measured evaluation reports supplied")
    return reports


def _comparison_signature(row):
    # Model-specific method names and checkpoints intentionally differ. These
    # fields must be exactly identical for a controlled, paired comparison.
    inference = tuple((k,row["inference"][k]) for k in INFERENCE_KEYS)
    training = tuple((k,json.dumps(row["training_protocol"][k],sort_keys=True))
                     for k in TRAINING_KEYS)
    return (row["annotation_sha256"],row["evaluator_commit"],row["upstream_commit"],
            inference,training)


def bootstrap_interval(values: list[float], *, iterations: int = 5000,
                       seed: int = 2026) -> tuple[float,float]:
    if not values or iterations < 100:
        raise ValueError("At least one paired value and 100 bootstrap iterations required")
    rng = random.Random(seed)
    n=len(values)
    means=sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n))
                 for _ in range(iterations))
    return (means[int(.025*(iterations-1))],
            means[int(.975*(iterations-1))])


def compare(reports, baseline: str, treatment: str, *, iterations=5000, seed=2026):
    if not baseline or not treatment or baseline==treatment:
        raise ValueError("Require distinct baseline and treatment IDs")
    a={s:r for (name,s),r in reports.items() if name==baseline}
    b={s:r for (name,s),r in reports.items() if name==treatment}
    if len(a) < 3 or set(a) != set(b):
        raise ValueError("A complete paired experiment needs >=3 identical seeds for both groups")
    seeds=sorted(a)
    reference_sig=_comparison_signature(a[seeds[0]])
    for s in seeds:
        if (_comparison_signature(a[s]) != reference_sig or
            _comparison_signature(b[s]) != reference_sig):
            raise ValueError(f"Annotation, evaluator, training or inference mismatch at seed {s}")
    result={
        "baseline":baseline,"treatment":treatment,"paired_seeds":seeds,
        "annotation_sha256":a[seeds[0]]["annotation_sha256"],
        "evaluator_commit":a[seeds[0]]["evaluator_commit"],
        "method":"paired seed-level bootstrap percentile CI (descriptive, not a significance test)",
        "iterations":iterations,"metrics":{},
    }
    for metric in METRICS:
        baseline_mean=statistics.mean(a[s][metric] for s in seeds)
        treatment_mean=statistics.mean(b[s][metric] for s in seeds)
        # Improvement-oriented sign, including lower-is-better mMR.
        values=[(a[s][metric]-b[s][metric]) if metric=="mMR" else
                (b[s][metric]-a[s][metric]) for s in seeds]
        low,high=bootstrap_interval(values,iterations=iterations,seed=seed)
        result["metrics"][metric]={
            "direction":"lower_is_better" if metric=="mMR" else "higher_is_better",
            "baseline_mean":baseline_mean,"treatment_mean":treatment_mean,
            "mean_improvement":statistics.mean(values),
            "ci95": [low,high],
            "paired_improvements_by_seed":{str(s):v for s,v in zip(seeds,values)},
        }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True,
                        help="JSONL of observed benchmark results (not plans)")
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--treatment",required=True)
    parser.add_argument("--iterations",type=int,default=5000)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=compare(load_reports(args.results),args.baseline,args.treatment,
                   iterations=args.iterations,seed=args.seed)
    payload=json.dumps(result,indent=2)
    if args.output:
        if args.output.exists():
            parser.error("Refusing to overwrite paired result")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload+"\n",encoding="utf-8")
    print(payload)


if __name__=="__main__":
    main()

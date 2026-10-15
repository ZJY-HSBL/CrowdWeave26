"""Evaluate CrowdHuman results using the pinned CrowdDetection reference toolkits.

The reference code is loaded from a separate checkout; we do not vendor it.
GT ignore flags and matching behavior come from that reference.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PINNED = "9786f58869a55af3e0b51fc78f8638a825dae4a2"
UPSTREAM = "https://github.com/megvii-model/CrowdDetection"


def _verify_root(root: Path):
    root = Path(root).resolve()
    if not (root/"evaluate"/"APMRToolkits"/"database.py").is_file():
        raise FileNotFoundError("CrowdDetection evaluator source missing")
    result = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if result != PINNED:
        raise ValueError(f"Evaluator checkout mismatch: {result}; expected {PINNED}")
    return root


def evaluate(gt, dt, joined, evaluator_root):
    """Compute official-toolkit AP and MR, and serial JI via official JI functions."""
    root = _verify_root(Path(evaluator_root))
    gt, dt, joined = (Path(p).resolve() for p in (gt, dt, joined))
    folder = str(root/"evaluate")
    sys.path.insert(0, folder)
    sys.path.insert(0, str(root))
    from APMRToolkits.database import Database
    from JIToolkits.JI_tools import compute_matching, get_ignores  # noqa: F401
    from utils.misc_utils import load_json_lines
    # This import adds a relative path in upstream code, but root is also
    # inserted explicitly above so current working directory is irrelevant.
    import compute_JI

    database = Database(str(gt), str(dt), "box", None, 0)
    database.compare()
    ap, _ = database.eval_AP()
    mr, _ = database.eval_MR()
    rows = load_json_lines(str(joined))
    if not rows:
        raise ValueError("JI joined record file is empty")
    # Call the official single-record JI matcher and aggregator without
    # launching 100 subprocesses for the usual ten thresholds.
    class Collector:
        def __init__(self):
            self.results = []
        def put_nowait(self, value):
            self.results.append(value)
    ji_values = []
    for i in range(10):
        sink = Collector()
        compute_JI.compute_JI_with_ignore(sink, rows, 0.1*i, "box")
        _, ji = compute_JI.gather(sink.results)
        ji_values.append(float(ji))
    return {"AP": float(ap), "mMR": float(mr), "JI": float(max(ji_values)),
            "JI_by_score_threshold": {str(i/10): ji_values[i] for i in range(10)},
            "evaluation_source": UPSTREAM, "evaluation_commit": PINNED,
            "note": "Official CrowdDetection matching/metric functions; JI orchestrated serially"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odgt", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--ji-joined", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = evaluate(args.odgt, args.detections, args.ji_joined, args.evaluator_root)
    print(json.dumps(result, indent=2))
    if args.report:
        if args.report.exists():
            parser.error("Report already exists, refusing to overwrite")
        args.report.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()

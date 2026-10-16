# Evaluation protocol

Pin YOLO26 upstream to `a0fd1d727fe5ef5c0f6837646f794eba5b0575f5`.
Use unmodified YOLO26n/s and official P2 variants as separate controls.

Do not interpret CrowdHuman YOLO-converted labels as official evaluation:
ignored boxes require correct train-time negative masking and official benchmark
metrics (AP, log-average miss rate mMR, Jaccard Index/JI).

Use identical train/validation partitions, image sizes, optimizers, augmentations,
epochs and inference score settings; evaluate NMS and one-to-one NMS-free separately.
Report small/medium/large and heavy-occlusion subsets where supported.
Record parameters, GFLOPs at stated resolution, per-image GPU latency, GPU memory.
Run at least 3 random seeds, with paired uncertainty intervals on key comparisons.
Do not call modules novel without targeted prior-art comparison.


## CRA experiment (v0.4.0 candidate)

Compare the same YOLO26-P2 initialization and seeds with `--assignment standard`
versus `--assignment cra` (and optionally repeat with `--mechanism both`).
The assigner starts with official TAL and only repairs GTs with zero positives
when a feasible augmenting path through existing top-k candidates exists.
Measure positive coverage per batch, recovered GT fraction, batch training time
and GPU synchronization overhead. Report negative results as well as improvements.
The method does not fix ignored-region negatives and does not imply a benchmark gain.


## CrowdHuman reference evaluation adapter (v0.5.0 candidate)

The `crowdweave.evaluation` exporter reads the **original validation ODGT**
(not YOLO training labels) and emits separate `detections.odgt` and
`ji_joined.odgt` files. The detection boxes are clipped original-image
`[x, y, width, height]` rectangles, with detector confidence.
All original full-body ignore annotations are preserved and tagged -1 in JI.
The AP/MR evaluator receives the **unmodified** original GT ODGT.

The toolkit is the public CrowdDetection reference repository at the commit
recorded in `reference_evaluator.json`; only the matching and metric code is
used, and **not redistributed**. We invoke its AP, MR and JI matching
algorithms, orchestrating JI serially for reproducibility.

Example (PowerShell, run from this repository):
```powershell
git clone https://github.com/megvii-model/CrowdDetection.git official-crowd
git -C official-crowd checkout --detach 9786f58869a55af3e0b51fc78f8638a825dae4a2
python -m pip install numpy tqdm
python -m crowdweave.evaluation --odgt D:/CrowdHuman/annotation_val.odgt --images D:/CrowdHuman/Images --weights path/to/best.pt --output D:/CrowdHuman/reports/run01 --device 0 --conf 0.001 --max-det 1000
python -m crowdweave.official_metrics --odgt D:/CrowdHuman/annotation_val.odgt --detections D:/CrowdHuman/reports/run01/detections.odgt --ji-joined D:/CrowdHuman/reports/run01/ji_joined.odgt --evaluator-root official-crowd --report D:/CrowdHuman/reports/run01/official_metrics.json
```

Use the same confidence floor, input resolution, max detections, NMS settings,
checkpoint selection and official annotation split for all comparisons.
A synthetic reference-evaluator CI test checks ignored-region matching but
**is not** a CrowdHuman measured result. Train-time rectangular ignore-aware suppression is opt-in since v0.6.0,
but pixel-exact ignored regions and CutMix/CopyPaste remain unverified.


## Ignore-aware train-time matching (v0.6.0 candidate)

With `--ignore-aware`, CrowdHuman ignored fboxes are inserted into the
**same** Ultralytics Instances stream as ordinary GT before Mosaic, resizing,
translation and flips, then removed from positive GT immediately after native
augmentation. The ignore rectangles mask background-only classification BCE
at their final transformed anchor-center positions; foreground TAL matches
remain supervised. CFSR overlap BCE excludes cells inside ignored rectangles.

Compare both with and without this option in matched-budget baselines, and
report the ignore-aware setting for every ablation. Native box transforms
provide conservative axis-aligned approximations, not precise ignore polygons.
The official reference validation evaluator from v0.5.0 remains separate.


## v0.7.0 GOTP hypothesis

GOTP may reduce spurious overlap between predicted boxes belonging to
different nearby pedestrians. To avoid self-penalties, it first chooses
one TAL-positive anchor for each distinct GT per image. It penalizes only
`ReLU(IoU(pred_i,pred_j) - IoU(gt_i,gt_j) - margin)`.
This is an **auxiliary** loss (not a replacement for the native box/cls/DFL).
Use `--gotp-weight 0` as the matched control; compare with
`--gotp-weight 0.15` at identical seeds, images and training budgets.
Report the effective number of pairs and overhead in future experiments.
The matching representative uses detached predicted IoU as a quality proxy,
not the upstream TAL target-score tensor. Accuracy has not been measured.


## GOTP telemetry interpretation (v0.7.1 candidate)

`gotp_pairs.csv` uses **per-epoch** and **per-branch** counters. A GOTP
loss of zero is ambiguous: if `nearby_pairs=0`, the auxiliary objective
did not evaluate any eligible nearby people; if `nearby_pairs>0` but
`violating_pairs=0`, the tested pairs satisfy the tolerance. Nonzero
`violating_pairs` indicates a positive hinge penalty, but is not in itself
proof of nonzero parameter gradient or increased detection performance.

For each mechanism configuration, collect these counters at all seeds,
then measure real-GPU training elapsed time, peak memory and official
validation AP/mMR/JI under the identical benchmark protocol. The standalone
synthetic microbenchmark (`scripts/benchmark_gotp.py`) is only a debugging
cost floor, not a full-model speed estimate.

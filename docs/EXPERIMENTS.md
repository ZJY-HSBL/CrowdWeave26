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

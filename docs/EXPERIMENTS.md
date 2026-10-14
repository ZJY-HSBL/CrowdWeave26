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

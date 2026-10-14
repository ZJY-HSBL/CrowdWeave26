# Reproducible experiments

- Pin upstream Ultralytics at SHA in `upstream.json`; log dependencies, CUDA/GPU, data hashes, and preprocessing.
- Compare unmodified YOLO26n/s (P3/8, P4/16, P5/32) with official n/s-P2 (P2/4–P5/32) under identical seeds and training configuration. Official P2 is NOT original.
- Use CrowdHuman original train/val split. Retain head/visible/full GT identities and ignore regions. The YOLO TXT baseline is NOT ignore-aware during loss calculation.
- Report official CrowdHuman AP/mMR/JI using original-compatible evaluator; separately report Ultralytics mAP50/mAP50-95. These are not interchangeable.
- Evaluate the same checkpoint with NMS-based one-to-many and NMS-free one-to-one modes; explicitly label modes.
- Each candidate receives a controlled one-at-a-time ablation; then combine SCAD+CFSR, add VGCR, add CRA+GOTP, add ACDT.
- >=3 independent seeds for selected models, paired confidence intervals, heavy-occlusion subgroup, CityPersons external validation, FLOPs/params, measured latency, GPU VRAM.

| ID | Comparison | Current status |
|---|---|---|
| B0 | Standard YOLO26n | Not trained |
| B1 | Official YOLO26n-P2 | Not trained |
| A1 | B0 + SCAD | Standalone module only |
| A2 | B0 + VGCR | Hypothesis |
| A3 | B0 + CFSR | Hypothesis |
| A4 | B0 + CRA | Hypothesis |
| A5 | B0 + GOTP | Standalone loss only |
| A6 | B0 + ACDT | Hypothesis |
| FULL | All six | Not integrated |

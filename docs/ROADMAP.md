# CrowdWeave26 reproducible research roadmap

- **v0.1.0:** fixed official YOLO26 source and baseline tools.
- **v0.2.0:** opt-in SCAD/CFSR YOLO26-P2 architecture integration.
- **v0.3.0:** CFSR live augmented full-body overlap supervision and annotation audit.
- **v0.4.0:** CRA sparse augmenting-path collision-reserved assignment.
- **v0.4.1:** per-epoch O2M/O2O CRA coverage telemetry.
- **v0.5.0:** official CrowdDetection reference metric adapter for CrowdHuman
  AP, mMR and JI; validation ignore handling and paired NMS/NMS-free export.
- **Next:** train-time ignored-negative masking with geometrically consistent
  image transformations; real dataset training and verified reference metrics.
  VGCR, GOTP integration and ACDT remain independently gated hypotheses.
- **v1.0.0 target:** controlled six-mechanism multi-seed CrowdHuman ablations,
  CityPersons transfer, and GPU latency/memory/params/GFLOPs evaluation.

Only implemented and verified milestones may be released. Passing synthetic
tests cannot substitute for real CrowdHuman metrics or establish novelty.

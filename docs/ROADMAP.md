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


- **v0.6.0:** opt-in synchronized ignored-region negative suppression through
  Mosaic, affine scaling, translation and horizontal flips; baseline and
  combined synthetic one-epoch training gates.
- **Next:** GPU overhead, DDP support and real CrowdHuman training/validation
  on an identical dataset/split, optimizer budget and fixed seed. Contrast
  no-ignore and ignore-aware baselines before attributing changes to CRA/CFSR.


- **v0.7.0:** GOTP now computes pairwise excess predicted overlap for
  representative positives matched to different GTs in the genuine
  YOLO26 O2M/O2O criterion. Tested separately and with CRA/CFSR/ignore.
- **Next priority:** actual CrowdHuman training, official reference
  validation and paired multi-seed comparisons. Before publishing claims,
  measure active-pair incidence, gradient frequency and GPU overhead.


- **v0.7.1:** measured per-epoch GOTP eligible pairs and active hinge-pairs
  on both O2M/O2O branches, changed capped pairs to nearest GT neighbors,
  and added a synchronized CPU/CUDA forward+backward microbenchmark.
  CPU GitHub Actions synthetic checks confirm active penalties and gradients.
- **Next:** real CrowdHuman data and weights, independent training seeds,
  official AP/mMR/JI with fixed inference, and actual CUDA model overhead.


- **v0.8.0:** validate exact CrowdHuman converted-data manifests, reject
  train/val image-ID overlap and stale YAML paths, create explicit 3-seed
  ablation job plans and reject unfair evaluation comparisons. No new
  detector module introduced; this is experiment infrastructure.
- **Next:** run CrowdHuman GPU training with locally available authorized
  data and capture reference metrics; measure full-model latency, memory,
  multi-seed confidence intervals and GOTP active-pair incidence.

- **v0.8.0 release gate:** 86 ordinary unit tests pass (5 skipped);
  pinned YOLO26 integration and reference evaluator tests pass.
  Only synthetic functional fixtures and plan generation are used in CI.

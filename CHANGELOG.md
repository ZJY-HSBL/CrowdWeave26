# Changelog

## v0.7.1 — 2026-10-17 GOTP instrumentation and pair selection

- Count per-image representative unique pedestrians, eligible nearby pairs,
  positive hinge-penalty pairs and unweighted loss without per-step host
  scalar transfers.
- Export per-epoch O2M/O2O GOTP diagnostics to `gotp_pairs.csv`, excluding
  evaluation and preserving undefined ratios when no eligible pair exists.
- Extend actual synthetic one-epoch training gate to require CSV output.
- Fix capped GOTP pair selection: prioritize spatially nearest GT pairs,
  instead of annotation-order truncation.
- Add deterministic CPU/GPU-capable standalone GOTP forward/backward
  timing with synchronized GPU measurements; no real CrowdHuman results.



## v0.7.0 — 2026-10-16 unique-GT GOTP integration

- Implement genuine YOLO26 O2M/O2O pairwise GT-overlap-tolerant supervision.
- Select at most one representative positive per unique assigned ground-truth
  pedestrian to prevent false self-collision penalties.
- Expose independent weight, tolerance margin, and maximum-person controls.
- Preserve native box/class/DFL losses, branch gain schedule, and ignore/CRA/CFSR.
- Add synthetic gradient/identity tests and real-network training and validation gates.
- Research status: algorithm integrated, no measured CrowdHuman AP/mMR/JI gain.



## v0.6.0 — 2026-10-15 ignore-aware negative suppression

- Carry excluded annotations through original YOLO26 Mosaic, affine and flip geometries.
- Disentangle augmented ignore boxes from trainable person targets after formatting.
- Mask only unmatched background-classification anchors in ignored regions.
- Mask ignored cells in CFSR auxiliary supervision and retain positive TAL box losses.
- Gate actual one-epoch synthetic combined training with nonzero Mosaic and scale.
- No real CrowdHuman benchmark accuracy or GPU speed claim.



## v0.5.0 — 2026-10-15 CrowdHuman reference evaluation

- Export original-image YOLO26 detections to official CrowdHuman ODGT box format.
- Preserve full-body ignored GT boxes in AP/MR and JI reference evaluation.
- Pin external CrowdDetection evaluator code and test AP/MR/JI against ignored synthetic instances.
- Keep reference toolkit, datasets and model weights out of the project archive.
- No complete CrowdHuman benchmark or train-time ignored-negative loss masking.



## v0.4.1 — 2026-10-15 CRA matching telemetry

- Add per-epoch `cra_coverage.csv` for One-to-Many and One-to-One matching.
- Exclude validation matching from training counters; preserve existing loss structure.
- Add a regression gate requiring the real one-epoch synthetic trainer to write both branch records.
- No real CrowdHuman improvement claims.



## v0.4.0 — 2026-10-15

- Introduce Collision-Reserved Assignment (CRA) using deterministic sparse
  alternating paths to recover feasible GTs orphaned by official TAL collisions.
- Preserve official One-to-Many/One-to-One criteria and offer a separate
  `--assignment standard|cra` ablation switch.
- Add randomized invariant tests, synthetic collision/long-chain tests, and
  real pinned-YOLO26 forward/backward criterion tests.
- Add one-epoch synthetic Ultralytics CPU training gates for CRA-only and SCAD+CFSR+CRA, including checkpoint and validation.
- Correct validation-mode CFSR loss reporting when training-only conflict logits are unavailable.
- Pin matching CPU PyTorch/torchvision versions for reliable CI and update
  the research matrix and upstream manifest.
- Research boundary: no full CrowdHuman training, ignored-negative masking,
  performance overhead measurement or accuracy/novelty claims.



## v0.3.0 — 2026-10-15 supervised crowd-overlap learning

- Add augmentation-aligned full-body GT-overlap targets computed from each live YOLO training batch.
- Add bounded class-balanced BCE conflict supervision to official YOLO26 E2E O2M/O2O criterion.
- Expose --conflict-weight and preserve base loss components and original decay schedule.
- Add CPU geometry/gradient tests and pinned-upstream integration criterion test.
- Include original-coordinate CrowdHuman overlap and ignore-region annotation audit. Still no ignored-negative training mask, official benchmarks, GPU training or export validation.


## v0.2.0 — 2026-10-14

- Integrate experimental SCAD backbone downsampling and CFSR head fusion into official pinned YOLO26-P2.
- Preserve official P2/P3/P4/P5 output scales and One-to-Many/One-to-One branches.
- Add a custom Ultralytics trainer which installs the modules before pretrained-weight transfer.
- Add four-mode official YOLO26 forward/backward contract tests and a synthetic-batch training criterion test to CI.
- Gate GitHub Releases on both PyTorch unit tests and pinned-upstream integration tests.
- Research boundary: no trained CrowdHuman metrics, no verified originality, no calibrated CFSR overlap supervision, no validated model export.

## v0.1.1 — 2026-10-14

- Added standalone CFSR spatial-conflict feature routing prototype.
- Added independent SCAD/CFSR/GOTP components, rasterized overlap supervision, and gradient tests.
- Corrected the test-gated Release workflow and reproducible, git-tracked source ZIP packaging.
- Included experiment matrix, source manifest, model integration roadmap and protocol notes.
- This patch release snapshots the latest passing research starter. No trained YOLO26 gains are claimed.


## v0.1.0 — 2026-10-14

Research starter with pinned upstream YOLO26 revision.
- Baseline CLI for official YOLO26 standard and P2 configurations.
- CrowdHuman ODGT conversion, preserving associated head/visible/full boxes and ignored metadata.
- Standalone PyTorch SCAD, CFSR and GOTP prototypes; conflict overlap rasterization.
- CPU unit tests, automated CI and test-gated release of source ZIP.
- Research documentation, experimental conditions and reproducibility notes.

No integrated custom YOLO26 detector, trained checkpoint, detection mAP,
CrowdHuman mMR/JI, improvement claim or verified first-of-kind claim.

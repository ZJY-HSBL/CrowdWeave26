# Changelog

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

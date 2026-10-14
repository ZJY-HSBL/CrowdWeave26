# Changelog

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

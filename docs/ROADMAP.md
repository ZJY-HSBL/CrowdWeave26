# CrowdWeave26 research roadmap

Six hypotheses (not six proven inventions or measured improvements):

1. **SCAD** — Spatial-Conflict Anti-Aliasing Downsampling. Spatially blend detail and low-pass stride-2 paths, train conflict-aware gate after constructing overlap labels.
2. **VGCR** — Visibility-Geometry Consistent Reconstruction. Joint head/visible/full-box supervision with calibrated uncertainty.
3. **CFSR** — Conflict-Field Scale Routing. Route P2/P3/P4 information using supervised local crowd conflict estimates, not just static FPN weights.
4. **CRA** — Collision-Reserved Assignment. Reserve candidate locations for heavily overlapping instances before normal task-aligned assignment.
5. **GOTP** — GT-Overlap-Tolerant Pairwise Localization. Only penalize predicted overlap exceeding real GT overlap by margin.
6. **ACDT** — Ambiguity-Calibrated Dual-Head Transfer. Reliability-gated one-to-many -> one-to-one teacher signals.

## Milestones

- **v0.1.0**: fixed upstream version, official YOLO26/P2 baseline runner, CrowdHuman converter, standalone SCAD/GOTP prototypes, tests and CI; no integrated novelty claims.
- **v0.2.0**: actually integrate SCAD with Ultralytics parser and YAML, implement local conflict-field supervision and CFSR prototype.
- **v0.3.0**: VGCR head/visible/full geometry losses with occlusion ablations.
- **v0.4.0**: CRA sparse matching, robust GOTP loss integration.
- **v0.5.0**: ACDT dual-head transfer, inference consistency.
- **v1.0.0**: 6 independent ablations, grouped combinations, official CrowdHuman protocol, 3+ seeds and statistical reports.

Reject or redesign failed modules rather than retaining all six for marketing. Originality requires independent systematic prior-art review.

**v0.1.0 limitations:** no YOLO integration of SCAD/GOTP; no official ignore-aware CrowdHuman training/evaluation; no trained weights or benchmark claims.

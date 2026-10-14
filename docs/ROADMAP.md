# CrowdWeave26 research milestones

- v0.1.0: pinned YOLO26 baseline, CrowdHuman conversion and standalone research prototypes.
- v0.2.0: SCAD/CFSR opt-in YOLO26-P2 graph integration with upstream contract tests.
- v0.3.0: augmented-batch full-body GT-overlap CFSR supervision and CrowdHuman metadata audit.
- v0.4.0 candidate: CRA sparse augmenting-path positive reservation on official O2M/O2O TAL.
- Next: ignore-aware negative training (must transform ignored geometry with augmentation),
  VGCR head/visible/full-body geometry, GOTP unique-GT pairing and ACDT transfer.
- v1.0.0 target: six-way controlled ablations with official CrowdHuman AP/mMR/JI,
  multi-seed paired intervals, CityPersons transfer and GPU throughput/latency.

**Research policy:** no originality or accuracy claims without prior-art screening,
matched-budget training, official evaluation and measured overhead. CRA is opt-in
because sparse repair currently incurs CPU synchronization on GPU.

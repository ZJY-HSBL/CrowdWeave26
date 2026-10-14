# CrowdWeave26 research milestones

v0.1.0: pinned official YOLO26 baseline, CrowdHuman conversion, standalone SCAD/CFSR/GOTP, CI, reproducible release.

v0.2.0: integrate SCAD and CFSR into Ultralytics `parse_model`, YAML and model forward, with conflict-map auxiliary supervision.

v0.3.0: VGCR linked full/visible/head box geometry and uncertainty loss.

v0.4.0: CRA collision-aware GT-to-candidate assignment, plus unique-GT matched GOTP loss.

v0.5.0: ACDT ambiguity-calibrated one-to-many → one-to-one transfer; validate NMS-free inference.

v1.0.0: integrated six-way multi-seed ablations on CrowdHuman; official AP/mMR/JI and heavy-occlusion results; CityPersons transfer; speed/parameter/GFLOPs profiling.

These milestones are **not** guarantees of accuracy improvement or novelty.
Do not promote failed mechanisms into the integrated model. Assess each change
with equal training budgets and random seeds.

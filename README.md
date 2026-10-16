# CrowdWeave26

**Occlusion-aware dense pedestrian detection research based on YOLO26**  
**基于 YOLO26 的密集行人遮挡感知检测研究**

[Releases](https://github.com/ZJY-HSBL/CrowdWeave26/releases) ·
[Research protocol](docs/EXPERIMENTS.md) · [Roadmap](docs/ROADMAP.md)

CrowdWeave26 is an **experimental** PyTorch/Ultralytics research project.
It preserves a pinned official YOLO26 source baseline and develops six
independently ablatable mechanisms. The code does **not** yet establish
novelty, higher AP, lower miss rate or a new state of the art.

本仓库以固定提交的官方 YOLO26 为基线，研究密集行人场景中的尺度混叠、
遮挡冲突与正样本竞争。所有改进均须通过真实数据消融、官方评测及多随机种子
检验后才能形成论文结论；代码测试通过不代表检测精度提升。

## Research status / 研究进度

| Mechanism | Hypothesis | Implementation status |
| --- | --- | --- |
| **SCAD** | Spatial-Conflict Anti-Aliasing Downsampling | Integrated into YOLO26-P2, opt-in |
| **CFSR** | Conflict-Field Scale Routing | Integrated into YOLO26-P2, supervised by augmented GT box overlap |
| **CRA** | Collision-Reserved Assignment | Integrated into official TAL as an opt-in training-only experiment on `main` |
| **GOTP** | GT-Overlap-Tolerant Pairwise Loss | Standalone prototype; not in official training criterion |
| **VGCR** | Visibility-Geometry Consistent Reconstruction | Design proposal, not integrated |
| **ACDT** | Ambiguity-Calibrated Dual-Head Transfer | Design proposal, not integrated |

SCAD/CFSR require the four-scale YOLO26-P2 graph. CRA is independent of
SCAD/CFSR and can be compared using an otherwise unchanged YOLO26-P2 baseline.

**Upstream:** [Ultralytics](https://github.com/ultralytics/ultralytics)
at commit `a0fd1d727fe5ef5c0f6837646f794eba5b0575f5`.
We do not vendor Ultralytics source, trained weights or CrowdHuman data.
Review upstream AGPL-3.0/commercial licensing before redistribution or use.

## Installation / 安装

Python 3.11 is used in CI. On Windows PowerShell, from the repository root:

```powershell
python -m pip install -e ".[dev]"
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
python -m pytest -q
python scripts/smoke_integration.py --mode both --imgsz 96
```

The bootstrap script clones and installs the pinned official YOLO26
source into `ultralytics-src/`. For reproducibility, the CPU GitHub Actions
environment pins PyTorch 2.10.0 and torchvision 0.25.0.

## CrowdHuman conversion / 数据准备

Prepare the official CrowdHuman images and ODGT annotations yourself.
No dataset is included in this repository.

```powershell
python -m crowdweave.prepare --odgt D:/CrowdHuman/annotation_train.odgt --images D:/CrowdHuman/Images --output D:/CrowdHuman/yolo --split train
python -m crowdweave.prepare --odgt D:/CrowdHuman/annotation_val.odgt --images D:/CrowdHuman/Images --output D:/CrowdHuman/yolo --split val --write-yaml
python -m crowdweave.audit --metadata D:/CrowdHuman/yolo/metadata --split train
```

The converter uses full-body person boxes for YOLO labels, while retaining
head/visible/full-body associations and ignore annotations in JSON metadata.
**Important:** ignored boxes are *not* transformed into train-time negative
masks. Regular YOLO mAP is **not** official CrowdHuman AP/mMR/JI.

## Controlled ablations / 对照实验

Use the same training budget, seed, input resolution, augmentation and data
for each comparison. These are **commands**, not completed benchmark runs.

```powershell
# P2 baseline
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n-p2 --mechanism none --assignment standard --seed 0

# CRA only: isolate the assignment mechanism
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n-p2 --mechanism none --assignment cra --seed 0

# SCAD + CFSR only
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n-p2 --mechanism both --assignment standard --conflict-weight 0.25 --seed 0

# SCAD + CFSR + CRA
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n-p2 --mechanism both --assignment cra --conflict-weight 0.25 --seed 0
```

Validate a trained checkpoint using the ordinary loader (trusted checkpoints
only; PyTorch checkpoints may execute code). For example:

```powershell
python -m crowdweave.cli val --data D:/CrowdHuman/yolo/crowdhuman.yaml --weights path/to/best.pt
```

CRA repairs only **feasible** zero-positive ground-truth objects using sparse
alternating paths over official TAL top-k candidates. Each anchor retains a
single GT owner, and existing GT coverage cannot decrease. The prototype
synchronizes sparse candidates to CPU during assignment; its training
throughput, GPU memory, DDP and torch.compile compatibility remain unverified.

## Verification and releases / 测试与版本管理

- `python -m pytest -q`: standalone geometry, assignment, CLI and data tests.
- `python -m pytest -q tests/test_integration_api.py`: real pinned YOLO26
  model construction, forward/backward and original criterion checks.
- `python scripts/smoke_train.py --mechanism none --assignment cra` and
  `python scripts/smoke_train.py --mechanism both --assignment cra`:
  synthetic CPU training epochs through the actual Ultralytics trainer,
  including checkpoint finalization and validation.
- `.github/workflows/release.yml`: runs test gates before creating a
  tagged source ZIP on version changes. No datasets or weights in releases.

Current stable tag: **v0.6.0**. CRA-only and combined SCAD+CFSR+CRA
synthetic one-epoch CPU train-loop gates pass; CrowdHuman accuracy remains untested.

## Research limitations / 研究边界

1. No CrowdHuman full training or official AP/mMR/JI result is published.
2. Ignore regions are preserved as metadata but not yet loss-masked.
3. CFSR overlap labels are rectangle-overlap proxies, **not** true visible
   occlusion maps.
4. CRA coverage gains in synthetic collisions do not establish detection gains.
5. Novelty requires a documented comparison against assignment and
   crowd-detection prior art, not merely a new module name.

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for the required controlled protocol.


### v0.4.1: CRA coverage telemetry

The `main` branch additionally records `cra_coverage.csv` for CRA-enabled
training. The log contains **per-epoch, per-branch** matcher calls, feasible
orphan GTs before/after recovery, recovered GTs and changed candidate owners.
Validation loss matching does not contribute to these training counters.
This file describes matching coverage, **not** official CrowdHuman AP/mMR/JI.
The telemetry is process-local; distributed aggregation is not yet supported.


## CrowdHuman reference AP/mMR/JI (v0.5.0)

The evaluation adapter exports original-image-coordinate `dtboxes`, preserves
non-person and explicitly ignored annotations for reference matching, and
runs the pinned CrowdDetection AP/MR and JI metric implementations.
The reference toolkit remains a **separate checkout**.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the exact two commands,
evaluator SHA, reproducibility settings and dataset requirements.

**Boundary:** this closes the prediction-to-evaluation format gap but does
not implement ignored-region loss masking during training; no actual trained
CrowdHuman accuracy numbers have been produced.


### Commands: original CrowdHuman AP/mMR/JI protocol

```powershell
git clone https://github.com/megvii-model/CrowdDetection.git official-crowd
git -C official-crowd checkout --detach 9786f58869a55af3e0b51fc78f8638a825dae4a2
python -m pip install numpy tqdm
python -m crowdweave.evaluation --odgt D:/CrowdHuman/annotation_val.odgt --images D:/CrowdHuman/Images --weights path/to/best.pt --output D:/CrowdHuman/reports/experiment1 --conf 0.001 --max-det 1000 --device 0
python -m crowdweave.official_metrics --odgt D:/CrowdHuman/annotation_val.odgt --detections D:/CrowdHuman/reports/experiment1/detections.odgt --ji-joined D:/CrowdHuman/reports/experiment1/ji_joined.odgt --evaluator-root official-crowd --report D:/CrowdHuman/reports/experiment1/official_metrics.json
```

Use `--nms-free` when specifically evaluating the one-to-one inference
branch; report both branches separately rather than selecting the stronger
one after seeing test results. This version tests the evaluation implementation
on synthetic fixtures but does not include real pretrained detection weights.


## v0.6.0 — ignore-aware negative suppression

`--ignore-aware` causes all ignored CrowdHuman full-body boxes to join the
original Ultralytics instance stream as sentinel class -1 **before** Mosaic,
affine resize and flip. A custom final Format removes the sentinels from
positive targets and passes their transformed boxes to the criterion.
Within these regions, only **unmatched classification background anchors**
are excluded from BCE; actual assigned person positives are preserved.
The CFSR auxiliary conflict BCE likewise ignores masked feature-grid cells.

```powershell
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n-p2 --mechanism both --assignment cra --ignore-aware --seed 0
```

This is opt-in to permit strictly matched baseline controls; it requires
`metadata/ignored_train.json` and fails if metadata are missing. Ignore
regions are rectangle approximations under geometric warps, **not**
pixel-accurate segmentation masks. Copy-paste and CutMix are not validated;
the research CLI defaults must retain those augmentations disabled for
ignore-aware training. Multi-GPU/DDP and production-speed benchmarks remain
unverified.


## v0.7.0 development: GOTP unique-person pairwise supervision

GOTP is now an optional `--gotp-weight` in the genuine YOLO26 training
criterion. Foreground positive anchors are grouped by the GT assigned by TAL,
with **one representative per GT** selected by detached predicted/assigned
box IoU. The auxiliary term penalizes excessive *predicted* overlap only for
nearby distinct GT people, relative to their **true** full-body overlap.
No person is compared against itself, and the original class/box/DFL losses
and official One-to-Many/One-to-One decay remain intact.

```powershell
# Matched ablation: change only --gotp-weight
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n-p2 --ignore-aware --assignment cra --gotp-weight 0.0 --seed 0
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n-p2 --ignore-aware --assignment cra --gotp-weight 0.15 --gotp-margin 0.03 --seed 0
```

**Limitations:** the representative per GT is selected from detached IoU,
not from the actual TAL target quality; pair capping drops some large-crowd
comparisons, and CPU synchronization/throughput is not yet benchmarked.
No actual CrowdHuman training has been run. GOTP is a candidate, not a
verified improvement or novel method.

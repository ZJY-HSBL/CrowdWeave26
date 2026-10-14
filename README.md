# CrowdWeave26

**Occlusion-Aware Dense Pedestrian Detection with YOLO26**  
**基于 YOLO26 的密集行人遮挡感知检测研究**

CrowdWeave26 develops six candidate mechanisms for dense pedestrian detection. This is a reproducible research project: proposed mechanisms are not assumed to be novel or beneficial until ablated and independently tested.

## v0.1.0
- Pinned upstream Ultralytics YOLO26 baseline runner, including official P2 variants.
- CrowdHuman ODGT converter retaining original head/visible/full boxes and ignored regions as metadata.
- Standalone PyTorch SCAD, CFSR and GOTP research prototypes (not yet integrated into YOLO26).
- Unit tests and an automated, test-gated GitHub Releases workflow, plus a roadmap and experiment protocol.

### Windows setup
```powershell
python -m pip install -e ".[dev]"
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
python -m pytest -q
```

### CrowdHuman preparation
```powershell
python -m crowdweave.prepare --odgt D:/CrowdHuman/annotation_train.odgt --images D:/CrowdHuman/Images --output D:/CrowdHuman/yolo --split train
python -m crowdweave.prepare --odgt D:/CrowdHuman/annotation_val.odgt --images D:/CrowdHuman/Images --output D:/CrowdHuman/yolo --split val --write-yaml
```

### YOLO26 training
```powershell
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n --batch 4 --epochs 200 --seed 0
python -m crowdweave.cli train --data D:/CrowdHuman/yolo/crowdhuman.yaml --variant n-p2 --batch 4 --epochs 200 --seed 0
python -m crowdweave.cli val --data D:/CrowdHuman/yolo/crowdhuman.yaml --weights runs/crowdweave26/yolo26_n_seed0/weights/best.pt
python -m crowdweave.cli val --data D:/CrowdHuman/yolo/crowdhuman.yaml --weights runs/crowdweave26/yolo26_n_seed0/weights/best.pt --nms-free
```

**Important:** Plain YOLO labels cannot mask ignored CrowdHuman regions. Our converter preserves ignore metadata but v0.1.0 does NOT solve ignore-aware negative supervision. Use an evaluator implementing original CrowdHuman AP/mMR/JI protocol; do not identify regular YOLO mAP with CrowdHuman official results.

Six candidates:
1. SCAD — Spatial-Conflict Anti-Aliasing Downsampling (standalone prototype)
2. VGCR — Visibility-Geometry Consistent Reconstruction (proposal)
3. CFSR — Conflict-Field Scale Routing (standalone prototype)
4. CRA — Collision-Reserved Assignment (proposal)
5. GOTP — GT-Overlap-Tolerant Pairwise Loss (standalone prototype)
6. ACDT — Ambiguity-Calibrated Dual-Head Transfer (proposal)

The repository does not vendor Ultralytics or distribute datasets/weights. Upstream uses AGPL-3.0/commercial licensing; examine obligations before integrating and redistributing derivatives. Details: [Research roadmap](docs/ROADMAP.md), [Experiments](docs/EXPERIMENTS.md).

**Release policy:** bump VERSION on main to trigger an automated GitHub Release.

The research modules in `crowdweave26/` are separate from the official baseline trainer in `crowdweave/cli.py`; they have **not** yet been connected to the Ultralytics model parser or optimized end-to-end. Run `python -m pytest -q` before any model training.

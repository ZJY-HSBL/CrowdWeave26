"""Official YOLO26 training/validation entry point."""
from __future__ import annotations

import argparse
import json
import os

MODELS = {"n": "yolo26n.yaml", "s": "yolo26s.yaml", "n-p2": "yolo26n-p2.yaml", "s-p2": "yolo26s-p2.yaml"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crowdweave")
    p.add_argument("mode", choices=["train", "val"])
    p.add_argument("--data", required=True)
    p.add_argument("--variant", choices=MODELS, default="n")
    p.add_argument("--weights", help="Existing checkpoint for validation/continuation")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="0")
    p.add_argument("--nms-free", action="store_true", help="Validate one-to-one without NMS")
    p.add_argument("--mechanism", choices=("none", "scad", "cfsr", "both"), default="none")
    p.add_argument("--conflict-weight", type=float, default=0.25,
                   help="CFSR auxiliary GT-overlap BCE gain, for cfsr/both training")
    p.add_argument("--project", default="runs/crowdweave26")
    return p


def run(args: argparse.Namespace) -> None:
    if args.mode == "val" and not args.weights:
        raise ValueError("Validation requires --weights")
    if args.nms_free and args.mode != "val":
        raise ValueError("--nms-free is for validation only")
    if min(args.imgsz, args.batch, args.epochs) <= 0:
        raise ValueError("imgsz, batch and epochs must be positive")
    if not 0 <= args.conflict_weight <= 10:
        raise ValueError("--conflict-weight must be in [0,10]")
    if args.mechanism != "none":
        if args.mode != "train":
            raise ValueError("Validate custom checkpoints without --mechanism")
        if args.variant not in ("n-p2", "s-p2"):
            raise ValueError("SCAD/CFSR require a P2 variant")
    from ultralytics import YOLO  # installed from pinned checkout, lazy import
    model = YOLO(args.weights or MODELS[args.variant])
    common = dict(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device, project=args.project)
    if args.mode == "train":
        kwargs = dict(**common, epochs=args.epochs, seed=args.seed, deterministic=True,
                      close_mosaic=min(20, args.epochs),
                      name=f"yolo26_{args.variant}_{args.mechanism}_seed{args.seed}")
        if args.mechanism != "none":
            from crowdweave26.integration import CrowdWeaveTrainer, ENV_MODE, ENV_CONFLICT_WEIGHT
            os.environ[ENV_MODE] = args.mechanism
            os.environ[ENV_CONFLICT_WEIGHT] = str(args.conflict_weight)
            kwargs["trainer"] = CrowdWeaveTrainer
        model.train(**kwargs)
    else:
        metrics = model.val(**common, nms=not args.nms_free,
                            name="one_to_one" if args.nms_free else "one_to_many_nms")
        print(json.dumps(metrics.results_dict, indent=2, default=str))
        print("YOLO mAP is not CrowdHuman official AP/mMR/JI; ignore regions need protocol-correct evaluation.")


def main() -> None:
    p = build_parser()
    try:
        run(p.parse_args())
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()

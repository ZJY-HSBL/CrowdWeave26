"""Tiny end-to-end CPU training smoke: verifies the actual trainer/optimizer path.

Creates only synthetic data. It is NOT a CrowdHuman accuracy experiment.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image, ImageDraw


def run(mechanism="none", assignment="cra"):
    from ultralytics import YOLO
    from crowdweave26.integration import (
        CrowdWeaveTrainer, ENV_MODE, ENV_CRA_ENABLED, ENV_CONFLICT_WEIGHT
    )
    os.environ[ENV_MODE] = mechanism
    os.environ[ENV_CRA_ENABLED] = "1" if assignment == "cra" else "0"
    os.environ[ENV_CONFLICT_WEIGHT] = "0.25"
    with TemporaryDirectory(prefix="crowdweave_train_") as work:
        root = Path(work)
        for split in ("train", "val"):
            (root/"images"/split).mkdir(parents=True)
            (root/"labels"/split).mkdir(parents=True)
            image = Image.new("RGB", (96, 96), (60, 70, 85))
            d = ImageDraw.Draw(image)
            d.rectangle((21, 20, 67, 84), outline=(230, 200, 160), width=3)
            d.rectangle((38, 18, 83, 83), outline=(170, 220, 220), width=3)
            image.save(root/"images"/split/"scene.jpg")
            (root/"labels"/split/"scene.txt").write_text(
                "0 0.458333 0.541667 0.479167 0.666667\n"
                "0 0.630208 0.526042 0.468750 0.677083\n",
                encoding="utf-8"
            )
        data = root/"data.yaml"
        data.write_text(
            "path: "+json.dumps(str(root))+"\ntrain: images/train\n"
            "val: images/val\nnc: 1\nnames: ['person']\n", encoding="utf-8"
        )
        model = YOLO("yolo26n-p2.yaml")
        model.train(
            data=str(data), trainer=CrowdWeaveTrainer, epochs=1,
            imgsz=96, batch=1, nbs=1, workers=0, device="cpu",
            amp=False, optimizer="SGD", lr0=0.001, warmup_epochs=0,
            pretrained=False, seed=0, deterministic=True, val=False,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, scale=0.0,
            translate=0.0, fliplr=0.0, flipud=0.0,
            plots=False, cache=False, save=True, verbose=False,
            project=str(root/"runs"), name="smoke", exist_ok=True,
        )
        assert (root/"runs"/"smoke"/"weights"/"last.pt").is_file()
        print("Synthetic one-epoch CPU train complete:", mechanism, assignment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", choices=("none","scad","cfsr","both"), default="none")
    parser.add_argument("--assignment", choices=("standard","cra"), default="cra")
    args = parser.parse_args()
    run(args.mechanism, args.assignment)

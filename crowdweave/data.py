"""CrowdHuman ODGT -> YOLO full-body labels, retaining ignored metadata.

WARNING: YOLO text files cannot mask ignored regions during loss computation.
A separate official evaluator is needed for comparable CrowdHuman metrics.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from PIL import Image


def xywh_to_yolo(box, width: int, height: int):
    if box is None or len(box) != 4 or width <= 0 or height <= 0:
        return None
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def is_ignored(gt: dict) -> bool:
    return gt.get("tag") != "person" or gt.get("extra", {}).get("ignore", 0) == 1


def _image_link(src: Path, dst: Path, mode: str):
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite: {dst}")
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown image mode: {mode}")


def convert_split(odgt: Path, images: Path, output: Path, split: str, mode: str = "hardlink") -> dict:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    output, odgt, images = Path(output), Path(odgt), Path(images)
    if not odgt.is_file() or not images.is_dir():
        raise FileNotFoundError("ODGT or image directory missing")
    image_dir, label_dir = output / "images" / split, output / "labels" / split
    if image_dir.exists() or label_dir.exists():
        raise FileExistsError(f"Split {split} already exists; use a fresh output directory")
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    ignored, associated = {}, {}
    n_images = n_boxes = n_ignored = 0
    try:
        with odgt.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                image_id = record["ID"]
                if (not isinstance(image_id, str) or Path(image_id).name != image_id or
                        image_id.endswith(".jpg") or image_id in {".", ".."}):
                    raise ValueError(f"Invalid ID on line {line_number}: {image_id!r}")
                source = images / (image_id + ".jpg")
                if not source.is_file():
                    raise FileNotFoundError(f"Missing image: {source}")
                with Image.open(source) as image:
                    width, height = image.size
                rows, ignored_boxes, persons = [], [], []
                for gt in record.get("gtboxes", []):
                    fbox = gt.get("fbox")
                    valid = xywh_to_yolo(fbox, width, height)
                    if is_ignored(gt):
                        n_ignored += 1
                        ignored_boxes.append({"tag": gt.get("tag"), "fbox": fbox,
                                              "extra": gt.get("extra", {})})
                        continue
                    if valid is None:
                        continue
                    rows.append("0 " + " ".join(f"{x:.8f}" for x in valid))
                    persons.append({"fbox": fbox, "vbox": gt.get("vbox"), "hbox": gt.get("hbox")})
                _image_link(source, image_dir / source.name, mode)
                (label_dir / (image_id + ".txt")).write_text(
                    "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                ignored[image_id], associated[image_id] = ignored_boxes, persons
                n_images += 1
                n_boxes += len(rows)
        meta = output / "metadata"
        meta.mkdir(exist_ok=True)
        (meta / f"ignored_{split}.json").write_text(json.dumps(ignored), encoding="utf-8")
        (meta / f"associated_{split}.json").write_text(json.dumps(associated), encoding="utf-8")
        summary = {"split": split, "images": n_images, "trainable_persons": n_boxes,
                   "ignored_annotations": n_ignored, "image_mode": mode}
        (meta / f"summary_{split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    except Exception:
        shutil.rmtree(image_dir, ignore_errors=True)
        shutil.rmtree(label_dir, ignore_errors=True)
        raise


def write_dataset_yaml(output: Path) -> Path:
    output = Path(output).resolve()
    if not all((output / "images" / split).is_dir() for split in ("train", "val")):
        raise FileNotFoundError("Convert both train and val splits first")
    path = output / "crowdhuman.yaml"
    path.write_text(f"path: {json.dumps(str(output))}\n"
                    "train: images/train\nval: images/val\n"
                    "nc: 1\nnames: ['person']\n", encoding="utf-8")
    return path

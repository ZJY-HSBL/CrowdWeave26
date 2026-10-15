"""CrowdHuman ODGT prediction exporter and benchmark-format adapter.

Builds both the official AP/MR detection JSONL and JI joined JSONL.
It keeps ignored GT annotations for evaluation; this does not correct
the training-time background loss inside ignored regions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from PIL import Image


def annotation_rows(path):
    """Yield intact CrowdHuman annotation records; reject unsafe/duplicate IDs."""
    ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record.get("ID")
            if (not isinstance(image_id, str) or not image_id or
                    Path(image_id).name != image_id or image_id in (".", "..") or
                    "/" in image_id or "\\" in image_id or image_id in ids):
                raise ValueError(f"Invalid or duplicate image ID at line {number}: {image_id!r}")
            if not isinstance(record.get("gtboxes"), list):
                raise ValueError(f"Missing gtboxes at line {number}")
            ids.add(image_id)
            yield record


def _box_xywh(box, width, height):
    """Clamp xywh to image dimensions, returning None for invalid area."""
    if len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
        raise ValueError("Box must contain 4 finite coordinates")
    x, y, w, h = map(float, box)
    if w <= 0 or h <= 0:
        return None
    x1 = min(max(x, 0.), float(width))
    y1 = min(max(y, 0.), float(height))
    x2 = min(max(x + w, 0.), float(width))
    y2 = min(max(y + h, 0.), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def detection_entry(xyxy_score, width, height):
    """Convert one [x1,y1,x2,y2,confidence] into official dtboxes format."""
    if len(xyxy_score) != 5:
        raise ValueError("Detection must be xyxy plus confidence")
    x1, y1, x2, y2, score = [float(v) for v in xyxy_score]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)):
        raise ValueError("Detection has NaN/inf")
    if not 0 <= score <= 1:
        raise ValueError("Detection confidence outside [0,1]")
    box = _box_xywh([x1, y1, x2-x1, y2-y1], width, height)
    return {"box": box, "score": score} if box is not None else None


def make_evaluation_records(annotation, width, height, detections):
    """Return (AP/MR record, JI joined record) with true ignored GTs preserved."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    dt = []
    for pred in detections:
        item = detection_entry(pred, width, height)
        if item is not None:
            dt.append(item)
    det_record = {"ID": annotation["ID"], "width": width,
                  "height": height, "dtboxes": dt}
    ji_gt = []
    for gt in annotation["gtboxes"]:
        # Official CrowdDetection body evaluation sets non-person or ignored GTs to -1.
        ignore = gt.get("tag") != "person" or gt.get("extra", {}).get("ignore", 0) != 0
        box = gt.get("fbox")
        if box is None:
            raise ValueError("CrowdHuman gtboxes element has no fbox")
        rect = _box_xywh(box, width, height)
        if rect is None:
            raise ValueError("Invalid GT full-body box; refusing a silent benchmark change")
        ji_gt.append({"box": rect, "tag": -1 if ignore else 1})
    joined = {**det_record, "gtboxes": ji_gt}
    return det_record, joined


def export_predictions(odgt, images, detections, ji_output, predictor):
    """Stream an image at a time; 'predictor' returns iterable xyxy-score detections."""
    images, detections, ji_output = Path(images), Path(detections), Path(ji_output)
    if detections.resolve() == ji_output.resolve():
        raise ValueError("Detection and joined files must be different paths")
    for dest in (detections, ji_output):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite existing evaluation output: {dest}")
    count = 0
    try:
        with detections.open("x", encoding="utf-8") as dt_stream, ji_output.open(
            "x", encoding="utf-8"
        ) as ji_stream:
            for annotation in annotation_rows(odgt):
                path = images / (annotation["ID"] + ".jpg")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing CrowdHuman image: {path}")
                with Image.open(path) as im:
                    width, height = im.size
                preds = predictor(path)
                record, joined = make_evaluation_records(annotation, width, height, preds)
                dt_stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                ji_stream.write(json.dumps(joined, ensure_ascii=False)+"\n")
                count += 1
        if count == 0:
            raise ValueError("Empty CrowdHuman annotation file")
    except Exception:
        for dest in (detections, ji_output):
            dest.unlink(missing_ok=True)
        raise
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odgt", type=Path, required=True, help="Original annotation_val.odgt")
    parser.add_argument("--images", type=Path, required=True, help="Original validation JPG directory")
    parser.add_argument("--weights", type=Path, required=True, help="Trusted YOLO26 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New directory for exported JSONL")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--nms-free", action="store_true",
                        help="Use YOLO26 one-to-one NMS-free inference")
    args = parser.parse_args()
    if not 0 < args.conf < 1 or not 0 < args.iou < 1 or args.max_det < 1 or args.imgsz < 32:
        parser.error("Require conf/iou in (0,1), max-det >=1, imgsz >=32")
    if args.output.exists():
        parser.error("Use a fresh output directory; evaluation artifacts must not be overwritten")
    args.output.mkdir(parents=True, exist_ok=False)
    from ultralytics import YOLO
    # Import research model definitions before loading trusted research checkpoints.
    import crowdweave26.integration  # noqa: F401
    detector = YOLO(str(args.weights))
    def predictor(path):
        result = detector.predict(
            source=str(path), device=args.device, imgsz=args.imgsz,
            conf=args.conf, iou=args.iou, max_det=args.max_det,
            nms=not args.nms_free, verbose=False
        )[0]
        if result.boxes is None:
            return []
        return [
            [*xyxy, float(score)]
            for xyxy, score, cls in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist()
            ) if int(cls) == 0
        ]
    try:
        count = export_predictions(
            args.odgt, args.images,
            args.output/"detections.odgt", args.output/"ji_joined.odgt", predictor
        )
        config = {"images": count, "checkpoint": str(args.weights.resolve()),
                  "source_annotations": str(args.odgt.resolve()),
                  "conf": args.conf, "iou": args.iou, "max_det": args.max_det,
                  "imgsz": args.imgsz, "nms_free": args.nms_free}
        (args.output/"inference_settings.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        print(json.dumps(config, indent=2))
    except Exception:
        import shutil
        shutil.rmtree(args.output)
        raise


if __name__ == "__main__":
    main()

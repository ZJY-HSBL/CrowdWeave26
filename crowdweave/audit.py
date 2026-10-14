"""CrowdHuman original-coordinate annotation overlap/ignore audit.

Diagnostic statistics only. Not an official CrowdHuman metric implementation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def _xyxy(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x, y, w, h = map(float, box)
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (x, y, w, h))) or min(w, h) <= 0:
        return None
    return x, y, x + w, y + h


def _intersection(a, b):
    if a is None or b is None:
        return 0.
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def _iou(a, b):
    inter = _intersection(a, b)
    if not inter:
        return 0.
    area = lambda q: (q[2]-q[0])*(q[3]-q[1])
    return inter/(area(a)+area(b)-inter)


def audit_metadata(directory, split="train", *, iou_threshold=0.3):
    """Read converter metadata and summarize nonignored body overlaps and ignored boxes."""
    if not 0 < iou_threshold < 1:
        raise ValueError("iou_threshold must be in (0, 1)")
    directory = Path(directory)
    associated = json.loads((directory/f"associated_{split}.json").read_text(encoding="utf-8"))
    ignored = json.loads((directory/f"ignored_{split}.json").read_text(encoding="utf-8"))
    if set(associated) != set(ignored):
        raise ValueError("associated and ignored IDs differ")
    images = len(associated)
    result = dict(split=split, images=images, valid_persons=0,
                  ignored_annotations=0, multi_person_images=0,
                  overlap_images=0, crowded_images=0, ignored_images=0,
                  person_pairs=0, overlapping_pairs=0, ignored_touching_persons=0)
    for key, person_rows in associated.items():
        people = [b for row in person_rows if (b := _xyxy(row.get("fbox"))) is not None]
        ignore_rows = ignored[key]
        regions = [b for row in ignore_rows if (b := _xyxy(row.get("fbox"))) is not None]
        result["valid_persons"] += len(people)
        result["ignored_annotations"] += len(ignore_rows)
        result["multi_person_images"] += int(len(people) > 1)
        result["crowded_images"] += int(len(people) >= 10)
        result["ignored_images"] += int(bool(ignore_rows))
        matched = 0
        for i, a in enumerate(people):
            for b in people[i+1:]:
                result["person_pairs"] += 1
                if _iou(a,b) >= iou_threshold:
                    matched += 1
            if any(_intersection(a,ib) > 0 for ib in regions):
                result["ignored_touching_persons"] += 1
        result["overlapping_pairs"] += matched
        result["overlap_images"] += int(matched > 0)
    result["overlap_image_fraction"] = round(result["overlap_images"]/max(images,1),6)
    result["ignored_image_fraction"] = round(result["ignored_images"]/max(images,1),6)
    result["iou_threshold"] = iou_threshold
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split", choices=("train","val"), default="train")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    a = parser.parse_args()
    print(json.dumps(audit_metadata(a.metadata, a.split, iou_threshold=a.iou_threshold), indent=2))


if __name__ == "__main__":
    main()

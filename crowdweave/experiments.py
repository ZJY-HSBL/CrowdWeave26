"""CrowdHuman data preflight and deterministic ablation plan generator.

No downloads, network calls, model training or synthetic benchmark results.
This tool is intended to stop invalid experiments BEFORE expensive GPU runs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from PIL import Image

DEFAULT_STUDY = ("B1", "B1_IGN", "R1_IGN", "SC1_IGN",
                 "SCR1_IGN", "GOTP1", "GOTP_CRA", "GOTP_FULL")
REQUIRED_MATRIX = ("experiment", "variant", "mechanism", "assignment",
                   "conflict_weight", "gotp_weight", "ignore_aware")


def _hash_annotations(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda x: x.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _valid_fbox(box):
    try:
        if len(box) != 4:
            return False
        x, y, w, h = map(float, box)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0


def _labels(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5 or parts[0] != "0":
                raise ValueError(f"Expected class 0 + xywh: {path}:{line_no}")
            try:
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError as exc:
                raise ValueError(f"Malformed YOLO box: {path}:{line_no}") from exc
            if (not all(math.isfinite(v) for v in (x, y, w, h))
                    or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                    or x-w/2 < -1e-6 or x+w/2 > 1+1e-6
                    or y-h/2 < -1e-6 or y+h/2 > 1+1e-6):
                raise ValueError(f"Out-of-bounds YOLO box: {path}:{line_no}")
            count += 1
    return count


def preflight(root: Path, *, verify_images: bool = False) -> dict:
    """Validate converted train/val images, labels and intact metadata keys.

    SHA-256 fingerprints annotation/metadata files, NOT JPEG image contents.
    """
    root = Path(root).resolve()
    if not (root / "crowdhuman.yaml").is_file():
        raise FileNotFoundError("Expected converter-created crowdhuman.yaml")
    files_for_hash = [root / "crowdhuman.yaml"]
    summaries = {}
    split_ids = {}
    for split in ("train", "val"):
        image_dir, label_dir = root/"images"/split, root/"labels"/split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing converted {split} images/labels")
        images = {p.stem:p for p in image_dir.glob("*.jpg")}
        labels = {p.stem:p for p in label_dir.glob("*.txt")}
        if not images or set(images) != set(labels):
            raise ValueError(f"Missing images/labels or unmatched IDs for {split}")
        if len(images) != len(list(image_dir.glob("*.jpg"))):
            raise ValueError(f"Duplicate image stems in {split}")
        ignored_path = root/"metadata"/f"ignored_{split}.json"
        associated_path = root/"metadata"/f"associated_{split}.json"
        for meta in (ignored_path, associated_path):
            if not meta.is_file():
                raise FileNotFoundError(f"Missing converted metadata: {meta}")
        ignored = json.loads(ignored_path.read_text(encoding="utf-8"))
        associated = json.loads(associated_path.read_text(encoding="utf-8"))
        if not isinstance(ignored, dict) or not isinstance(associated, dict):
            raise ValueError(f"Metadata must be image-ID dictionaries ({split})")
        if set(images) != set(ignored) or set(images) != set(associated):
            raise ValueError(f"Image IDs differ from metadata in {split}")
        total_boxes, total_ignore = 0, 0
        if verify_images:
            for image in images.values():
                try:
                    with Image.open(image) as img:
                        img.verify()
                except (OSError,ValueError) as exc:
                    raise ValueError(f"Invalid JPEG image: {image}") from exc
        for image_id in sorted(images):
            n = _labels(labels[image_id])
            rows = associated[image_id]
            ig = ignored[image_id]
            if not isinstance(rows, list) or not isinstance(ig, list):
                raise ValueError(f"Expected lists of GT annotations: {split}/{image_id}")
            if n != len(rows):
                raise ValueError(f"Valid box count differs from associated metadata: {split}/{image_id}")
            if any(not isinstance(g,dict) or not _valid_fbox(g.get("fbox")) for g in rows):
                raise ValueError(f"Invalid associated GT boxes: {split}/{image_id}")
            if any(not isinstance(g,dict) or not _valid_fbox(g.get("fbox")) for g in ig):
                raise ValueError(f"Invalid ignored GT boxes: {split}/{image_id}")
            total_boxes += n
            total_ignore += len(ig)
        files_for_hash.extend([ignored_path, associated_path])
        files_for_hash.extend(labels.values())
        summaries[split] = {"images":len(images), "trainable_persons":total_boxes,
                            "ignored_annotations":total_ignore}
        split_ids[split] = set(images)
    intersection = split_ids["train"] & split_ids["val"]
    if intersection:
        raise ValueError(f"Train/val image IDs overlap, e.g. {sorted(intersection)[0]}")
    return {
        "dataset_root":str(root),
        "annotation_sha256":_hash_annotations(files_for_hash,root),
        "fingerprint_scope":"converter YAML, all label TXT and associated/ignored JSON; JPEG bytes NOT hashed",
        "splits":summaries,
        "checked":True,
        "jpeg_headers_verified":bool(verify_images),
        "official_scores":"not_computed",
    }


def read_matrix(path: Path) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8",newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(REQUIRED_MATRIX).issubset(reader.fieldnames or []):
            raise ValueError("Experiment matrix is missing required columns")
        rows = {}
        for row in reader:
            name = row["experiment"]
            if not name or name in rows:
                raise ValueError(f"Duplicate or empty experiment ID: {name}")
            if row["variant"] not in ("n","s","n-p2","s-p2"):
                raise ValueError(f"Invalid variant in {name}")
            if row["mechanism"] not in ("none","scad","cfsr","both") or (
                row["mechanism"] != "none" and not row["variant"].endswith("-p2")
            ):
                raise ValueError(f"Invalid architecture in {name}")
            if row["assignment"] not in ("standard","cra"):
                raise ValueError(f"Invalid assignment in {name}")
            if row["ignore_aware"] not in ("0","1"):
                raise ValueError(f"Invalid ignore flag in {name}")
            for key in ("conflict_weight","gotp_weight"):
                try:
                    val = float(row[key])
                except (ValueError,TypeError) as exc:
                    raise ValueError(f"Invalid {key} in {name}") from exc
                if not math.isfinite(val) or not 0 <= val <= 10:
                    raise ValueError(f"Invalid {key} in {name}")
            if row["mechanism"] not in ("cfsr","both") and float(row["conflict_weight"]) != 0:
                raise ValueError(f"Conflict weight without CFSR in {name}")
            rows[name] = row
    return rows


def build_plan(preflight_result, matrix, ids, seeds, *, imgsz, batch, epochs, device, project):
    if not preflight_result.get("checked"):
        raise ValueError("Dataset preflight required")
    if len(set(ids)) != len(ids) or any(name not in matrix for name in ids):
        raise ValueError("Unknown or duplicate experiment IDs")
    if len(set(seeds)) != len(seeds) or not seeds or any(s < 0 for s in seeds):
        raise ValueError("Seeds must be nonempty distinct nonnegative integers")
    if min(imgsz,batch,epochs) < 1:
        raise ValueError("Positive image size/batch/epochs required")
    runs = []
    for experiment in ids:
        row = matrix[experiment]
        for seed in seeds:
            command = [
                "python","-m","crowdweave.cli","train",
                "--data",str(Path(preflight_result["dataset_root"])/"crowdhuman.yaml"),
                "--variant",row["variant"],
                "--mechanism",row["mechanism"],"--assignment",row["assignment"],
                "--conflict-weight",row["conflict_weight"],
                "--gotp-weight",row["gotp_weight"],
                "--imgsz",str(imgsz),"--batch",str(batch),"--epochs",str(epochs),
                "--seed",str(seed),"--device",device,
                "--project",str(Path(project)/experiment),
            ]
            if row["ignore_aware"] == "1":
                command.append("--ignore-aware")
            runs.append({"experiment":experiment,"seed":seed,
                         "data_sha256":preflight_result["annotation_sha256"],
                         "configuration":{k:row[k] for k in REQUIRED_MATRIX},
                         "command":command,"status":"planned_not_executed"})
    return {
        "schema":"crowdweave-experiment-plan-v1",
        "dataset":preflight_result,
        "protocol":{"imgsz":imgsz,"batch":batch,"epochs":epochs,"device":device,
                    "seeds":list(seeds),"source_matrix":"experiment_matrix.csv"},
        "run_count":len(runs), "runs":runs,
        "benchmark_results":"none; commands only",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("preflight","plan"))
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--matrix",type=Path,default=Path(__file__).resolve().parents[1]/"experiment_matrix.csv")
    ap.add_argument("--verify-images",action="store_true",
                    help="Validate JPEG headers in addition to annotation structures")
    ap.add_argument("--experiments",nargs="+",default=list(DEFAULT_STUDY))
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--epochs",type=int,default=200)
    ap.add_argument("--device",default="0")
    ap.add_argument("--project",type=Path,default=Path("runs/crowdweave_study"))
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    verified=preflight(args.data_root,verify_images=args.verify_images)
    if args.action=="preflight":
        result=verified
    else:
        result=build_plan(verified,read_matrix(args.matrix),args.experiments,args.seeds,
                          imgsz=args.imgsz,batch=args.batch,epochs=args.epochs,
                          device=args.device,project=args.project)
    if args.output:
        if args.output.exists():
            ap.error("Refusing to overwrite an existing experiment manifest")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result if args.action=="preflight" else {
        "run_count":result["run_count"],
        "data_sha256":verified["annotation_sha256"],
        "output":str(args.output) if args.output else None,
        "status":"planned_not_executed",
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

"""CrowdHuman ODGT converter CLI."""
import argparse
import json
from pathlib import Path
from .data import convert_split, write_dataset_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--odgt", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    p.add_argument("--write-yaml", action="store_true")
    args = p.parse_args()
    result = convert_split(args.odgt, args.images, args.output, args.split, args.mode)
    print(json.dumps(result, indent=2))
    if args.write_yaml:
        print(f"Dataset YAML: {write_dataset_yaml(args.output)}")
    if result["ignored_annotations"]:
        print("WARNING: YOLO text labels cannot mask ignored regions; use protocol-correct evaluation.")


if __name__ == "__main__":
    main()

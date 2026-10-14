"""Create deterministic source ZIP of git-tracked text files; no datasets/weights."""
from __future__ import annotations
import argparse
import subprocess
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_SUFFIXES = {".pt",".pth",".onnx",".engine",".pyc",".pyo",".zip",".odgt"}

def gather_files():
    data = subprocess.check_output(["git","ls-files","-z"],cwd=ROOT)
    items = []
    for item in data.split(b"\0"):
        if not item:
            continue
        rel=Path(item.decode("utf-8"))
        if rel.suffix.lower() in SKIP_SUFFIXES or any(p in {"__pycache__",".pytest_cache"} for p in rel.parts):
            continue
        path=ROOT / rel
        if path.is_file():
            items.append(path)
    return sorted(items,key=lambda p:p.relative_to(ROOT).as_posix())

def build_archive(tag: str, output: Path):
    version=tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    declared=(ROOT/"VERSION").read_text(encoding="utf-8").strip()
    if tag != "v"+version or tag != declared:
        raise ValueError("Tag, VERSION and pyproject.toml version must agree")
    output.mkdir(parents=True,exist_ok=True)
    dest=output/f"CrowdWeave26-{tag}.zip"
    with zipfile.ZipFile(dest,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in gather_files():
            rel=path.relative_to(ROOT).as_posix()
            info=zipfile.ZipInfo(f"CrowdWeave26-{tag}/{rel}",date_time=(2020,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            archive.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return dest

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--output",type=Path,default=Path("dist"))
    args=parser.parse_args()
    print(build_archive(args.tag,args.output))

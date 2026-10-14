import json
from pathlib import Path
import pytest
from PIL import Image
from crowdweave.data import convert_split, is_ignored, write_dataset_yaml, xywh_to_yolo


def test_clips_boxes_and_rejects_invalid():
    assert xywh_to_yolo([0, 0, 50, 100], 100, 100) == (0.25, 0.5, 0.5, 1.)
    assert xywh_to_yolo([-20, 10, 40, 50], 100, 100) == (0.1, 0.35, 0.2, 0.5)
    assert xywh_to_yolo([200, 10, 40, 50], 100, 100) is None
    assert xywh_to_yolo([0, 0, 0, 50], 100, 100) is None


def test_ignore_logic():
    assert is_ignored({"tag": "mask"})
    assert is_ignored({"tag": "person", "extra": {"ignore": 1}})
    assert not is_ignored({"tag": "person"})


def test_split_conversion_and_metadata(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    Image.new("RGB", (100, 100)).save(source / "sample.jpg")
    record = {"ID": "sample", "gtboxes": [
        {"tag": "person", "fbox": [0, 0, 50, 100], "vbox": [0, 0, 30, 50], "hbox": [4, 2, 12, 14]},
        {"tag": "mask", "fbox": [20, 20, 30, 30]}]}
    odgt = tmp_path / "ann.odgt"
    odgt.write_text(json.dumps(record) + "\n")
    root = tmp_path / "out"
    stats = convert_split(odgt, source, root, "train", mode="copy")
    assert stats["trainable_persons"] == 1 and stats["ignored_annotations"] == 1
    assert (root / "labels/train/sample.txt").read_text().startswith("0 0.25000000")
    assert len(json.loads((root / "metadata/associated_train.json").read_text())["sample"]) == 1
    with pytest.raises(FileExistsError):
        convert_split(odgt, source, root, "train", mode="copy")
    convert_split(odgt, source, root, "val", mode="copy")
    assert "nc: 1" in write_dataset_yaml(root).read_text()


def test_unsafe_id_refused(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    odgt = tmp_path / "ann.odgt"
    odgt.write_text('{"ID":"../escape","gtboxes":[]}\n')
    with pytest.raises(ValueError):
        convert_split(odgt, source, tmp_path / "out", "train")

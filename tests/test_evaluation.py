"""CrowdHuman reference-format tests with ignored annotations."""
import json
from pathlib import Path
import pytest
from PIL import Image
from crowdweave.evaluation import (
    annotation_rows, detection_entry, make_evaluation_records, export_predictions
)


def example():
    return {"ID": "scene0001", "gtboxes": [
        {"tag": "person", "fbox": [5, 5, 40, 40], "extra": {"ignore": 0}},
        {"tag": "person", "fbox": [20, 18, 40, 35], "extra": {"ignore": 1}},
        {"tag": "mask", "fbox": [60, 5, 20, 30]}
    ]}


def test_boxes_are_xywh_after_export_and_ignore_survives():
    dt, joined = make_evaluation_records(
        example(), 100, 80, [[5,5,45,45,.9], [61,5,79,25,.2]]
    )
    assert dt["dtboxes"][0]["box"] == [5.0,5.0,40.0,40.0]
    assert len(joined["gtboxes"]) == 3
    assert [g["tag"] for g in joined["gtboxes"]] == [1, -1, -1]
    assert joined["dtboxes"] == dt["dtboxes"]


def test_clips_detection_to_boundaries():
    b = detection_entry([-4,-8,120,110,.5], 100, 80)
    assert b["box"] == [0.,0.,100.,80.]


def test_bad_scores_are_rejected():
    with pytest.raises(ValueError):
        detection_entry([0,0,10,10,float("nan")], 100,100)
    with pytest.raises(ValueError):
        detection_entry([0,0,10,10,1.2], 100,100)


def test_streaming_export_and_integrity(tmp_path):
    images = tmp_path/"Images"
    images.mkdir()
    Image.new("RGB", (100,80)).save(images/"scene0001.jpg")
    gt = tmp_path/"annotation_val.odgt"
    gt.write_text(json.dumps(example())+"\n", encoding="utf-8")
    output = tmp_path/"out"
    output.mkdir()
    cnt = export_predictions(gt, images, output/"dt.odgt", output/"ji.odgt",
                             lambda path: [[5,5,45,45,.9]])
    assert cnt == 1
    dt = json.loads((output/"dt.odgt").read_text())
    ji = json.loads((output/"ji.odgt").read_text())
    assert dt["ID"] == ji["ID"] == "scene0001"
    assert ji["width"] == 100 and ji["height"] == 80
    assert ji["gtboxes"][1]["tag"] == -1
    with pytest.raises(FileExistsError):
        export_predictions(gt, images, output/"dt.odgt", output/"other.odgt",
                           lambda path: [])


def test_duplicate_ids_rejected(tmp_path):
    gt = tmp_path/"dup.odgt"
    gt.write_text((json.dumps(example())+"\n")*2)
    with pytest.raises(ValueError):
        list(annotation_rows(gt))


def test_missing_image_rolls_back_outputs(tmp_path):
    gt = tmp_path/"a.odgt"
    gt.write_text(json.dumps(example())+"\n")
    with pytest.raises(FileNotFoundError):
        export_predictions(gt, tmp_path, tmp_path/"dt.odgt", tmp_path/"ji.odgt",
                           lambda _: [])
    assert not (tmp_path/"dt.odgt").exists()
    assert not (tmp_path/"ji.odgt").exists()


def test_invalid_gt_never_silently_removed():
    a=example()
    a["gtboxes"][0]["fbox"] = [5,5,0,40]
    with pytest.raises(ValueError):
        make_evaluation_records(a,100,80,[])



def test_preserves_original_outside_image_gt_coordinates():
    ann = {"ID": "out", "gtboxes": [{"tag": "person", "fbox": [-2, 3, 10, 12]}]}
    _, joined = make_evaluation_records(ann, 100, 100, [])
    assert joined["gtboxes"][0]["box"] == [-2., 3., 10., 12.]

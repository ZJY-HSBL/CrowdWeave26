"""Integration against pinned CrowdDetection reference AP/MR/JI functions."""
import json
import os
from pathlib import Path
import pytest
from PIL import Image
from crowdweave.evaluation import export_predictions
from crowdweave.official_metrics import evaluate

REFERENCE = os.environ.get("CROWDWEAVE_REFERENCE_EVAL")
pytestmark = pytest.mark.skipif(not REFERENCE, reason="Pinned CrowdDetection checkout not installed")


def test_reference_apmr_ji_with_ignore(tmp_path):
    imgdir = tmp_path/"Images"
    imgdir.mkdir()
    annotation = tmp_path/"annotation_val.odgt"
    rows = [
        {"ID": "sample_a", "gtboxes": [
            {"tag": "person", "fbox": [5, 5, 35, 40], "hbox": [10, 6, 12, 12]},
            {"tag": "person", "fbox": [55, 5, 35, 40], "hbox": [60, 6, 12, 12],
             "extra": {"ignore": 1}}
        ]},
        {"ID": "sample_b", "gtboxes": [
            {"tag": "person", "fbox": [10, 10, 35, 40], "hbox": [12, 11, 13, 12]},
        ]}
    ]
    annotation.write_text("".join(json.dumps(x)+"\n" for x in rows), encoding="utf-8")
    for row in rows:
        Image.new("RGB", (100, 80)).save(imgdir/(row["ID"]+".jpg"))
    detections = tmp_path/"det.odgt"
    joined = tmp_path/"ji.odgt"
    predictions = {
        "sample_a": [[5,5,40,45,.95], [57,5,89,44,.90]],
        "sample_b": [[10,10,45,50,.94], [65,51,80,70,.8]]
    }
    assert export_predictions(annotation,imgdir,detections,joined,
                               lambda p: predictions[p.stem]) == 2
    result = evaluate(annotation,detections,joined,Path(REFERENCE))
    assert 0 <= result["AP"] <= 1
    assert 0 <= result["mMR"] <= 1
    assert 0 <= result["JI"] <= 1
    assert len(result["JI_by_score_threshold"]) == 10
    # Ignored "person" with an overlapping detection must not count as FP.
    # Compare with a control where that extra detection is deleted.
    control_dt = tmp_path/"control_dt.odgt"
    control_ji = tmp_path/"control_ji.odgt"
    predictions["sample_a"] = predictions["sample_a"][:1]
    export_predictions(annotation,imgdir,control_dt,control_ji,lambda p: predictions[p.stem])
    control = evaluate(annotation,control_dt,control_ji,Path(REFERENCE))
    assert result["AP"] == pytest.approx(control["AP"], abs=1e-8)
    assert result["mMR"] == pytest.approx(control["mMR"], abs=1e-8)

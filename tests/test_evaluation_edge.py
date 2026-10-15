"""Additional format checks for official benchmark exporter."""
import json
import pytest
from crowdweave.evaluation import make_evaluation_records


def test_nonperson_ignore_region_preserved():
    gt = {"ID": "im", "gtboxes": [{"tag": "mask","fbox":[0,0,10,10]}]}
    det, ji = make_evaluation_records(gt,32,32,[[1,1,9,9,.5]])
    assert ji["gtboxes"][0]["tag"] == -1
    assert det["dtboxes"][0]["score"] == .5


def test_no_predictions_still_exports_image():
    gt = {"ID": "im", "gtboxes":[{"tag":"person","fbox":[1,1,10,10]}]}
    det, ji = make_evaluation_records(gt,32,32,[])
    assert det["dtboxes"] == []
    assert len(ji["gtboxes"]) == 1


def test_invalid_scores_prevent_false_evaluation():
    gt = {"ID":"im","gtboxes":[]}
    with pytest.raises(ValueError):
        make_evaluation_records(gt, 40, 40, [[1,1,10,10,-.1]])

"""Paired metric comparison rejects invalid/cross-protocol claims."""
import json
import pytest
from crowdweave.comparison import load_reports, compare, bootstrap_interval


def sample(experiment,seed,ap,mmr,ji):
    return {
        "experiment":experiment,"seed":seed,"AP":ap,"mMR":mmr,"JI":ji,
        "annotation_sha256":"a"*64,
        "upstream_commit":"pinnedYOLO26",
        "evaluator_commit":"pinnedCrowdDetection",
        "inference":{"imgsz":640,"conf":.001,"iou":.7,
                     "max_det":1000,"nms_free":False},
        "training_protocol":{"epochs":200,"batch":4,"imgsz":640,
                             "optimizer":"SGD","augmentations":"fixed_default"},
    }


def records():
    rows={}
    for seed in range(3):
        a=sample("B1_IGN",seed,.70+seed*.01,.45-seed*.01,.60+seed*.01)
        b=sample("GOTP1",seed,.72+seed*.01,.43-seed*.01,.61+seed*.01)
        rows[(a["experiment"],seed)]=a
        rows[(b["experiment"],seed)]=b
    return rows


def test_paired_delta_and_mmr_lower_is_better():
    result=compare(records(),"B1_IGN","GOTP1",iterations=250,seed=17)
    assert result["paired_seeds"]==[0,1,2]
    assert result["metrics"]["AP"]["mean_improvement"]==pytest.approx(.02)
    assert result["metrics"]["mMR"]["mean_improvement"]==pytest.approx(.02)
    assert result["metrics"]["JI"]["mean_improvement"]==pytest.approx(.01)
    assert result["metrics"]["AP"]["ci95"]==pytest.approx([.02,.02])


def test_duplicate_or_missing_seed_rejected():
    rows=records()
    rows.pop(("GOTP1",1))
    with pytest.raises(ValueError,match="identical seeds"):
        compare(rows,"B1_IGN","GOTP1")


def test_cross_protocol_or_dataset_scores_cannot_be_compared():
    for field,value in [("annotation_sha256","mismatch"),
                        ("evaluator_commit","another")]:
        rows=records()
        rows[("GOTP1",1)][field]=value
        with pytest.raises(ValueError,match="mismatch"):
            compare(rows,"B1_IGN","GOTP1")
    rows=records()
    rows[("GOTP1",2)]["inference"]["nms_free"]=True
    with pytest.raises(ValueError,match="mismatch"):
        compare(rows,"B1_IGN","GOTP1")
    rows=records()
    rows[("GOTP1",0)]["training_protocol"]["epochs"]=100
    with pytest.raises(ValueError,match="mismatch"):
        compare(rows,"B1_IGN","GOTP1")


def test_result_jsonl_validation_rejects_fake_and_invalid_scores(tmp_path):
    file=tmp_path/"metrics.jsonl"
    row=sample("B1_IGN",0,.70,.45,.60)
    file.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="duplicate"):
        load_reports(file)
    row["AP"]=float("nan")
    file.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="AP"):
        load_reports(file)
    row.pop("AP")
    file.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="AP"):
        load_reports(file)


def test_deterministic_bootstrap_and_minimum_iterations():
    values=[-0.1,0.0,0.1,0.2]
    one=bootstrap_interval(values,iterations=200,seed=42)
    two=bootstrap_interval(values,iterations=200,seed=42)
    assert one==two
    with pytest.raises(ValueError,match="iterations"):
        bootstrap_interval(values,iterations=20)


def test_prevent_inference_seed_boolean_and_insufficient_seeds(tmp_path):
    p=tmp_path/"records.jsonl"
    row=sample("A",True,.5,.5,.5)
    p.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="seed"):
        load_reports(p)
    with pytest.raises(ValueError,match=">=3"):
        compare({("A",0):sample("A",0,.5,.5,.5),
                 ("B",0):sample("B",0,.5,.5,.5)},"A","B")

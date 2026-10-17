"""Preflight and ablation-plan regression tests using tiny converted fixtures."""
import json
from pathlib import Path

import pytest

from crowdweave.experiments import build_plan, preflight, read_matrix


def fixture_dataset(tmp_path):
    root=tmp_path/"crowdhuman"
    for split in ("train","val"):
        image_dir=root/"images"/split
        label_dir=root/"labels"/split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (root/"metadata").mkdir(exist_ok=True)
        ident="sample_"+split
        (image_dir/(ident+".jpg")).write_bytes(b"mock image marker only")
        (label_dir/(ident+".txt")).write_text("0 0.5 0.5 0.2 0.4\n",encoding="utf-8")
        (root/"metadata"/("associated_"+split+".json")).write_text(
            json.dumps({ident:[{"fbox":[20,10,30,55],"vbox":[24,15,21,49]}]}),
            encoding="utf-8"
        )
        (root/"metadata"/("ignored_"+split+".json")).write_text(
            json.dumps({ident:[{"tag":"mask","fbox":[40,10,30,60]}]}),
            encoding="utf-8"
        )
    (root/"crowdhuman.yaml").write_text(
        "path: \""+str(root)+"\"\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['person']\n",
        encoding="utf-8"
    )
    return root


def fixture_matrix(tmp_path):
    p=tmp_path/"matrix.csv"
    p.write_text(
        "experiment,variant,mechanism,assignment,conflict_weight,gotp_weight,ignore_aware\n"
        "B1_IGN,n-p2,none,standard,0,0,1\n"
        "GOTP1,n-p2,none,standard,0,0.15,1\n"
        "SCR1_IGN,n-p2,both,cra,0.25,0,1\n",
        encoding="utf-8"
    )
    return p


def test_converted_dataset_preflight_and_deterministic_fingerprint(tmp_path):
    root=fixture_dataset(tmp_path)
    first=preflight(root)
    second=preflight(root)
    assert first["checked"] and first["annotation_sha256"]==second["annotation_sha256"]
    assert first["splits"]["train"]=={
        "images":1,"trainable_persons":1,"ignored_annotations":1
    }
    assert "JPEG bytes NOT hashed" in first["fingerprint_scope"]
    label=root/"labels"/"train"/"sample_train.txt"
    label.write_text("0 0.51 0.50 0.20 0.40\n")
    assert preflight(root)["annotation_sha256"] != first["annotation_sha256"]


def test_three_seed_paired_ablation_commands_are_only_plans(tmp_path):
    p=preflight(fixture_dataset(tmp_path))
    m=read_matrix(fixture_matrix(tmp_path))
    selection=["B1_IGN","GOTP1","SCR1_IGN"]
    plan=build_plan(p,m,selection,[0,1,2],imgsz=640,batch=4,epochs=200,
                    device="0",project=tmp_path/"runs")
    assert plan["run_count"]==9 and len(plan["runs"])==9
    assert plan["benchmark_results"].startswith("none")
    assert all(row["status"]=="planned_not_executed" for row in plan["runs"])
    for row in plan["runs"]:
        assert "--ignore-aware" in row["command"]
        assert row["configuration"]["ignore_aware"]=="1"
        assert row["data_sha256"]==p["annotation_sha256"]
    baseline=plan["runs"][0]["command"]
    gotp=plan["runs"][3]["command"]
    assert baseline[baseline.index("--gotp-weight")+1]=="0"
    assert gotp[gotp.index("--gotp-weight")+1]=="0.15"


def test_preflight_fails_closed_on_misalignment_and_leakage(tmp_path):
    root=fixture_dataset(tmp_path)
    (root/"metadata"/"ignored_train.json").write_text("{}")
    with pytest.raises(ValueError,match="Image IDs differ"):
        preflight(root)
    fixture_dataset(tmp_path)  # replenish the metadata
    (root/"images"/"val"/"sample_train.jpg").write_bytes(b"same name")
    (root/"labels"/"val"/"sample_train.txt").write_text("")
    import json
    for typ in ("ignored","associated"):
        p=root/"metadata"/f"{typ}_val.json"
        original=json.loads(p.read_text())
        original["sample_train"]=[]
        p.write_text(json.dumps(original))
    with pytest.raises(ValueError,match="overlap"):
        preflight(root)


def test_preflight_rejects_corrupt_labels_and_box_count(tmp_path):
    root=fixture_dataset(tmp_path)
    labels=root/"labels"/"train"/"sample_train.txt"
    labels.write_text("0 0.01 0.50 0.30 0.30\n")
    with pytest.raises(ValueError,match="Out-of-bounds"):
        preflight(root)
    labels.write_text("")
    with pytest.raises(ValueError,match="box count"):
        preflight(root)


def test_plan_rejects_duplicate_seeds_unknown_experiments(tmp_path):
    root=fixture_dataset(tmp_path)
    p=preflight(root)
    m=read_matrix(fixture_matrix(tmp_path))
    with pytest.raises(ValueError,match="Seeds"):
        build_plan(p,m,["B1_IGN"],[1,1],imgsz=640,batch=4,epochs=1,
                   device="cpu",project=tmp_path)
    with pytest.raises(ValueError,match="Unknown"):
        build_plan(p,m,["not-a-real-experiment"],[0],imgsz=640,batch=4,epochs=1,
                   device="cpu",project=tmp_path)


def test_preflight_does_not_parse_image_pixels(tmp_path):
    root=fixture_dataset(tmp_path)
    # Data gate validates existence, labels and metadata only. Actual JPG
    # decode must still be checked by the real dataloader training tests.
    assert preflight(root)["checked"]

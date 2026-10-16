"""GOTP CSV diagnostics: matching coverage and active-pair rates."""
import csv
from types import SimpleNamespace
import torch
import pytest
from crowdweave26.telemetry import gotp_totals, record_gotp_epoch


def fake_trainer(directory):
    arr=torch.tensor([1.,4.,3.,2.,1.,0.30],dtype=torch.float64)
    return SimpleNamespace(
        model=SimpleNamespace(criterion=SimpleNamespace(
            one2many=SimpleNamespace(gotp_running=arr.clone()),
            one2one=SimpleNamespace(gotp_running=arr.clone()))),
        save_dir=directory,rank=-1,epoch=0)


def test_gotp_epoch_stats_and_ratios(tmp_path):
    t=fake_trainer(tmp_path)
    assert gotp_totals(t.model)["one2many"]["violating_pairs"] == 2.0
    record_gotp_epoch(t)
    t.epoch=1
    t.model.criterion.one2many.gotp_running*=2
    t.model.criterion.one2one.gotp_running*=2
    record_gotp_epoch(t)
    with (tmp_path/"gotp_pairs.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==4
    assert [r["epoch"] for r in rows] == ["1","1","2","2"]
    assert all(int(r["nearby_pairs"])==3 for r in rows)
    assert all(float(r["violating_pair_fraction"])==pytest.approx(2/3) for r in rows)
    assert all(float(r["mean_raw_loss_per_call"])==pytest.approx(.30) for r in rows)


def test_missing_pair_denominator_does_not_fake_zero_rate(tmp_path):
    t=fake_trainer(tmp_path)
    for branch in ("one2many","one2one"):
        t.model.criterion.__dict__[branch].gotp_running.zero_()
    record_gotp_epoch(t)
    with (tmp_path/"gotp_pairs.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2
    assert rows[0]["violating_pair_fraction"] == ""
    assert rows[0]["mean_raw_loss_per_call"] == ""


def test_non_gotp_and_ddp_secondary_rank_no_logging(tmp_path):
    t=SimpleNamespace(model=SimpleNamespace(criterion=SimpleNamespace()),
                      save_dir=tmp_path,rank=-1,epoch=0)
    record_gotp_epoch(t)
    assert not (tmp_path/"gotp_pairs.csv").exists()
    t=fake_trainer(tmp_path)
    t.rank=1
    record_gotp_epoch(t)
    assert not (tmp_path/"gotp_pairs.csv").exists()

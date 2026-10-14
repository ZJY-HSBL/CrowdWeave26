"""Pure Python per-epoch CRA CSV report tests."""
import csv
from types import SimpleNamespace
from crowdweave26.telemetry import cra_totals, record_cra_epoch


def _fake_trainer(directory):
    a = SimpleNamespace(cumulative_cra_stats={
        "calls": 1, "orphans_before": 3, "orphans_after": 1,
        "recovered": 2, "changed_anchors": 4
    })
    criterion = SimpleNamespace(one2many=SimpleNamespace(assigner=a),
                                one2one=SimpleNamespace(assigner=a))
    return SimpleNamespace(model=SimpleNamespace(criterion=criterion),
                           save_dir=directory, rank=-1, epoch=0)


def test_epoch_csv_and_incremental_deltas(tmp_path):
    trainer = _fake_trainer(tmp_path)
    assert set(cra_totals(trainer.model)) == {"one2many", "one2one"}
    record_cra_epoch(trainer)
    trainer.epoch = 1
    c = trainer.model.criterion.one2many.assigner.cumulative_cra_stats
    c.update({"calls": 2, "orphans_before": 6, "orphans_after": 2,
              "recovered": 4, "changed_anchors": 8})
    record_cra_epoch(trainer)
    with (tmp_path / "cra_coverage.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert [r["epoch"] for r in rows] == ["1", "1", "2", "2"]
    assert all(r["calls"] == "1" and r["recovered"] == "2" for r in rows)


def test_standard_assignment_produces_no_csv(tmp_path):
    trainer = SimpleNamespace(model=SimpleNamespace(criterion=SimpleNamespace()),
                              save_dir=tmp_path, rank=-1, epoch=0)
    record_cra_epoch(trainer)
    assert not (tmp_path / "cra_coverage.csv").exists()


def test_nonmain_rank_is_not_logged(tmp_path):
    trainer = _fake_trainer(tmp_path)
    trainer.rank = 1
    record_cra_epoch(trainer)
    assert not (tmp_path / "cra_coverage.csv").exists()

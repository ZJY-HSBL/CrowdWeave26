"""Pinned-upstream contract tests, skipped when Ultralytics is unavailable."""
import pytest
pytest.importorskip("ultralytics", reason="Pinned YOLO26 is optional in local unit environment")
from scripts.smoke_integration import check


@pytest.mark.parametrize("mode", ("none", "scad", "cfsr", "both"))
def test_official_yolo26_contract(mode):
    result = check(mode)
    assert result["grad_ok"]
    assert result["params"] > 0

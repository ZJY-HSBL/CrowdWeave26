import pytest
from crowdweave.cli import build_parser, run


def test_disallow_non_p2_experiment():
    args = build_parser().parse_args(["train", "--data", "test.yaml", "--variant", "n", "--mechanism", "scad"])
    with pytest.raises(ValueError, match="P2"):
        run(args)


def test_disallow_mechanism_on_val():
    args = build_parser().parse_args(["val", "--data", "test.yaml", "--weights", "example.pt", "--mechanism", "both"])
    with pytest.raises(ValueError, match="checkpoint"):
        run(args)

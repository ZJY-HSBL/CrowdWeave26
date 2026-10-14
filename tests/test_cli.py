import pytest
from crowdweave.cli import MODELS, build_parser, run


def test_variants():
    assert set(MODELS) == {"n", "s", "n-p2", "s-p2"}


def test_val_requires_checkpoint():
    args = build_parser().parse_args(["val", "--data", "crowdhuman.yaml"])
    with pytest.raises(ValueError, match="requires --weights"):
        run(args)


def test_train_disallows_nms_free():
    args = build_parser().parse_args(["train", "--data", "crowdhuman.yaml", "--nms-free"])
    with pytest.raises(ValueError, match="validation only"):
        run(args)

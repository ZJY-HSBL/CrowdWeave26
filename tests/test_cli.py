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



def test_invalid_gotp_weight_rejected_before_loading_yolo():
    args = build_parser().parse_args([
        "train", "--data", "fake.yaml", "--gotp-weight", "-0.1"
    ])
    with pytest.raises(ValueError, match="GOTP"):
        run(args)


def test_validation_gotp_weight_rejected():
    args = build_parser().parse_args([
        "val", "--data", "fake.yaml", "--weights", "fake.pt", "--gotp-weight", "0.2"
    ])
    with pytest.raises(ValueError, match="training only"):
        run(args)


def test_invalid_gotp_population_cap_rejected():
    args = build_parser().parse_args([
        "train", "--data", "fake.yaml", "--gotp-max-gt", "1"
    ])
    with pytest.raises(ValueError, match="max-gt"):
        run(args)

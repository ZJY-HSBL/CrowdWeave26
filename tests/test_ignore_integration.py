"""Pinned upstream tests for geometrically synchronized ignored negatives."""
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("ultralytics")
from ultralytics.cfg import get_cfg
from crowdweave26.ignore import CrowdHumanIgnoredDataset, ignored_anchor_mask, ignored_feature_mask


def test_mask_multiple_images_and_anchor_geometry():
    batch = {
        "img": torch.zeros(2,3,64,64),
        "ignore_bboxes": torch.tensor([[.25,.25,.3,.3],[.75,.75,.3,.3]]),
        "ignore_batch_idx": torch.tensor([0.,1.]),
    }
    pts = torch.tensor([[16.,16.],[48.,48.],[1.,1.]])
    mask = ignored_anchor_mask(batch,pts,torch.ones(3,1))
    assert mask.tolist() == [[True,False,False],[False,True,False]]
    feature = ignored_feature_mask(batch,(4,4),device=pts.device)
    assert feature.shape == (2,1,4,4)
    assert feature[0,0,1,1] and feature[1,0,3,3]


def test_missing_ignore_fields_returns_all_true_background():
    b={"img": torch.zeros(1,3,64,64)}
    pts=torch.tensor([[16.,16.],[48.,48.]])
    assert ignored_anchor_mask(b,pts,torch.ones(2,1)).sum() == 0


def test_dataset_keeps_native_augmented_ignore_and_filters_class(tmp_path):
    random.seed(17)
    np.random.seed(17)
    train=tmp_path/"images"/"train"
    labels=tmp_path/"labels"/"train"
    meta=tmp_path/"metadata"
    for x in (train,labels,meta):
        x.mkdir(parents=True)
    Image.new("RGB",(96,96), (90,90,90)).save(train/"sample.jpg")
    (labels/"sample.txt").write_text("0 0.2 0.2 0.15 0.15\n")
    (meta/"ignored_train.json").write_text(json.dumps({
        "sample":[{"fbox":[26,15,40,58],"extra":{"ignore":1}}]
    }))
    args=get_cfg(overrides={
        "imgsz":96,"mosaic":1.0,"scale":0.0,"translate":0.0,
        "fliplr":0.0,"flipud":0.0,"copy_paste":0.0,"mixup":0.0,"cutmix":0.0
    })
    ds = CrowdHumanIgnoredDataset(
        img_path=str(train), imgsz=96, batch_size=1,
        cache=False, augment=True, hyp=args, rect=False,
        stride=32, pad=0.0, task="detect",
        data={"path":str(tmp_path),"nc":1,"names":{0:"person"}}
    )
    obj=ds[0]
    assert (obj["cls"]>=0).all(), "ignored annotations leaked into positive targets"
    assert len(obj["ignore_bboxes"])>=1, "mosaic must retain geometry of ignored regions"
    assert len(obj["ignore_bboxes"])==len(obj["ignore_batch_idx"])
    assert ((obj["ignore_bboxes"]>=0)&(obj["ignore_bboxes"]<=1)).all()
    batched=ds.collate_fn([obj,obj])
    assert batched["img"].shape[0]==2
    assert (batched["ignore_batch_idx"]==1).any()
    assert (batched["ignore_batch_idx"]==0).any()


def test_fliplr_transforms_ignore_center_with_valid_box(tmp_path):
    train=tmp_path/"images"/"train"
    labels=tmp_path/"labels"/"train"
    meta=tmp_path/"metadata"
    for x in (train,labels,meta): x.mkdir(parents=True)
    Image.new("RGB",(96,96)).save(train/"sample.jpg")
    (labels/"sample.txt").write_text("0 0.20 0.25 0.10 0.10\n")
    (meta/"ignored_train.json").write_text(json.dumps({
        "sample":[{"fbox":[19.2,19.2,19.2,19.2]}]
    }))
    args=get_cfg(overrides={
        "imgsz":96,"mosaic":0.0,"scale":0.0,"translate":0.0,
        "fliplr":1.0,"flipud":0.0,"copy_paste":0.0,"mixup":0.0,"cutmix":0.0
    })
    ds=CrowdHumanIgnoredDataset(
        img_path=str(train), imgsz=96, batch_size=1,
        cache=False, augment=True, hyp=args, rect=False,
        stride=32, pad=0, task="detect",
        data={"path":str(tmp_path),"nc":1,"names":{0:"person"}}
    )
    result=ds[0]
    # Original ignore center: .3 in X; native fliplr maps it to .7.
    assert result["ignore_bboxes"][0,0].item() == pytest.approx(.7, abs=.02)
    assert result["cls"].shape[0] == 1


def test_ignore_aware_loss_only_reduces_unmatched_background_bce():
    from crowdweave26.integration import CrowdSupervisedDetectionModel
    from crowdweave26.ignore_loss import IgnoreAwareDetectionLoss
    model = CrowdSupervisedDetectionModel("yolo26n-p2.yaml", nc=1, verbose=False)
    model.args = get_cfg()
    model.crowd_ignore_aware = True
    model.train()
    b={
        "img":torch.rand(1,3,96,96),
        "batch_idx":torch.tensor([0.]),
        "cls":torch.tensor([[0.]]),
        "bboxes":torch.tensor([[.12,.12,.14,.14]]),
        "ignore_bboxes":torch.tensor([[.65,.65,.55,.55]]),
        "ignore_batch_idx":torch.tensor([0.])
    }
    raw=model(b["img"])
    ignored_loss, _=model.loss(b,raw)
    no_ignore={k:v for k,v in b.items() if not k.startswith("ignore_")}
    baseline_loss, _=model.loss(no_ignore,raw)
    assert isinstance(model.criterion.one2many, IgnoreAwareDetectionLoss)
    assert ignored_loss.numel()==3
    assert ignored_loss[1] < baseline_loss[1]
    assert torch.allclose(ignored_loss[[0,2]],baseline_loss[[0,2]],atol=1e-5)
    ignored_loss.sum().backward()
    assert model.model[-1].cv2[0][0].conv.weight.grad is not None

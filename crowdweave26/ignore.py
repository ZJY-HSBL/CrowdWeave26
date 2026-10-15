"""Geometry-synchronized CrowdHuman ignore regions for pinned Ultralytics YOLO26.

Ignored boxes use class -1 **only during augmentation**. Mosaic, affine
transforms and flips therefore act on both valid and ignored boxes in the same
native Instances object. Format then separates them before model training.
Bounding rectangles are a conservative proxy, not pixel-wise ignore masks.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Format

from crowdweave.data import xywh_to_yolo


class CrowdIgnoreFormat(Format):
    """Strip class -1 boxes after native Ultralytics geometric augmentation."""

    def __call__(self, labels):
        formatted = super().__call__(labels)
        cls = formatted["cls"].reshape(-1)
        box = formatted["bboxes"]
        if cls.shape[0] != box.shape[0]:
            raise ValueError("Native augmentation desynchronized ignore classes/boxes")
        ignored = cls < 0
        keep = ~ignored
        formatted["ignore_bboxes"] = box[ignored].clone()
        formatted["ignore_batch_idx"] = torch.zeros(int(ignored.sum()), dtype=torch.float32)
        formatted["cls"] = formatted["cls"][keep]
        formatted["bboxes"] = box[keep]
        formatted["batch_idx"] = formatted["batch_idx"][keep]
        return formatted


class CrowdHumanIgnoredDataset(YOLODataset):
    """Append ignored full-body rectangles to the instance augmentation stream."""

    format_class = CrowdIgnoreFormat

    def get_labels(self):
        labels = super().get_labels()  # native YOLO cache verification, before -1 insertion
        if self.single_cls:
            raise ValueError("ignore-aware training does not support single_cls=True")
        if self.data.get("nc") != 1:
            raise ValueError("CrowdHuman ignore-aware training expects one person class")
        meta_file = Path(self.data["path"]) / "metadata" / "ignored_train.json"
        if not meta_file.is_file():
            raise FileNotFoundError(
                f"Missing {meta_file}; convert original CrowdHuman ODGT first. "
                "Refusing training without a synchronized ignore source."
            )
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("ignored_train.json must map image IDs to annotations")
        seen = set()
        for label in labels:
            image_id = Path(label["im_file"]).stem
            if image_id in seen or image_id not in metadata:
                raise ValueError(f"Duplicate or missing ignore metadata for {image_id}")
            seen.add(image_id)
            height, width = map(int, label["shape"])
            extra = []
            for row in metadata[image_id]:
                # Image metadata is original-space (x,y,w,h), not augmented.
                box = xywh_to_yolo(row.get("fbox"), width, height)
                if box is not None:
                    extra.append(box)
            if extra:
                old = label["bboxes"]
                cls = label["cls"]
                label["bboxes"] = np.concatenate(
                    (old, np.asarray(extra, dtype=np.float32)), axis=0
                )
                label["cls"] = np.concatenate(
                    (cls, -np.ones((len(extra), 1), dtype=np.float32)), axis=0
                )
        return labels

    @staticmethod
    def collate_fn(batch):
        """Collate ignore boxes and indices without presenting them to TAL as GT."""
        copied = [dict(sample) for sample in batch]
        ignored = [row.pop("ignore_bboxes") for row in copied]
        indices = [row.pop("ignore_batch_idx") for row in copied]
        merged = YOLODataset.collate_fn(copied)
        merged["ignore_bboxes"] = torch.cat(ignored, dim=0)
        merged["ignore_batch_idx"] = torch.cat(
            [index + i for i, index in enumerate(indices)], dim=0
        )
        return merged


def ignored_anchor_mask(batch, anchor_points, stride_tensor, *, device=None):
    """Boolean [B,N] mask for anchor centers inside augmented ignored boxes.

    All coordinates are from the SAME augmented, optionally multi-scale-resized
    batch. Box data are normalized xywh; centers are model pixel coordinates.
    """
    if "img" not in batch:
        raise ValueError("batch must contain images")
    images = batch["img"]
    bs, _, h, w = images.shape
    if device is None:
        device = anchor_points.device
    centers = (anchor_points * stride_tensor).to(device=device, dtype=torch.float32)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("anchor points must be N,2")
    result = torch.zeros((bs, centers.shape[0]), dtype=torch.bool, device=device)
    boxes = batch.get("ignore_bboxes")
    inds = batch.get("ignore_batch_idx")
    if boxes is None and inds is None:
        return result
    if boxes is None or inds is None:
        raise ValueError("ignore boxes and indices must both be present")
    boxes = boxes.reshape(-1, 4).to(device=device, dtype=torch.float32)
    inds = inds.reshape(-1).to(device=device)
    if inds.shape[0] != boxes.shape[0]:
        raise ValueError("ignore box count differs from image indices")
    if len(inds) and (not bool(((inds >= 0) & (inds < bs) &
                              (inds == inds.round())).all()) or
                      not bool(torch.isfinite(boxes).all())):
        raise ValueError("invalid ignore annotations in augmented batch")
    scale = boxes.new_tensor([w, h, w, h])
    xywh = boxes * scale
    xyxy = torch.stack((xywh[:, 0]-xywh[:, 2]/2,
                        xywh[:, 1]-xywh[:, 3]/2,
                        xywh[:, 0]+xywh[:, 2]/2,
                        xywh[:, 1]+xywh[:, 3]/2), dim=-1)
    for i in range(bs):
        for x1, y1, x2, y2 in xyxy[inds == i]:
            result[i] |= (
                (centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
                (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
            )
    return result


def ignored_feature_mask(batch, shape, *, device):
    """Project ignore rectangles to the CFSR map's anchor-center grid."""
    height, width = shape
    img_h, img_w = batch["img"].shape[-2:]
    if height < 1 or width < 1:
        raise ValueError("feature map size invalid")
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing="ij")
    pts = torch.stack(((xx.flatten()+0.5)*img_w/width,
                       (yy.flatten()+0.5)*img_h/height), -1).float()
    return ignored_anchor_mask(batch, pts, torch.ones((len(pts),1),device=device),
                               device=device).reshape(-1,1,height,width)

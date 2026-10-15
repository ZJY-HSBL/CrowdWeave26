"""Pinned YOLO26 inference contract for CrowdHuman exporter arguments.

These random-weight predictions test the inference API only, not accuracy.
"""
import pytest
from PIL import Image
pytest.importorskip("ultralytics")
from ultralytics import YOLO


@pytest.mark.parametrize("nms", [True, False])
def test_original_resolution_prediction_accepts_nms_flag(tmp_path, nms):
    image = tmp_path/"test.jpg"
    Image.new("RGB", (96, 96), (33, 44, 55)).save(image)
    model = YOLO("yolo26n-p2.yaml")
    output = model.predict(
        source=str(image), device="cpu", imgsz=96, conf=0.001,
        iou=0.7, max_det=1000, nms=nms, verbose=False
    )
    assert len(output) == 1
    assert tuple(output[0].orig_shape) == (96, 96)
    assert output[0].boxes is not None

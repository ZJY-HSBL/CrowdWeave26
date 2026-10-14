"""Standalone research components for CrowdWeave26; not yet installed into YOLO26."""
from .modules import SCADDownsample, CFSRNeck
from .losses import gotp_pairwise_loss, conflict_supervision_loss
from .conflict import rasterize_overlap

__all__ = ["SCADDownsample", "CFSRNeck", "gotp_pairwise_loss", "conflict_supervision_loss", "rasterize_overlap"]
__version__ = "0.1.1"

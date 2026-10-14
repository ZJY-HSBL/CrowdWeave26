"""Ground-truth rectangle overlap raster, for training supervision only."""
from __future__ import annotations
import torch


def rasterize_overlap(boxes: torch.Tensor, image_hw: tuple[int,int],
                      output_hw: tuple[int,int]) -> torch.Tensor:
    """Return HxW float mask of cells covered by >=2 full-body GT rectangles.

    This describes rectangular annotation overlap, not true visible occlusion.
    Integer difference array: O(N + output H*W).
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be Nx4 xyxy")
    ih, iw = image_hw
    h, w = output_hw
    if min(ih, iw, h, w) <= 0:
        raise ValueError("dimensions must be positive")
    with torch.no_grad():
        b = boxes.detach().float()
        b = b[torch.isfinite(b).all(dim=1)]
        if b.numel() == 0:
            return torch.zeros((h, w), dtype=torch.float32, device=boxes.device)
        x0 = torch.floor(b[:,0]/iw*w).long().clamp(0,w)
        y0 = torch.floor(b[:,1]/ih*h).long().clamp(0,h)
        x1 = torch.ceil(b[:,2]/iw*w).long().clamp(0,w)
        y1 = torch.ceil(b[:,3]/ih*h).long().clamp(0,h)
        valid = (x1>x0)&(y1>y0)
        x0,x1,y0,y1 = x0[valid],x1[valid],y0[valid],y1[valid]
        diff = torch.zeros((h+1)*(w+1),dtype=torch.int32,device=boxes.device)
        one = torch.ones_like(x0,dtype=diff.dtype)
        s = w+1
        diff.scatter_add_(0,y0*s+x0,one)
        diff.scatter_add_(0,y0*s+x1,-one)
        diff.scatter_add_(0,y1*s+x0,-one)
        diff.scatter_add_(0,y1*s+x1,one)
        return (diff.view(h+1,w+1).cumsum(0).cumsum(1)[:h,:w]>=2).float()

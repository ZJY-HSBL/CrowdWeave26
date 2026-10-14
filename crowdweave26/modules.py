"""Standalone learnable SCAD and CFSR prototypes; no YOLO parser integration yet."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class SCADDownsample(nn.Module):
    """Dual-path anti-aliased stride-2 feature extraction with image-derived conflict gate.

    A gate cannot be interpreted as a crowd map without GT-overlap supervision.
    Inference consumes features only, never ground-truth information.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if min(in_channels, out_channels) <= 0:
            raise ValueError("channels must be positive")
        self.detail = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(out_channels), nn.SiLU())
        self.smooth_proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                         nn.BatchNorm2d(out_channels), nn.SiLU())
        hidden = max(8, min(64, in_channels // 4))
        self.conflict = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1),
                                      nn.SiLU(), nn.Conv2d(hidden, 1, 1))
        k = torch.tensor([1., 2., 1.])
        self.register_buffer("blur_kernel", (torch.outer(k, k)/16).view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor, *, return_conflict_logits: bool = False):
        if x.ndim != 4:
            raise ValueError("x must be B,C,H,W")
        detail = self.detail(x)
        smoothed = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"),
                            self.blur_kernel.to(dtype=x.dtype).expand(x.shape[1], 1, 3, 3),
                            stride=2, groups=x.shape[1])
        smooth = self.smooth_proj(smoothed)
        logits = F.interpolate(self.conflict(x), size=detail.shape[2:],
                               mode="bilinear", align_corners=False)
        gate = torch.sigmoid(logits)
        result = gate * detail + (1.0 - gate) * smooth
        return (result, logits) if return_conflict_logits else result


class CFSRNeck(nn.Module):
    """Conflict-conditioned P2/P3/P4 routing to P3-sized output."""
    def __init__(self, channels: tuple[int, int, int], out_channels: int):
        super().__init__()
        if len(channels) != 3 or min(*channels, out_channels) <= 0:
            raise ValueError("three positive channel sizes and output required")
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.SiLU()) for c in channels
        ])
        self.conflict = nn.Conv2d(channels[0], 1, 3, padding=1)
        self.router = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1),
                                    nn.SiLU(), nn.Conv2d(out_channels, 3, 1))
        self.conflict_scale = nn.Parameter(torch.tensor(0.25))
        self.out = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(out_channels), nn.SiLU())

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                *, return_aux: bool = False):
        if len(features) != 3:
            raise ValueError("features must be P2,P3,P4")
        p2, p3, p4 = features
        hw = p3.shape[2:]
        projected = [self._resize(conv(inp), hw) for conv, inp in zip(self.projections, features)]
        logits = self._resize(self.conflict(p2), hw)
        density = logits.sigmoid()
        route_logits = self.router(torch.cat(projected, dim=1))
        bias = torch.cat((density, torch.zeros_like(density), -density), dim=1)
        weights = torch.softmax(route_logits + self.conflict_scale * bias, dim=1)
        fused = sum(z * weights[:, i:i+1] for i, z in enumerate(projected))
        output = self.out(fused)
        return (output, logits, weights) if return_aux else output

    @staticmethod
    def _resize(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        return x if x.shape[2:] == hw else F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

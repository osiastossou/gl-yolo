"""
TinyYOLO — TOSA (Tiny Object Spatial Attention) module.
=======================================================
Drop-in replacements for use in YOLOv11.

  • TOSA            — channel + spatial attention with residual blend
  • AvgPoolDWConv   — soft downsampler replacing nn.MaxPool2d(2, stride=2)

Integration
-----------
    from tosa_attention import register_tosa
    register_tosa()          # call BEFORE importing or constructing YOLO
    from ultralytics import YOLO
    model = YOLO("tinyyolo.yaml")
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class _DWConv(nn.Module):
    """Depthwise conv + BN + SiLU."""

    def __init__(self, c: int, k: int = 3, s: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, stride=s, padding=k // 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.dw(x)))


class _ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel gate (CBAM variant). Both global avg-pool and max-pool statistics share the same
    MLP; their outputs are summed before the sigmoid gate.

    AdaptiveMaxPool2d is intentionally avoided: it has no deterministic CUDA backward implementation and triggers a
    UserWarning when torch.use_deterministic_algorithms(True) is set (common on Kaggle). Instead we use amax() over
    spatial dims, which is fully deterministic and produces the identical mathematical result.
    """

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = _make_divisible(max(c // r, 8))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # deterministic ✓
        # max_pool replaced by amax() in forward      deterministic ✓
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg = self.mlp(self.avg_pool(x).view(B, C))
        # amax over H,W — deterministic on all backends, same result as
        # AdaptiveMaxPool2d(1) followed by .view(B, C)
        mx = self.mlp(x.amax(dim=(2, 3)))
        return x * torch.sigmoid(avg + mx).view(B, C, 1, 1)


class _SpatialAttention(nn.Module):
    """CBAM-style spatial gate with a 7×7 conv. Large kernel leverages background context — critical for tiny objects
    where the surrounding scene (sky, road) is more discriminative than the few object pixels themselves.
    """

    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))


# ---------------------------------------------------------------------------
# TOSA
# ---------------------------------------------------------------------------


class TOSA(nn.Module):
    """Tiny Object Spatial Attention.

    Channel attention → spatial attention → residual blend:
        output = x + α * (attended − x)

    α is a learnable scalar initialized at 0.5 (near-identity at training start, lets gradients flow through the skip
    path).

    WHY LAZY INIT
    -------------
    Ultralytics parse_model passes YAML args into __init__ *before* applying width_multiple scaling. The actual runtime
    channel count (x.shape[1]) may therefore differ from any arg in __init__. Sub-modules that depend on the channel
    count (_ChannelAttention, _SpatialAttention) are built on the first forward() call from the real tensor shape and
    cached.

    __init__ accepts *args / **kwargs so it works regardless of how many positional arguments the local Ultralytics
    version of parse_model passes.
    """

    def __init__(self, *args, r: int = 16, k: int = 7, **kwargs):
        super().__init__()
        self._r = r
        self._k = k
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self._built_c: int = -1

    # ------------------------------------------------------------------
    def _build(self, c: int, device: torch.device) -> None:
        self.ch_att = _ChannelAttention(c, self._r).to(device)
        self.sp_att = _SpatialAttention(self._k).to(device)
        self._built_c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if self._built_c != c:
            self._build(c, x.device)
        attended = self.sp_att(self.ch_att(x))
        return x + self.alpha * (attended - x)


# ---------------------------------------------------------------------------
# AvgPoolDWConv
# ---------------------------------------------------------------------------


class AvgPoolDWConv(nn.Module):
    """Soft 2× downsampler: AvgPool2d(2) → DWConv(3×3, stride=1).

    Replaces nn.MaxPool2d(2, stride=2) throughout the backbone.

    MaxPool keeps only the peak activation in each 2×2 window, effectively erasing the spatial structure of objects that
    are only a few pixels wide. AvgPool preserves distributed signal energy; the subsequent 3×3 DWConv re-synthesises
    local patterns from it at negligible extra cost.

    Like TOSA, uses lazy init so it is robust to any number of args from parse_model and to Ultralytics' width_multiple
    channel scaling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self._built_c: int = -1

    # ------------------------------------------------------------------
    def _build(self, c: int, device: torch.device) -> None:
        self.dw = _DWConv(c, k=3, s=1).to(device)
        self._built_c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        c = x.shape[1]
        if self._built_c != c:
            self._build(c, x.device)
        return self.dw(x)

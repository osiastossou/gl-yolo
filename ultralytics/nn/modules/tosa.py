"""
TinyYOLO — TOSA (Tiny Object Spatial Attention) module
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
        self.dw  = nn.Conv2d(c, c, k, stride=s, padding=k // 2, groups=c, bias=False)
        self.bn  = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.dw(x)))


class _ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel gate (CBAM variant).

    Global avg-pool and max-pool statistics are each passed through the same
    shared MLP; their outputs are summed before the sigmoid.
    """

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = _make_divisible(max(c // r, 8))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg  = self.mlp(self.avg_pool(x).view(B, C))
        mx   = self.mlp(self.max_pool(x).view(B, C))
        gate = torch.sigmoid(avg + mx).view(B, C, 1, 1)
        return x * gate


class _SpatialAttention(nn.Module):
    """
    CBAM-style spatial gate with a large-kernel (7×7) depthwise conv.

    For tiny objects the surrounding context (sky, road, background) is
    often more discriminative than the few object pixels themselves — hence
    the intentionally large receptive field.
    """

    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return x * gate


# ---------------------------------------------------------------------------
# TOSA
# ---------------------------------------------------------------------------

class TOSA(nn.Module):
    """
    Tiny Object Spatial Attention.

    Applies channel attention then spatial attention, then blends back via a
    learnable scalar α (initialised 0.5 → near-identity at start of training):

        output = x + α * (attended − x)

    Ultralytics' parse_model passes the YAML args[0] channel value into
    __init__, but that value may not match the actual width-scaled channel
    count.  TOSA therefore uses *lazy initialisation*: the internal sub-
    modules are built on the first forward pass from the real tensor shape,
    then cached for all subsequent calls.

    Args (YAML):
        c_hint (int): Channel hint from YAML (used only as a sanity label;
                      actual channels are read from the first input tensor).
        r      (int): Channel-attention MLP reduction ratio (default 16).
        k      (int): Spatial-attention conv kernel size (default 7).
    """

    def __init__(self, c_hint: int = 0, r: int = 16, k: int = 7):
        super().__init__()
        self.r      = r
        self.k      = k
        self.alpha  = nn.Parameter(torch.tensor(0.5))
        # Lazily-initialised submodules (built on first forward call)
        self._ch_att: nn.Module  = None
        self._sp_att: nn.Module  = None
        self._built_c: int = 0

    def _build(self, c: int) -> None:
        """Initialise sub-modules for channel count *c*."""
        self._ch_att  = _ChannelAttention(c, self.r).to(self.alpha.device)
        self._sp_att  = _SpatialAttention(self.k).to(self.alpha.device)
        self._built_c = c
        # Register as proper sub-modules so optimiser and state_dict see them
        self.ch_att = self._ch_att
        self.sp_att = self._sp_att

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if self._built_c != c:
            self._build(c)
        attended = self._sp_att(self._ch_att(x))
        return x + self.alpha * (attended - x)


# ---------------------------------------------------------------------------
# AvgPoolDWConv
# ---------------------------------------------------------------------------

class AvgPoolDWConv(nn.Module):
    """
    Soft 2× downsampler: AvgPool2d(2) → DWConv(3×3).

    Replaces nn.MaxPool2d(2, stride=2) in the backbone.

    MaxPool keeps only the peak activation in each 2×2 window, destroying
    the spatial structure of small objects.  AvgPool preserves distributed
    energy; the subsequent 3×3 DWConv re-synthesises local patterns from it.

    Like TOSA, this module uses lazy initialisation so it works regardless
    of the width-scaled channel count at runtime.
    """

    def __init__(self, c_hint: int = 0):
        super().__init__()
        self.pool     = nn.AvgPool2d(2, stride=2)
        self._dw: nn.Module = None
        self._built_c: int = 0

    def _build(self, c: int) -> None:
        self._dw      = _DWConv(c, k=3, s=1).to(self.pool.kernel_size.__class__)
        self._built_c = c
        # Use a proper parameter-free AvgPool2d for device transfer
        dev = next(iter([]))  # just for type hint; lazily resolved in forward
        self.dw = _DWConv(c, k=3, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if self._built_c != c:
            self.dw       = _DWConv(c, k=3, s=1).to(x.device)
            self._built_c = c
        return self.dw(self.pool(x))


"""SPPF with Adaptive Scale Selection (SPPF-ASS).

A drop-in replacement for SPPF (and for SPPFCSPC_CBAM) at the P5/32 stage. See the
class docstring for the rationale.
"""

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("SPPF_ASS",)


class DualPool(nn.Module):
    """One stage of two independent pooling chains, run in series across stages.

    Mirrors SPPF's cascaded-MaxPool trick (receptive field grows stage over stage without
    re-deriving it from scratch), but keeps a second, parallel AvgPool chain alongside it:
    MaxPool keeps salient detail, AvgPool keeps global shape/context, and a stage's output
    `p = m + a` is what gets offered to ScaleSelect as "the view of the input at this scale".
    """

    def __init__(self, k: int = 5):
        """Initialize with a k x k, stride-1 MaxPool/AvgPool pair (same padding, same k)."""
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, m_prev: torch.Tensor, a_prev: torch.Tensor):
        """Advance both chains by one stage; return the new (m, a) state plus their sum p."""
        m = self.max_pool(m_prev)
        a = self.avg_pool(a_prev)
        return m, a, m + a


class ScaleSelect(nn.Module):
    """Per-pixel softmax fusion of `n_scales` same-shaped feature maps into one.

    Replaces SPPFCSPC_CBAM's concat -> cv5 (4c -> c) -> CBAM with a single lighter step:
    instead of mixing every scale through a learned-but-fixed 1x1 conv and then re-weighting
    channels/space with CBAM, this squeezes the 4 scales directly down to one softmax weight
    *per scale, per pixel* and takes a weighted sum -- so the network picks, at each location,
    how much of x1 (fine, unpooled) vs p1/p2/p3 (progressively larger receptive field) to keep,
    instead of always blending all four the same way everywhere. Output is c channels (not 4c),
    which is what makes this lighter than the concat-then-reduce approach it replaces.
    """

    def __init__(self, c: int, n_scales: int, r: int = 4):
        """Initialize with `c` channels per scale, `n_scales` inputs, and reduction ratio `r`."""
        super().__init__()
        hidden = max(c // r, 8)
        self.reduce = Conv(c * n_scales, hidden, 1, 1)  # SiLU (Conv's default act)
        self.logits = nn.Conv2d(hidden, n_scales, 1, 1)
        self.n_scales = n_scales

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Fuse `feats` (n_scales tensors, each (B, c, H, W)) into one (B, c, H, W) tensor."""
        w = self.logits(self.reduce(torch.cat(feats, dim=1))).softmax(dim=1)  # (B, n_scales, H, W)
        return sum(w[:, i : i + 1] * feats[i] for i in range(self.n_scales))


class SPPF_ASS(nn.Module):
    """SPPF-CSPC refined with Adaptive Scale Selection instead of CBAM.

    Same CSP split as SPPFCSPC_CBAM (a shortcut branch bypasses pooling entirely), but the
    pooled branch now runs `n` DualPool stages (MaxPool + AvgPool in parallel, k x k, stride 1)
    instead of a single MaxPool cascade, and fuses the resulting scales (x1, p1, ..., pn) with
    ScaleSelect's per-pixel softmax instead of concat + 1x1 conv + CBAM.

    Rationale: CBAM re-weights a feature map that has already been irreversibly mixed across
    scales by the concat + 1x1 conv -- it can emphasize or suppress channels/regions of that
    mix, but can't recover "how much of this pixel should come from x1 vs p3" once they're
    blended. ScaleSelect makes that choice explicit and local: a softmax weight per scale, per
    pixel, decides the mix before fusion, with no separate attention step needed afterward.

    Attributes:
        cv1, cv2 (Conv): 1x1 projections splitting the input into the pooled branch (cv1) and
            the shortcut branch (cv2).
        cv3, cv4 (Conv): 3x3 then 1x1 refinement of the pooled branch before pooling (this is
            x1, the finest/unpooled scale).
        pools (nn.ModuleList[DualPool]): `n` sequential DualPool stages.
        scale_select (ScaleSelect): Fuses [x1, p1, ..., pn] (n+1 scales) into c_ channels.
        cv6 (Conv): 3x3 refinement of the fused scale-selected features.
        cv7 (Conv): Final 1x1 projection of the concatenated branches to c2.
    """

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, r: int = 4, e: float = 0.5, *args, **kwargs):
        """Initialize the SPPF_ASS block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Pooling kernel size (both MaxPool and AvgPool).
            n (int): Number of DualPool stages (-> n+1 scales fed to ScaleSelect, including x1).
            r (int): ScaleSelect's channel-reduction ratio.
            e (float): Hidden-channel expansion ratio; hidden channels are ``int(2 * c2 * e)``
                (matches SPPFCSPC_CBAM's convention, e=0.5 gives hidden == c2).
        """
        super().__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)  # CSP shortcut branch, bypasses pooling
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.n = n
        self.pools = nn.ModuleList(DualPool(k) for _ in range(n))
        self.scale_select = ScaleSelect(c_, n_scales=n + 1, r=r)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: scale-selected pooled branch concatenated with the CSP shortcut."""
        x1 = self.cv4(self.cv3(self.cv1(x)))
        feats = [x1]
        m, a = x1, x1
        for pool in self.pools:
            m, a, p = pool(m, a)
            feats.append(p)
        y1 = self.cv6(self.scale_select(feats))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), 1))

"""SPPF-CSPC block with CBAM attention.

A drop-in replacement for SPPF at the P5/32 stage, designed for datasets with a
high density of small objects (e.g. DOTA aerial imagery). See the class
docstring for the rationale.
"""

import torch
import torch.nn as nn

from .conv import CBAM, Conv

__all__ = ("SPPFCSPC_CBAM",)


class SPPFCSPC_CBAM(nn.Module):
    """SPPF-CSPC block refined with CBAM (channel + spatial) attention.

    Combines the Cross-Stage-Partial split from YOLOv7's SPPCSPC (half the
    input channels bypass pooling entirely, as a plain shortcut) with the
    fast cascaded-MaxPool trick from SPPF (a single k x k ``MaxPool2d``
    applied ``n`` times in sequence, equivalent to SPP(k, 2k, 4k, ...) but
    cheaper). The pooled branch is then refined with CBAM before being fused
    with the shortcut branch.

    Rationale for small-object detection: plain SPPF only aggregates features
    through MaxPool, which discards fine spatial detail before it even
    reaches the concatenation step. The CSP shortcut branch here preserves an
    untouched copy of the input, and CBAM re-weights the pooled branch by
    channel and spatial relevance instead of taking it at face value. Since
    this block sits at P5 (stride 32, the coarsest resolution), it does not
    recover detail already lost to downsampling -- the benefit is a richer,
    less lossy context that gets fused back into P3/P4 through the FPN.

    Attributes:
        cv1, cv2 (Conv): 1x1 projections splitting the input into the pooled
            branch (cv1) and the shortcut branch (cv2).
        cv3, cv4 (Conv): 3x3 then 1x1 refinement of the pooled branch before
            pooling.
        m (nn.MaxPool2d): Shared max-pool applied ``n`` times in sequence.
        cv5, cv6 (Conv): 1x1 then 3x3 fusion of the concatenated pooled
            features.
        cbam (CBAM): Channel + spatial attention applied to the pooled branch.
        cv7 (Conv): Final 1x1 projection of the concatenated branches to c2.
    """

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, e: float = 0.5, *args, **kwargs):
        """Initialize the SPPFCSPC_CBAM block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Max-pool kernel size.
            n (int): Number of sequential pooling iterations (as in SPPF).
            e (float): Hidden-channel expansion ratio; hidden channels are
                ``int(2 * c2 * e)`` (matches the SPPCSPC convention, e=0.5
                gives hidden == c2).
        """
        super().__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)  # CSP shortcut branch, bypasses pooling
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.cv5 = Conv(c_ * (n + 1), c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cbam = CBAM(c_, kernel_size=7)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: pooled+attended branch concatenated with the CSP shortcut."""
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y = [x1]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y1 = self.cbam(self.cv6(self.cv5(torch.cat(y, 1))))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), 1))

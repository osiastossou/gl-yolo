"""
Ultralytics-compatible implementation of GL-CAB and MB-FPN from:
"Global-Local Attention Mechanism Based Small Object Detection"
Bao Liu, Jinlei Huang — IEEE DDCLS 2023.

HOW TO INTEGRATE INTO ULTRALYTICS
──────────────────────────────────
1. Copy this file into:
       ultralytics/nn/modules/gld.py

2. Export from ultralytics/nn/modules/__init__.py:
       from .gld import GLCAB, MBFPN

3. Register in ultralytics/nn/tasks.py  parse_model():
       Add 'GLCAB' and 'MBFPN' to the modules that receive
       feature-list inputs (same pattern as Concat / C2f).
       See TASKS_PY_PATCH string at the bottom of this file.

4. Reference in your YAML model config (see YAML_EXAMPLE at the bottom).

KEY DESIGN DECISION — Ultralytics compatibility
────────────────────────────────────────────────
Ultralytics calls every module with a SINGLE argument:
    x = m(x)
where x can be:
  • a single Tensor      -> standard conv / attention modules
  • a list of Tensors    -> concat-style modules (e.g. Concat, FPN)

Both GLCAB and MBFPN follow this convention:
  • GLCAB.forward(x)   x : Tensor          (B, C, H, W)
  • MBFPN.forward(x)   x : list of Tensors collected from earlier
                           layers via the YAML 'from' field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────


class ConvBNAct(nn.Module):
    """Conv2d -> BN -> optional ReLU."""

    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# GL-CAB  —  Global and Local Combined Attention Block
# Paper equations (2)-(5)
# ─────────────────────────────────────────────────────────────────────────────


class GLCAB(nn.Module):
    """Single-tensor attention module — drop-in replacement for SE / CBAM.

    forward(x) -> same shape as x.

    Three internal branches (all 1x1 conv, as stated in the paper):
        L(P) — local detail   eq.(2)   Conv -> Hardswish -> Conv -> BN
        Z(P) — local branch   eq.(3)   Conv -> Hardswish -> Conv -> BN
        G(P) — global branch  eq.(4)   GAP  -> Conv -> BN -> ReLU -> Conv -> BN

    Output:
        W(P) = sigmoid( L(P) + Z(P) ) * G(P)          eq.(5)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)

        # L(P) — eq.(2)  BN(Conv(Hs(Conv(P))))
        self.local_detail = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Z(P) — eq.(3)  BN(Conv(Hs(Conv(P))))
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # G(P) — eq.(4)  BN(Conv(ReLU(BN(Conv(GAP(P))))))
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,C,1,1)
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    # ── Ultralytics-compatible: single Tensor in, single Tensor out ──────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """X : (B, C, H, W) -> (B, C, H, W)."""
        L = self.local_detail(x)  # eq.(2)
        Z = self.local_branch(x)  # eq.(3)
        G = self.global_branch(x)  # eq.(4)  shape (B,C,1,1) broadcasts

        weight = self.sigmoid(L + Z)  # sigma(L + Z)
        return weight * G  # element-wise multiply


# ─────────────────────────────────────────────────────────────────────────────
# MB-FPN  —  Multi-Branch Feature Pyramid Network
# Paper equation (1)
# ─────────────────────────────────────────────────────────────────────────────


class MBFPN(nn.Module):
    """Ultralytics-compatible MB-FPN.

    In the YAML config use the 'from' field to feed a list of 10 feature maps:
        [P1, P2, P3, P4, P5,  x1, x2, x3, x4, x5]
         <-- main branch -->   <-- super-res branch -->

    forward(x) receives that list and returns [P1*, P2*, P3*, P4*, P5*].

    Fusion rules — eq.(1):
        P1* = P1
        P2* = Conv3x3(x1 + P1)
        P3* = GL-CAB(P3)
        P4* = Conv3x3(x3 + P3*)
        P5* = Conv3x3(x4 + P4) + P5

    Args:
        ────
        in_channels: int or list[int] of length 10 Channels for [P1, P2, P3, P4, P5, x1, x2, x3, x4, x5]. Pass a single
            int to use the same value for all 10.
        out_channels: unified channel width after lateral projections
        use_glcab: insert GL-CAB at P3 level (default True)
        return_all: return all fused outputs (default False)
    """

    def __init__(
        self,
        in_channels=256,
        out_channels: int = 256,
        use_glcab: bool = True,
        return_all: bool = False,
    ):
        super().__init__()
        self.use_glcab = use_glcab
        self.return_all = return_all
        c = out_channels

        if isinstance(in_channels, int):
            in_channels = [in_channels] * 10
        assert isinstance(in_channels, (list, tuple)) and len(in_channels) == 10, (
            "MBFPN expects 10 input channel values."
        )

        self.in_channels = list(in_channels)

        # Lateral 1x1 projections — main branch (P1..P5)
        self.lat_main = nn.ModuleList([nn.Conv2d(in_channels[i], c, 1, bias=False) for i in range(5)])
        # Lateral 1x1 projections — super-res branch (x1..x5)
        self.lat_branch = nn.ModuleList([nn.Conv2d(in_channels[i + 5], c, 1, bias=False) for i in range(5)])

        if use_glcab:
            self.glcab = GLCAB(c)

        # 3x3 fusion convolutions
        self.fuse_P2 = ConvBNAct(c, c, k=3, p=1)
        self.fuse_P4 = ConvBNAct(c, c, k=3, p=1)
        self.fuse_P5 = ConvBNAct(c, c, k=3, p=1)

    @staticmethod
    def _match_size(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize src to the spatial size of ref (nearest-neighbour)."""
        if src.shape[-2:] != ref.shape[-2:]:
            src = F.interpolate(src, size=ref.shape[-2:], mode="nearest")
        return src

    # ── Ultralytics-compatible: list of Tensors in, list of Tensors out ──────
    def forward(self, x: list) -> list:
        """X : list of 10 Tensors [P1, P2, P3, P4, P5, x1, x2, x3, x4, x5].

        Returns : [P1*, P2*, P3*, P4*, P5*]
        """
        assert len(x) == 10, (
            f"MBFPN.forward() expects a list of 10 tensors, got {len(x)}.\nCheck the 'from' field in your YAML config."
        )

        # Lateral projections
        P = [self.lat_main[i](x[i]) for i in range(5)]  # P1..P5
        X = [self.lat_branch[i](x[i + 5]) for i in range(5)]  # x1..x5

        # GL-CAB on P3 (index 2)
        if self.use_glcab:
            P[2] = self.glcab(P[2])

        # eq.(1) fusion
        # P1* = P1
        P1_star = P[0]

        # P2* = Conv3x3(x1 + P1)
        x1 = self._match_size(X[0], P[0])
        P2_star = self.fuse_P2(x1 + P[0])

        # P3* = GL-CAB(P3)  — already updated in P[2]
        P3_star = P[2]

        # P4* = Conv3x3(x3 + P3*)
        x3 = self._match_size(X[2], P[2])
        P4_star = self.fuse_P4(x3 + P[2])

        # P5* = Conv3x3(x4 + P4) + P5
        x4 = self._match_size(X[3], P[3])
        tmp = self.fuse_P5(x4 + P[3])
        tmp = self._match_size(tmp, P[4])
        P5_star = tmp + P[4]

        return [P1_star, P2_star, P3_star, P4_star, P5_star]


# ─────────────────────────────────────────────────────────────────────────────
# tasks.py patch
# Add this block inside parse_model() in ultralytics/nn/tasks.py
# ─────────────────────────────────────────────────────────────────────────────
TASKS_PY_PATCH = """
# ── Step 1: import at top of tasks.py ────────────────────────────────────────
from ultralytics.nn.modules.gld import GLCAB, MBFPN

# ── Step 2: inside parse_model(), where module args are resolved ──────────────
# GLCAB receives a single tensor — no extra handling needed.
# MBFPN receives a list of tensors just like Concat.

elif m is MBFPN:
    # in_channels: automatically collected from previous layer channel sizes
    # e.g. args = [256, 256, True]  ->  (out_ch=256, use_glcab=True)
    # The 'from' list in YAML controls which 10 layers are concatenated.
    c2 = args[0]   # out_channels
"""


# ─────────────────────────────────────────────────────────────────────────────
# Example YAML snippet  (YOLOv8-style backbone + neck)
# ─────────────────────────────────────────────────────────────────────────────
YAML_EXAMPLE = """
# Assumes backbone stride outputs at layers 4,6,8,10 (P2-P5)
# and a super-resolution branch outputs at layers 3,5,7,9 (x2-x5).
# Adjust layer indices to match your actual model.

neck:
  # Optional: stand-alone GL-CAB on the last backbone feature
  - [-1, 1, GLCAB, [512]]

  # MB-FPN: collect 10 feature maps via 'from'
  # Order: [P1, P2, P3, P4, P5,  x1, x2, x3, x4, x5]
  - [[2, 4, 6, 8, 10,  1, 3, 5, 7, 9], 1, MBFPN, [256, 256, True]]
  #   ^-- 'from' indices                   ^out   ^use_glcab
"""


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python gld_modules.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    B = 2
    C = 256
    sizes = [80, 40, 20, 10, 5]  # typical FPN spatial resolutions

    # ── GL-CAB ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  GL-CAB  (single-tensor attention)")
    print("=" * 55)
    glcab = GLCAB(C)
    for s in sizes:
        t = torch.randn(B, C, s, s)
        out = glcab(t)
        assert out.shape == t.shape, f"Shape mismatch at size {s}"
        print(f"  Input ({B},{C},{s:2d},{s:2d})  ->  Output {tuple(out.shape)}  ok")

    # ── MB-FPN ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MB-FPN  (multi-branch FPN)")
    print("=" * 55)

    # Build the 10-tensor input list  [P1..P5, x1..x5]
    tensors = [torch.randn(B, C, s, s) for s in sizes] + [torch.randn(B, C, s, s) for s in sizes]

    mbfpn = MBFPN(in_channels=C, out_channels=C, use_glcab=True)
    fused = mbfpn(tensors)

    assert len(fused) == 5, "Expected 5 output feature maps"
    for i, f in enumerate(fused):
        print(f"  P{i + 1}*  {tuple(f.shape)}  ok")

    # ── Parameter counts ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  GLCAB  parameters : {sum(p.numel() for p in GLCAB(C).parameters()):>10,}")
    print(f"  MBFPN  parameters : {sum(p.numel() for p in mbfpn.parameters()):>10,}")
    print("=" * 55)
    print("  All tests passed!")

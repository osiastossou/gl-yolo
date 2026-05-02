"""
YOLO11s-UAV Modules — VERSION FINALE
=====================================
Mi et al., J. Imaging 2026, 12, 69
DOI: 10.3390/jimaging12020069

Modules :
  S2DResConv  — Space-to-Depth + Dilation-wise Residual Convolution
  SimAM       — Simple Parameter-Free Attention Module (Yang et al., ICML 2021)
  FlexSimAM   — C3k2 + SimAM, drop-in pour C3k2 dans YOLOv11
  CARAFEFast  — Content-Aware Upsampling, remplace nn.Upsample dans le neck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_divisible(x: int, divisor: int) -> int:
    """Retourne le plus grand multiple de divisor ≤ x."""
    return max(divisor, (x // divisor) * divisor)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DWR — Dilation-wise Residual
# ─────────────────────────────────────────────────────────────────────────────

class DWR(nn.Module):
    """
    Dilation-wise Residual module.
    Wei et al., DWRSeg — arXiv:2212.01173

    Branches :
      RR : 3×3 Conv régionale
      SR : 3 convolutions dilatées (d=1,3,5) en parallèle
    Sortie = résidu ajouté à l'entrée.
    """

    def __init__(self, c: int):
        super().__init__()
        c2 = c // 2  # moitié des canaux pour RR

        # RR — Regional Residualization
        self.rr = nn.Sequential(
            nn.Conv2d(c, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        # SR — Semantic Residualization : 3 branches dilatées
        # On divise c2 en 3 groupes équilibrés
        c3 = _make_divisible(c2 // 3, 1)   # ~c2/3, diviseur de c2

        # Ajuster pour que c2 soit divisible par le nb de branches
        # On utilise standard conv (groups=1) pour éviter la contrainte
        self.sr1 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1,  dilation=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.sr3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=3,  dilation=3, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.sr5 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=5,  dilation=5, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )

        # Fusion : (c2 + c3*3) → c
        c_fuse = c2 + c3 * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(c_fuse, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rr = self.rr(x)
        sr = torch.cat([self.sr1(rr), self.sr3(rr), self.sr5(rr)], dim=1)
        return x + self.fuse(torch.cat([rr, sr], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 2. S2DResConv — Space-to-Depth + Dilation-wise Residual
# ─────────────────────────────────────────────────────────────────────────────

class S2DResConv(nn.Module):
    """
    Space-to-Depth for Dilation-wise Residual Convolution (Figure 4).

    Remplace Conv(stride=2) dans le backbone YOLOv11.
    Préserve 100% de l'information spatiale via space-to-depth,
    puis enrichit le contexte multi-échelle via DWR.

    YAML  : [-1, 1, S2DResConv, [256]]
    parse_model → S2DResConv(c1=ch[f], c2=256)

    Args:
        c1 (int): Canaux d'entrée.
        c2 (int): Canaux de sortie.
        s  (int): Facteur de down-sampling (défaut 2).
    """

    def __init__(self, c1: int, c2: int, s: int = 2, *args, **kwargs):
        super().__init__()
        self.s   = s
        c_spd    = c1 * s * s   # canaux après space-to-depth

        # Skip résidu : avg pool pour downsampler x → même HW que x_spd
        self.skip = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )

        # 1×1 Conv pour ajuster (c_spd + c1) → c2
        c_cat = c_spd + c1
        self.chan_adj = nn.Sequential(
            nn.Conv2d(c_cat, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        # DWR — enrichissement contextuel multi-scale
        self.dwr = DWR(c2)

        # BN + GELU final (Équation 4 du papier)
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.s

        # ── Space-to-depth ────────────────────────────────────────────────
        x_spd = x.view(B, C, H // s, s, W // s, s)
        x_spd = x_spd.permute(0, 1, 3, 5, 2, 4).contiguous()
        x_spd = x_spd.view(B, C * s * s, H // s, W // s)

        # ── Skip résidu downsampled ───────────────────────────────────────
        x_skip = F.avg_pool2d(self.skip(x), kernel_size=s, stride=s)

        # ── Fusion et projection canaux ───────────────────────────────────
        out = self.chan_adj(torch.cat([x_spd, x_skip], dim=1))

        # ── DWR + BN + GELU ───────────────────────────────────────────────
        return self.bn_act(self.dwr(out))


# ─────────────────────────────────────────────────────────────────────────────
# 3. SimAM — Parameter-Free Attention Module
# ─────────────────────────────────────────────────────────────────────────────

class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module.
    Yang et al., ICML 2021.

    Génère des poids 3D (H×W×C) sans aucun paramètre appris.
    Équation 6 : e*_t = 4(σ²+λ) / ((t-μ)² + 2σ² + 2λ)
    Équation 7 : X̃ = sigmoid(1/E) ⊙ X

    Args:
        e_lambda (float): Terme de régularisation (défaut 1e-4).
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1           # H×W - 1
        x_sq = (x - x.mean(dim=[2, 3], keepdim=True)) ** 2
        e_t = (
            4 * (x_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            / (x_sq + 2 * x_sq.sum(dim=[2, 3], keepdim=True) / n
               + 2 * self.e_lambda)
            + 0.5
        )
        return x * torch.sigmoid(e_t)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FlexSimAM — C3k2 + SimAM (drop-in pour C3k2 dans YOLOv11)
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckSimAM(nn.Module):
    """Bottleneck avec SimAM après le 2ème Conv (Figure 5)."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1, k: tuple = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = max(int(c2 * e), 1)
        self.cv1   = Conv(c1, c_, k[0], 1)
        self.cv2   = Conv(c_, c2, k[1], 1, g=g)
        self.simam = SimAM()
        self.add   = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.simam(self.cv2(self.cv1(x)))
        return x + out if self.add else out


class C3kSimAM(nn.Module):
    """C3k avec BottleneckSimAM (branche c3k=True de FlexSimAM)."""

    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = True, g: int = 1,
                 e: float = 0.5, k: tuple = (3, 3)):
        super().__init__()
        c_ = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m   = nn.Sequential(*[
            BottleneckSimAM(c_, c_, shortcut=shortcut, g=g, k=k, e=1.0)
            for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class FlexSimAM(nn.Module):
    """
    Flexible SimAM — drop-in replacement pour C3k2 dans YOLOv11.

    c3k=True  → branche C3kSimAM (attention renforcée pour le neck)
    c3k=False → branche Bottleneck standard (léger, pour le backbone)

    Signature identique à C3k2 :
        (c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True)

    YAML exemples :
        [-1, 2, FlexSimAM, [256, False, 0.25]]   # backbone c3k=False
        [-1, 2, FlexSimAM, [512, True]]           # neck c3k=True
    """

    def __init__(self, c1: int, c2: int, n: int = 1,
                 c3k: bool = False, e: float = 0.5,
                 attn: bool = False, g: int = 1,
                 shortcut: bool = True, **kwargs):
        super().__init__()
        from ultralytics.nn.modules.block import Bottleneck
        c_ = max(int(c2 * e), 1)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        if c3k:
            self.m = nn.Sequential(*[
                C3kSimAM(c_, c_, 1, shortcut=shortcut, g=g, e=1.0)
                for _ in range(n)
            ])
        else:
            self.m = nn.Sequential(*[
                Bottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0)
                for _ in range(n)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 5. CARAFEFast — Content-Aware Upsampling
# ─────────────────────────────────────────────────────────────────────────────

class CARAFEFast(nn.Module):
    """
    CARAFE: Content-Aware ReAssembly of FEatures.
    Wang et al., ICCV 2019.

    Remplace nn.Upsample dans le neck (CARIFPN).
    Génère des kernels d'upsampling adaptatifs au contenu local
    via unfold (version rapide pour l'entraînement).

    YAML : [-1, 1, CARAFEFast, []]   ou   [-1, 1, nn.Upsample, [None, 2, 'nearest']]

    Note : CARAFEFast ne change pas le nombre de canaux,
    il double H et W (scale=2).

    Args:
        c     (int): Canaux d'entrée = canaux de sortie.
        scale (int): Facteur d'upsampling (défaut 2).
        k_enc (int): Kernel encodeur contenu (défaut 3).
        k_up  (int): Kernel réassemblage (défaut 5).
        c_mid (int): Canaux intermédiaires encodeur (défaut 64).
    """

    def __init__(self, c: int, c2: int = None,
                 scale: int = 2, k_enc: int = 3,
                 k_up: int = 5, c_mid: int = 64, *args, **kwargs):
        super().__init__()
        # c2 ignoré (pas de changement de canaux) — gardé pour compat parse_model
        self.scale = scale
        self.k_up  = k_up

        self.encoder = nn.Sequential(
            nn.Conv2d(c, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, scale * scale * k_up * k_up,
                      k_enc, padding=k_enc // 2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s, k = self.scale, self.k_up

        # Predict kernels
        ker = self.encoder(x)                          # (B, s²k², H, W)
        ker = F.softmax(
            ker.view(B, s * s, k * k, H * W), dim=2
        ).view(B, s * s, k * k, H, W)                 # (B, s², k², H, W)

        # Extract patches via unfold
        x_pad = F.pad(x, [k // 2] * 4)
        x_unf = F.unfold(x_pad, kernel_size=k)        # (B, C×k², H×W)
        x_unf = x_unf.view(B, C, k * k, H, W)        # (B, C, k², H, W)

        # Weighted reassembly : sum over k² dimension
        # x_unf unsqueeze(2): (B, C, 1, k², H, W)
        # ker   unsqueeze(1): (B, 1, s², k², H, W)
        out = (x_unf.unsqueeze(2) * ker.unsqueeze(1)).sum(dim=3)
        # out: (B, C, s², H, W) → pixel shuffle
        out = out.view(B, C, s, s, H, W)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        return out.view(B, C, H * s, W * s)
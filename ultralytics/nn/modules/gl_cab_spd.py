"""
gl_cab_spd.py
=============
Combinaison SPDConv + GL-CAB pour YOLOv12 sur VisDrone.

SPDConv  — préserve l'information spatiale au down-sampling
GL-CAB   — enrichit la représentation locale+globale dans les blocs
           (fidèle à : Bao Liu et al., IEEE DDCLS 2023)

Les deux se complètent exactement selon le papier GL-CAB (section 2.2) :
  "supplement the small object information lost during the down-sampling"
  → SPDConv supprime la perte
  → GL-CAB exploite ce qui est préservé
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# ─────────────────────────────────────────────────────────────────────────────
# SPDConv — Space-to-Depth Convolution
# ─────────────────────────────────────────────────────────────────────────────
class SPDConv(nn.Module):
    """
    Remplace Conv(stride=2) sans perte d'information.

    Réarrange (B, C, H, W) → (B, C·s², H/s, W/s) puis applique
    une Conv stride=1. Aucun pixel n'est abandonné.

    Args:
        c1 (int): Canaux d'entrée.
        c2 (int): Canaux de sortie.
        k  (int): Taille noyau Conv suivant (défaut 3).
        s  (int): Facteur space-to-depth (défaut 2).
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        self.s = s
        self.conv = Conv(c1 * s * s, c2, k, 1)   # stride=1 après rearrangement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.s
        x = x.view(B, C, H // s, s, W // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * s * s, H // s, W // s)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# GL-CAB — Global-Local Combined Attention Block
# Bao Liu et al., IEEE DDCLS 2023, équations (2)–(5)
# ─────────────────────────────────────────────────────────────────────────────
class GL_CAB(nn.Module):
    """
    Équations du papier :
        L(P) = BN(Conv(Hs(Conv(G(P)))))    # LDFE — local detail
        Z(P) = BN(Conv(Hs(Conv(P))))        # LFE  — local feature
        G(P) = BN(Conv(R(BN(Conv(P)))))    # GFE  — global feature
        W(P) = σ(L(P) ⊕ Z(P)) ⊗ G(P)     # sortie

    Args:
        c (int): Canaux (entrée = sortie).
    """

    def __init__(self, c: int):
        super().__init__()

        # LDFE : L(P) = BN(Conv(Hs(Conv(G(P)))))
        self.ldfe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # G(·) global avg pool
            nn.Conv2d(c, c, 1, bias=False),     # Conv
            nn.Hardswish(inplace=True),         # Hs
            nn.Conv2d(c, c, 1, bias=False),     # Conv
            nn.BatchNorm2d(c),                  # BN
        )

        # LFE : Z(P) = BN(Conv(Hs(Conv(P))))
        self.lfe = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),     # Conv
            nn.Hardswish(inplace=True),         # Hs
            nn.Conv2d(c, c, 1, bias=False),     # Conv
            nn.BatchNorm2d(c),                  # BN
        )

        # GFE : G(P) = BN(Conv(R(BN(Conv(P)))))
        self.gfe = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),     # Conv
            nn.BatchNorm2d(c),                  # BN
            nn.ReLU(inplace=True),              # R
            nn.Conv2d(c, c, 1, bias=False),     # Conv
            nn.BatchNorm2d(c),                  # BN
        )

        # Intégration : Conv(L⊕Z) → sigmoid
        self.integrate = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        lp = self.ldfe(x).expand(B, C, H, W)   # L(P) broadcast → (B,C,H,W)
        zp = self.lfe(x)                        # Z(P)
        gp = self.gfe(x)                        # G(P)

        gate = self.integrate(
            torch.cat([lp, zp], dim=1)          # L(P) ⊕ Z(P)
        )
        return gate * gp                        # W(P) = σ(L⊕Z) ⊗ G(P)


# ─────────────────────────────────────────────────────────────────────────────
# C2f_GLCAB — bloc C2f avec GL_CAB intégré
# Drop-in replacement pour C3k2 et A2C2f dans le YAML YOLOv12
# ─────────────────────────────────────────────────────────────────────────────
class C2f_GLCAB(nn.Module):
    """
    C2f avec GL_CAB — signature identique à C3k2 pour compatibilité YAML.

    Structure :
        cv1(x) → [x_0, GL_CAB(x_0)+x_0, GL_CAB(x_1)+x_1, ...] → cv2

    YAML : [-1, 2, C2f_GLCAB, [256, False, 0.25]]

    Args:
        c1       (int)  : canaux entrée.
        c2       (int)  : canaux sortie.
        n        (int)  : nombre de blocs GL_CAB.
        c3k      (bool) : ignoré, compat. signature C3k2.
        e        (float): ratio expansion canaux cachés.
        shortcut (bool) : ignoré, résidu géré en interne.
    """

    def __init__(
        self,
        c1: int, c2: int, n: int = 1,
        c3k: bool = False, e: float = 0.5,
        shortcut: bool = True, **kwargs,
    ):
        super().__init__()
        c_ = max(int(c2 * e), 32)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.m   = nn.ModuleList([GL_CAB(c_) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        for m in self.m:
            y.append(y[-1] + m(y[-1]))          # résidu explicite
        return self.cv2(torch.cat(y, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Enregistrement dans Ultralytics (à appeler avant YOLO())
# ─────────────────────────────────────────────────────────────────────────────
def register_modules():
    """
    Injecte SPDConv, GL_CAB et C2f_GLCAB dans le namespace Ultralytics
    pour que parse_model() les reconnaisse dans le YAML.

    Appeler UNE FOIS avant YOLO() ou DetectionModel().
    """
    import ultralytics.nn.tasks  as _tasks
    import ultralytics.nn.modules as _mods

    for name, cls in [
        ("SPDConv",    SPDConv),
        ("GL_CAB",     GL_CAB),
        ("C2f_GLCAB",  C2f_GLCAB),
    ]:
        _tasks.__dict__[name] = cls
        setattr(_mods, name, cls)

    print("✅  SPDConv + GL_CAB + C2f_GLCAB enregistrés dans Ultralytics.")
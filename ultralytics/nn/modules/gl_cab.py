"""
gl_cab.py
=========
Implémentation fidèle au papier :
  "Global-Local Attention Mechanism Based Small Object Detection"
  Bao Liu, Jinlei Huang — IEEE DDCLS 2023
  DOI: 10.1109/DDCLS58216.2023.10165957

Modules implémentés :
  - GL_CAB  : Global-Local Combined Attention Block (Section 2.2)
  - C2f_GLCAB : wrapper C2f avec GL_CAB intégré (drop-in YOLOv12)
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class GL_CAB(nn.Module):
    """
    Global and Local Combined Attention Block.

    Formules du papier (équations 2-5) :
        L(P) = BN(Conv(Hs(Conv(G(P)))))   # Local Detail — LDFE
        Z(P) = BN(Conv(Hs(Conv(P))))       # Local Feature — LFE
        G(P) = BN(Conv(R(BN(Conv(P)))))   # Global Feature — GFE
        W(P) = σ(L(P) ⊕ Z(P)) ⊗ G(P)    # Sortie finale

    où :
        G(·) = global average pooling
        Hs   = Hard-swish activation
        R    = ReLU activation
        BN   = Batch Normalisation
        ⊕    = concaténation canal
        ⊗    = multiplication élément par élément
        σ    = sigmoid

    Args:
        c (int): Nombre de canaux (entrée = sortie).
    """

    def __init__(self, c: int):
        super().__init__()

        # ── LDFE branch : L(P) = BN(Conv(Hs(Conv(G(P))))) ───────────────
        # G(P) = global average pooling → (B, C, 1, 1)
        # Conv 1×1 → Hardswish → Conv 1×1 → BN
        self.ldfe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                        # G(·)
            nn.Conv2d(c, c, 1, bias=False),                 # Conv
            nn.Hardswish(inplace=True),                     # Hs
            nn.Conv2d(c, c, 1, bias=False),                 # Conv
            nn.BatchNorm2d(c),                              # BN
        )

        # ── LFE branch : Z(P) = BN(Conv(Hs(Conv(P)))) ───────────────────
        # Opère directement sur P (pas de pooling global)
        self.lfe = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),                 # Conv
            nn.Hardswish(inplace=True),                     # Hs
            nn.Conv2d(c, c, 1, bias=False),                 # Conv
            nn.BatchNorm2d(c),                              # BN
        )

        # ── GFE branch : G(P) = BN(Conv(R(BN(Conv(P))))) ────────────────
        self.gfe = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),                 # Conv
            nn.BatchNorm2d(c),                              # BN
            nn.ReLU(inplace=True),                          # R
            nn.Conv2d(c, c, 1, bias=False),                 # Conv
            nn.BatchNorm2d(c),                              # BN
        )

        # ── Intégration : σ(L(P) ⊕ Z(P)) ────────────────────────────────
        # Après concaténation L⊕Z, on a 2c canaux → projection → sigmoid
        self.integrate = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),             # proj 2c→c
            nn.Sigmoid(),                                    # σ
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        W(P) = σ(L(P) ⊕ Z(P)) ⊗ G(P)
        """
        B, C, H, W = x.shape

        # L(P) : LDFE — pool global puis expand à (B,C,H,W)
        lp = self.ldfe(x)                                   # (B, C, 1, 1)
        lp = lp.expand(B, C, H, W)                         # (B, C, H, W)

        # Z(P) : LFE — features locales à résolution native
        zp = self.lfe(x)                                    # (B, C, H, W)

        # G(P) : GFE — features globales sémantiques
        gp = self.gfe(x)                                    # (B, C, H, W)

        # Intégration : σ(L(P) ⊕ Z(P))
        gate = self.integrate(
            torch.cat([lp, zp], dim=1)                      # (B, 2C, H, W)
        )                                                   # (B, C, H, W)

        # W(P) = gate ⊗ G(P)
        return gate * gp                                    # (B, C, H, W)


class C2f_GLCAB(nn.Module):
    """
    C2f avec GL_CAB intégré — drop-in replacement pour C3k2 / A2C2f dans YOLOv12.

    Structure :
        cv1 → split → [GL_CAB] × n → concat → cv2

    Signature identique à C3k2 pour compatibilité YAML directe :
        [-1, 2, C2f_GLCAB, [256, False, 0.25]]

    Args:
        c1       (int)  : canaux d'entrée (inféré par parse_model).
        c2       (int)  : canaux de sortie.
        n        (int)  : nombre de blocs GL_CAB empilés.
        c3k      (bool) : ignoré, gardé pour compat. signature C3k2.
        e        (float): ratio d'expansion des canaux cachés.
        shortcut (bool) : connexion résiduelle (ignoré ici, GL_CAB est déjà résiduel).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        shortcut: bool = True,
        **kwargs,
    ):
        super().__init__()
        c_ = max(int(c2 * e), 32)                          # canaux cachés

        self.cv1 = Conv(c1, c_, 1, 1)                      # projection entrée
        self.cv2 = Conv((1 + n) * c_, c2, 1)               # projection sortie
        self.m   = nn.ModuleList(
            [GL_CAB(c_) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        for m in self.m:
            # résidu explicite : x_i + GL_CAB(x_i)
            y.append(y[-1] + m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))
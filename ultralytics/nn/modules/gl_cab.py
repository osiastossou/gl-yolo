# gl_cab.py — version corrigée

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




class GL_CAB1(nn.Module):
    """
    Global and Local Combined Attention Block.
    Bao Liu et al., IEEE DDCLS 2023 — DOI: 10.1109/DDCLS58216.2023.10165957

    Équations du papier :
        L(P) = BN(Conv(Hs(Conv(G(P)))))   LDFE
        Z(P) = BN(Conv(Hs(Conv(P))))       LFE
        G(P) = BN(Conv(R(BN(Conv(P)))))   GFE
        W(P) = σ(L(P) ⊕ Z(P)) ⊗ G(P)

    Fix : la branche LDFE réduit H×W → 1×1 via GlobalAvgPool.
    BatchNorm2d crash quand B=1 car variance indéfinie sur 1 valeur.
    Solution : GroupNorm(1, c) = LayerNorm sur les canaux, fonctionne
    quelle que soit la taille spatiale.
    """

    def __init__(self, c: int):
        super().__init__()

        # ── LDFE : L(P) = BN(Conv(Hs(Conv(G(P))))) ──────────────────────
        # G(·) réduit à (B,C,1,1) → BN impossible → GroupNorm à la place
        self.ldfe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # G(·) : (B,C,H,W)→(B,C,1,1)
            nn.Conv2d(c, c, 1, bias=False),
            nn.Hardswish(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.GroupNorm(1, c),                 # ← GroupNorm au lieu de BN
        )

        # ── LFE : Z(P) = BN(Conv(Hs(Conv(P)))) ──────────────────────────
        # Opère à résolution native → BN OK
        self.lfe = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.Hardswish(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        # ── GFE : G(P) = BN(Conv(R(BN(Conv(P))))) ───────────────────────
        # Opère à résolution native → BN OK
        self.gfe = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        # ── Intégration : σ(L(P) ⊕ Z(P)) ────────────────────────────────
        self.integrate = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # L(P) : pool global → expand à taille native
        lp = self.ldfe(x)                       # (B, C, 1, 1)
        lp = lp.expand(B, C, H, W)             # (B, C, H, W)

        # Z(P) : features locales
        zp = self.lfe(x)                        # (B, C, H, W)

        # G(P) : features globales
        gp = self.gfe(x)                        # (B, C, H, W)

        # Gate : σ(L(P) ⊕ Z(P))
        gate = self.integrate(
            torch.cat([lp, zp], dim=1)          # (B, 2C, H, W)
        )                                       # (B, C, H, W)

        # W(P) = gate ⊗ G(P)
        return gate * gp


class C2f_GLCAB1(nn.Module):
    """
    C2f avec GL_CAB — drop-in pour C3k2 / A2C2f dans YOLOv12.
    Signature identique à C3k2 pour le YAML.
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
        c_ = max(int(c2 * e), 32)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.m   = nn.ModuleList([GL_CAB1(c_) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        for m in self.m:
            y.append(y[-1] + m(y[-1]))          # résidu
        return self.cv2(torch.cat(y, dim=1))


class GL_CAB(nn.Module):
    """
    Global-Local Combined Attention Block — parse_model compatible.

    Signature : GL_CAB(c1, c2, *args, **kwargs)
    *args et **kwargs absorbent les arguments supplémentaires
    que parse_model peut envoyer selon la version d'Ultralytics.

    YAML : [-1, 1, GL_CAB, [1024]]
    parse_model envoie : GL_CAB(c1=ch[f], c2=1024)
    """

    def __init__(self, c1: int, c2: int, *args, **kwargs):
        super().__init__()
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        c = c2

        self.ldfe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=False),
            nn.Hardswish(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.GroupNorm(1, c),
        )
        self.lfe = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.Hardswish(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.gfe = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.integrate = nn.Sequential(
            nn.Conv2d(2 * c, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        lp = self.ldfe(x).expand(B, C, H, W)
        zp = self.lfe(x)
        gp = self.gfe(x)
        gate = self.integrate(torch.cat([lp, zp], dim=1))
        return gate * gp


class C2f_GLCAB(nn.Module):
    """
    C2f avec GL_CAB — drop-in pour C3k2 dans YOLOv11.

    Signature identique à C3k2 :
        (c1, c2, n=1, c3k=False, e=0.5, shortcut=True)

    parse_model insère n automatiquement dans args[2]
    quand C2f_GLCAB est dans repeat_modules.

    YAML : [-1, 2, C2f_GLCAB, [256, False, 0.25]]
    parse_model envoie : C2f_GLCAB(c1, c2, n=2, c3k=False, e=0.25)
    """

    def __init__(self, c1: int, c2: int, n: int = 1,
                 c3k: bool = False, e: float = 0.5,
                 shortcut: bool = True, **kwargs):
        super().__init__()
        c_ = max(int(c2 * e), 32)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.m   = nn.ModuleList([
            GL_CAB(c_, c_) for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        for m in self.m:
            y.append(y[-1] + m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))
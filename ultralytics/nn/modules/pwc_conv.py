"""
pwc_conv.py
===========
Probability-Weighted Convolution (PWC) — module compatible Ultralytics YOLO11.

PWCConv hérite de Conv (Ultralytics) et override uniquement le forward.

Stratégie de pondération :
    Pour chaque pixel (b, c, h, w), son poids est l'inverse de la probabilité
    de sa valeur dans l'histogramme GLOBAL de l'image d'entrée x.
    Simple, rapide, et capture bien les pixels rares (petits objets).

Auteur : Nelson Akaffou (2026)
"""

import torch
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class PWCConv(Conv):
    """
    Probability-Weighted Convolution — hérite de Conv (Ultralytics).

    Pour chaque pixel, son poids = 1 / (p_bin + eps) où p_bin est la
    probabilité du bin de sa valeur dans l'histogramme global de l'image.
    Pixels rares (petits objets) → poids élevé.
    Pixels fréquents (fond) → poids faible.

    Zéro paramètre supplémentaire par rapport à Conv.
    Interface identique : PWCConv(c1, c2, k, s, p, g, d, act)
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,
                 n_bins=17, eps=1e-6):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.n_bins = n_bins
        self.eps    = eps

    @torch.no_grad()
    def _pwc_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule les poids inverse-probabilité pour chaque pixel.

        x       : (B, C, H, W)
        retourne : (B, C, H, W) — poids normalisés (moyenne = 1)

        Histogramme calculé par image (sur tous les C×H×W pixels),
        puis appliqué pixel par pixel.
        """
        B, C, H, W = x.shape
        N = C * H * W

        # Aplatit x en (B, N) pour calculer l'histogramme par image
        p = x.float().reshape(B, N)

        # Min/max par image
        p_min = p.min(dim=1, keepdim=True).values   # (B, 1)
        p_max = p.max(dim=1, keepdim=True).values   # (B, 1)

        # Bin index : (B, N)
        scale   = self.n_bins / (p_max - p_min + self.eps)
        bin_idx = ((p - p_min) * scale).long().clamp(0, self.n_bins - 1)

        # Histogramme : (B, n_bins)
        counts = torch.zeros(B, self.n_bins, device=x.device)
        ones   = torch.ones(B, N, device=x.device)
        counts.scatter_add_(1, bin_idx, ones)

        # Poids inverse : (B, n_bins)
        inv_p = N / (counts + self.eps)

        # Récupère le poids de chaque pixel : (B, N)
        w = inv_p.gather(1, bin_idx)

        # Normalise par la moyenne (pixels rares > 1, fond ≈ 1)
        w = w / (w.mean(dim=1, keepdim=True) + self.eps)

        return w.reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights  = self._pwc_weights(x).to(x.dtype)
        x_w      = x * weights
        return self.act(self.bn(self.conv(x_w)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        weights = self._pwc_weights(x).to(x.dtype)
        return self.act(self.conv(x * weights))

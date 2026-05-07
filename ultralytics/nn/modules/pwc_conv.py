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
from .conv import Conv


import torch
import torch.nn as nn

from .conv import Conv


class PWCConv(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
        n_bins=17,
        eps=1e-6
    ):

        super().__init__()

        self.conv_block = Conv(
            c1=c1,
            c2=c2,
            k=k,
            s=s,
            p=p,
            g=g,
            d=d,
            act=act
        )
        self.n_bins = n_bins
        self.eps = eps

    def _pwc_weights(self, x):

        B, C, H, W = x.shape

        x_flat = x.view(B, -1)

        weights = []

        for b in range(B):

            xb = x_flat[b].float()

            xb = torch.nan_to_num(
                xb,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            x_min = xb.min()
            x_max = xb.max()

            if torch.abs(x_max - x_min) < self.eps:
                weights.append(torch.ones_like(xb))
                continue

            scale = (self.n_bins - 1) / (x_max - x_min + self.eps)

            bins = ((xb - x_min) * scale).long()

            bins = bins.clamp(0, self.n_bins - 1)

            counts = torch.bincount(
                bins,
                minlength=self.n_bins
            ).float()

            counts = counts + self.eps

            inv_freq = 1.0 / counts[bins]

            inv_freq = inv_freq / inv_freq.mean()

            weights.append(inv_freq)

        weights = torch.stack(weights)

        return weights.view(B, C, H, W)

    def forward(self, x):

        w = self._pwc_weights(x).to(x.dtype)
        x = x * w


        return self.conv_block(x)

    def forward_fuse(self, x):

        w = self._pwc_weights(x).to(x.dtype)

        x = x * w

        return self.conv_block.forward_fuse(x)
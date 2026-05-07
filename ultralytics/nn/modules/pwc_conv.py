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

            # Replace all the nan by 0 in all tensor of xb
            xb = torch.nan_to_num(xb, nan=0.0)

            x_min = xb.min()
            x_max = xb.max()

            # IMPORTANT
            if torch.abs(x_max - x_min) < self.eps:
                w = torch.ones_like(xb)

                weights.append(w)

                continue

            bins = torch.linspace(
                x_min,
                x_max,
                self.n_bins + 1,
                device=x.device
            )
            print(xb)
            hist = torch.histc(
                xb,
                bins=self.n_bins,
                min=x_min.item(),
                max=x_max.item()
            )

            hist = hist + self.eps

            idx = torch.bucketize(xb, bins[:-1])

            idx = idx.clamp(0, self.n_bins - 1)

            inv_freq = 1.0 / hist[idx]

            inv_freq = inv_freq / inv_freq.mean()

            inv_freq = torch.nan_to_num(
                inv_freq,
                nan=1.0,
                posinf=1.0,
                neginf=1.0
            )

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
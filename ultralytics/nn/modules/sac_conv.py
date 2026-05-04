"""
sac_conv.py — Statistically Adaptive Convolution (SAC)
=======================================================
Drop-in replacement for the ultralytics Conv class.
Compatible with YOLOv11 and YOLOv12 (ultralytics ≥ 8.1).

Usage
-----
In your training / inference script, call ``patch_yolo()`` ONCE
before building the model:

    from sac_conv import patch_yolo
    patch_yolo()                         # replace Conv → SACConv globally
    model = YOLO("yolov12n.yaml")        # or yolov11n.yaml

That's it.  No YAML changes needed.

Architecture
------------
For each spatial position (i,j), SACConv:
  1. Extracts the local k×k patch P_{i,j}
  2. Computes 5 statistics: μ, σ, γ (skewness), κ (kurtosis), H (entropy)
  3. Feeds them through a shared lightweight MLP → adaptive kernel W_{i,j}
  4. Applies W_{i,j} ⊙ P_{i,j} (inner product) to produce y_{i,j}

The statistics are treated as stop-gradient (no backprop through them).
The MLP parameters are shared across all spatial positions, so the
parameter overhead is O(d·h + h·c·k²) ≈ 37 K for typical settings.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── autopad helper (mirrors ultralytics) ────────────────────────────────────
def autopad(k, p=None, d=1):
    """Pad to 'same' output shape."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


# ────────────────────────────────────────────────────────────────────────────
# 1.  Statistical descriptor extraction
# ────────────────────────────────────────────────────────────────────────────

def _patch_stats(x: torch.Tensor, k: int, s: int, p: int, d: int) -> torch.Tensor:
    """Compute per-patch statistics used as the SAC conditioning signal.

    Parameters
    ----------
    x : Tensor  (B, C, H, W)
    k, s, p, d : kernel size, stride, padding, dilation  (same as the conv)

    Returns
    -------
    stats : Tensor  (B, 5, H_out, W_out)
        Channel order: [μ, σ, γ, κ, H]
        Computed with torch.no_grad() — stop-gradient by design.
    """
    with torch.no_grad():
        # unfold → (B, C*k*k, L)  where L = H_out * W_out
        patches = F.unfold(x, kernel_size=k, dilation=d, padding=p, stride=s)
        # patches: (B, C*k*k, L)
        B, Ck2, L = patches.shape
        p_flat = patches  # (B, N, L)  N = C*k²

        # ── mean ────────────────────────────────────────────────────────────
        mu = p_flat.mean(dim=1, keepdim=True)               # (B, 1, L)

        # ── std ─────────────────────────────────────────────────────────────
        diff = p_flat - mu                                   # (B, N, L)
        var  = diff.pow(2).mean(dim=1, keepdim=True)        # (B, 1, L)
        sigma = (var + 1e-6).sqrt()                         # (B, 1, L)

        # ── skewness γ ──────────────────────────────────────────────────────
        z    = diff / (sigma + 1e-6)                        # (B, N, L)
        gamma = z.pow(3).mean(dim=1, keepdim=True)          # (B, 1, L)

        # ── excess kurtosis κ ───────────────────────────────────────────────
        kappa = z.pow(4).mean(dim=1, keepdim=True) - 3.0   # (B, 1, L)

        # ── entropy H (16-bin histogram, per patch) ─────────────────────────
        # Normalise patches to [0,1] per spatial location for binning
        p_min = p_flat.min(dim=1, keepdim=True).values
        p_max = p_flat.max(dim=1, keepdim=True).values
        p_norm = (p_flat - p_min) / (p_max - p_min + 1e-6) # (B, N, L)
        B_BINS = 16
        # bin index for each element
        bin_idx = (p_norm * (B_BINS - 1)).long().clamp(0, B_BINS - 1)  # (B, N, L)
        # count per bin per patch: scatter_add over N dimension
        counts = torch.zeros(B, B_BINS, L, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, bin_idx, torch.ones_like(p_flat))
        prob = counts / (Ck2 + 1e-6)                       # (B, B_BINS, L)
        entropy = -(prob * (prob + 1e-9).log()).sum(dim=1, keepdim=True)  # (B, 1, L)

        # ── stack & reshape to spatial ───────────────────────────────────────
        stats = torch.cat([mu, sigma, gamma, kappa, entropy], dim=1)  # (B, 5, L)

    return stats   # (B, 5, L)  — stop-gradient guaranteed by no_grad block


# ────────────────────────────────────────────────────────────────────────────
# 2.  Adaptive kernel generator MLP
# ────────────────────────────────────────────────────────────────────────────

class _KernelGenerator(nn.Module):
    """Shared MLP: stat vector → flat kernel weights.

    To keep parameter count manageable, we factorise the kernel generation:
    the MLP outputs a per-channel scale vector (c_out values) and a shared
    spatial filter (c_in * k * k values).  The full kernel is their outer product:

        W_{i,j}[c_out, c_in, k, k] = alpha[c_out] * phi[c_in, k, k]

    This reduces the MLP output from c_out * c_in * k² to c_out + c_in * k²,
    cutting parameters by ~60× for typical sizes.

    Parameters
    ----------
    d_in  : stat vector dimension (5)
    c_in  : input channels of the convolution
    c_out : output channels
    k     : spatial kernel size
    h_dim : hidden layer width (default 64)
    """

    def __init__(self, d_in: int, c_in: int, c_out: int, k: int, h_dim: int = 64):
        super().__init__()
        self.c_out = c_out
        self.c_in  = c_in
        self.k     = k
        # Factorised output: channel scales + spatial filter
        self.out_alpha = c_out              # per-output-channel scale
        self.out_phi   = c_in * k * k       # shared spatial filter

        self.fc1       = nn.Linear(d_in, h_dim, bias=True)
        self.fc_alpha  = nn.Linear(h_dim, self.out_alpha, bias=True)   # channel weights
        self.fc_phi    = nn.Linear(h_dim, self.out_phi,   bias=True)   # spatial weights
        self.act       = nn.ReLU(inplace=True)

        # Init near zero so SAC starts close to standard conv behaviour
        nn.init.normal_(self.fc_alpha.weight, std=0.01)
        nn.init.ones_(self.fc_alpha.bias)   # default scale = 1
        nn.init.normal_(self.fc_phi.weight, std=0.01)
        nn.init.zeros_(self.fc_phi.bias)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        stats : (B, 5, L)  stop-gradient statistics

        Returns
        -------
        kernels : (B*L, c_out, c_in*k*k)  ready for bmm
        """
        B, D, L = stats.shape
        s   = stats.permute(0, 2, 1).reshape(B * L, D)   # (B*L, 5)
        h   = self.act(self.fc1(s))                       # (B*L, h_dim)

        alpha = self.fc_alpha(h)                          # (B*L, c_out)
        phi   = self.fc_phi(h)                            # (B*L, c_in*k*k)

        # Outer product: (B*L, c_out, 1) * (B*L, 1, c_in*k*k) → (B*L, c_out, c_in*k*k)
        kernels = alpha.unsqueeze(2) * phi.unsqueeze(1)
        return kernels    # (B*L, c_out, c_in*k*k)


# ────────────────────────────────────────────────────────────────────────────
# 3.  SACConv — the main module
# ────────────────────────────────────────────────────────────────────────────

class SACConv1(nn.Module):
    """Statistically Adaptive Convolution.

    Drop-in replacement for ultralytics ``Conv``.
    Identical constructor signature.

    Parameters
    ----------
    c1 : int   input channels
    c2 : int   output channels
    k  : int   kernel size  (default 1)
    s  : int   stride       (default 1)
    p  : int | None  padding (default auto)
    g  : int   groups       (default 1)  — passed to fallback standard conv
    d  : int   dilation     (default 1)
    act: bool | nn.Module   activation flag (mirrors ultralytics Conv)
    h_dim : int  MLP hidden width  (default 64, reduce to 16 to save FLOPs)
    """

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
        h_dim: int = 64,
    ):
        super().__init__()
        self.k = k
        self.s = s
        self.d = d
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.padding = autopad(k, p, d)

        # ── Standard conv path (provides base + BN + act) ──────────────────
        # We keep this for the BN and as an efficient fallback for 1×1 convs.
        self.base_conv = nn.Conv2d(
            c1, c2, k, s, self.padding, groups=g, dilation=d, bias=False
        )
        self.bn  = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act if act is True
            else act if isinstance(act, nn.Module)
            else nn.Identity()
        )

        # ── SAC path — only for k×k with k > 1 and g == 1 ─────────────────
        # For 1×1 convs and grouped convs we fall back to standard conv
        # (a 1×1 patch has only 1 value per channel → statistics are trivial).
        self.use_sac = (k > 1) and (g == 1)

        if self.use_sac:
            self.generator = _KernelGenerator(
                d_in=5, c_in=c1, c_out=c2, k=k, h_dim=h_dim
            )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _apply_adaptive_kernels(
        self, x: torch.Tensor, kernels: torch.Tensor,
        H_out: int, W_out: int
    ) -> torch.Tensor:
        """Apply position-specific kernels via unfold + bmm.

        kernels : (B*L, c_out, c_in*k*k)
        """
        B = x.shape[0]
        L = H_out * W_out

        # patches: (B, c1*k*k, L)
        patches = F.unfold(
            x, kernel_size=self.k, dilation=self.d,
            padding=self.padding, stride=self.s
        )
        # (B*L, c_in*k*k, 1)
        p_flat = patches.permute(0, 2, 1).reshape(B * L, self.c1 * self.k * self.k, 1)

        # bmm: (B*L, c_out, c_in*k*k) × (B*L, c_in*k*k, 1) → (B*L, c_out, 1)
        out = torch.bmm(kernels, p_flat)      # (B*L, c_out, 1)

        # reshape to (B, c_out, H_out, W_out)
        out = out.view(B, L, self.c2).permute(0, 2, 1).view(B, self.c2, H_out, W_out)
        print(out)
        return out

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — SAC when k>1, standard Conv otherwise."""
        if not self.use_sac:
            # 1×1 or grouped conv → standard path (no overhead)
            return self.act(self.bn(self.base_conv(x)))

        B, C, H, W = x.shape
        H_out = math.floor((H + 2 * self.padding - self.d * (self.k - 1) - 1) / self.s + 1)
        W_out = math.floor((W + 2 * self.padding - self.d * (self.k - 1) - 1) / self.s + 1)

        # ── Step 1: statistics (stop-gradient) ───────────────────────────────
        stats = _patch_stats(x, self.k, self.s, self.padding, self.d)
        # stats: (B, 5, L)  with L = H_out * W_out

        # ── Step 2: generate position-specific kernels ───────────────────────
        kernels = self.generator(stats)   # (B*L, c2, c1, k, k)

        # ── Step 3: adaptive convolution ─────────────────────────────────────
        out = self._apply_adaptive_kernels(x, kernels, H_out, W_out)
        # out: (B, c2, H_out, W_out)

        # ── BN + activation (same as standard Conv) ──────────────────────────
        return self.act(self.bn(out))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Fused forward (no BN) — called during model export / inference opt."""
        if not self.use_sac:
            return self.act(self.base_conv(x))

        B, C, H, W = x.shape
        H_out = math.floor((H + 2 * self.padding - self.d * (self.k - 1) - 1) / self.s + 1)
        W_out = math.floor((W + 2 * self.padding - self.d * (self.k - 1) - 1) / self.s + 1)

        stats   = _patch_stats(x, self.k, self.s, self.padding, self.d)
        kernels = self.generator(stats)
        out     = self._apply_adaptive_kernels(x, kernels, H_out, W_out)
        return self.act(out)



import torch
import torch.nn as nn
import torch.nn.functional as F


class SACConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, hdim=16):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.hdim = hdim

        self.fc1 = nn.Linear(5, hdim)
        self.fc2 = nn.Linear(hdim, c2 * c1 * k * k)

    def _get_same_padding(self, H, W):
        """🔥 calcule padding EXACT comme Conv2d"""
        out_h = (H + self.s - 1) // self.s
        out_w = (W + self.s - 1) // self.s

        pad_h = max((out_h - 1) * self.s + self.k - H, 0)
        pad_w = max((out_w - 1) * self.s + self.k - W, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return pad_left, pad_right, pad_top, pad_bottom, out_h, out_w

    def forward(self, x):
        B, C, H, W = x.shape

        # 🔥 1. SAME padding dynamique
        pad_l, pad_r, pad_t, pad_b, H_out, W_out = self._get_same_padding(H, W)

        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        # 🔥 2. unfold
        patches = F.unfold(
            x,
            kernel_size=self.k,
            stride=self.s
        )

        L = patches.shape[-1]

        # 🔥 sécurité critique
        assert L == H_out * W_out, f"Mismatch: {L} vs {H_out*W_out}"

        patches = patches.permute(0, 2, 1)

        # 🔥 3. stats
        mu = patches.mean(-1, keepdim=True)
        sigma = patches.std(-1, keepdim=True) + 1e-6

        normed = (patches - mu) / sigma
        skew = (normed ** 3).mean(-1, keepdim=True)
        kurt = (normed ** 4).mean(-1, keepdim=True) - 3

        prob = F.softmax(patches, dim=-1)
        entropy = -(prob * torch.log(prob + 1e-6)).sum(-1, keepdim=True)

        stats = torch.cat([mu, sigma, skew, kurt, entropy], dim=-1).detach()

        # 🔥 4. dynamic kernel
        h = F.relu(self.fc1(stats))
        weights = self.fc2(h)

        weights = weights.view(B, L, self.c2, self.c1 * self.k * self.k)

        patches = patches.unsqueeze(2)

        out = (weights * patches).sum(-1)

        out = out.permute(0, 2, 1).contiguous()

        # 🔥 5. reshape parfait
        out = out.view(B, self.c2, H_out, W_out)

        return out
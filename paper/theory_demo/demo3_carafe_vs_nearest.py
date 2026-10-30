"""Demonstration 3: content-aware upsampling reconstructs more detail than
nearest-neighbor interpolation.

A toy 2x upsampling reconstruction task on real DOTA image patches (used as
a proxy for a feature map, since CARAFEFast is channel-generic): downsample
real patches 2x, try to reconstruct the original from the downsampled
version with (a) nearest-neighbor interpolation (no training) and (b) a
briefly-trained CARAFEFast module, and compare PSNR on a held-out split.

This is a small, fast proxy task -- not the full detection pipeline -- used
only to validate the mechanism (content-aware reassembly vs fixed
interpolation) in isolation.

Usage:
    python demo3_carafe_vs_nearest.py --dota-images /path/to/DOTAv1.5/images/train
"""

import argparse
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ultralytics.nn.modules.uva import CARAFEFast

torch.manual_seed(0)
random.seed(0)


def load_patches(image_dir: str, n_images=40, patch=64, crops_per_image=8) -> torch.Tensor:
    patches = []
    for p in sorted(glob.glob(f"{image_dir}/*.jpg"))[:n_images]:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        if w < patch * 2 or h < patch * 2:
            continue
        for _ in range(crops_per_image):
            x0, y0 = random.randint(0, w - patch), random.randint(0, h - patch)
            arr = np.asarray(im.crop((x0, y0, x0 + patch, y0 + patch)), dtype=np.float32) / 255.0
            patches.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(patches)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b, reduction="none").mean(dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dota-images", default="/Users/osias/Documents/PHD/CODE/datasets/DOTAv1.5/images/train")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    patches = load_patches(args.dota_images)
    patches = patches[torch.randperm(len(patches))]
    n_train = int(len(patches) * 0.85)
    train_hi, test_hi = patches[:n_train], patches[n_train:]
    print(f"patches: {len(patches)} total, {len(train_hi)} train, {len(test_hi)} test")

    def downsample(hi):
        return F.avg_pool2d(hi, kernel_size=2, stride=2)

    test_lo = downsample(test_hi)
    nn_recon = F.interpolate(test_lo, scale_factor=2, mode="nearest")
    psnr_nn = psnr(nn_recon, test_hi)

    carafe = CARAFEFast(3, scale=2, k_enc=3, k_up=5, c_mid=32)
    opt = torch.optim.Adam(carafe.parameters(), lr=1e-3)
    train_lo = downsample(train_hi)
    for step in range(args.steps):
        idx = torch.randint(0, len(train_hi), (args.batch_size,))
        loss = F.mse_loss(carafe(train_lo[idx]), train_hi[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 300 == 0:
            print(f"  step {step + 1}/{args.steps}  train MSE {loss.item():.5f}")

    carafe.eval()
    with torch.no_grad():
        psnr_carafe = psnr(carafe(test_lo).clamp(0, 1), test_hi)

    print(f"\nPSNR nearest-neighbor upsample : {psnr_nn:.2f} dB")
    print(f"PSNR trained CARAFE upsample   : {psnr_carafe:.2f} dB  (+{psnr_carafe - psnr_nn:.2f} dB)")

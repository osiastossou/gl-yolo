"""Demonstration 1: small-object resolution collapse, grounded in real data.

For every object instance in the DOTA v1.5 and VisDrone training sets,
measures its pixel width/height and checks how much of its footprint
survives at each pyramid stride (4, 8, 16, 32). Produces Table (stride
coverage) in paper/main.tex, Section "Small-object resolution collapse".

Usage:
    python demo1_stride_coverage.py \
        --dota-root /path/to/DOTAv1.5 --visdrone-root /path/to/VisDrone
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image


def dota_instance_sizes(root: str) -> np.ndarray:
    """Read every DOTA YOLO-OBB label, convert to axis-aligned (w, h) in pixels."""
    sizes = []
    for lf in glob.glob(f"{root}/labels/train/*.txt"):
        stem = os.path.splitext(os.path.basename(lf))[0]
        img_path = next(
            (p for ext in (".jpg", ".png", ".jpeg") if os.path.exists(p := f"{root}/images/train/{stem}{ext}")),
            None,
        )
        if img_path is None:
            continue
        try:
            w_img, h_img = Image.open(img_path).size
        except Exception:
            continue
        with open(lf) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 9:
                    continue
                coords = list(map(float, parts[1:9]))
                xs, ys = coords[0::2], coords[1::2]
                sizes.append(((max(xs) - min(xs)) * w_img, (max(ys) - min(ys)) * h_img))
    return np.array(sizes)


def visdrone_instance_sizes(root: str) -> np.ndarray:
    """Read native VisDrone annotations (absolute-pixel bbox), excluding ignored/other categories."""
    sizes = []
    for af in glob.glob(f"{root}/annotations/*.txt"):
        with open(af) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                w, h, cat = float(parts[2]), float(parts[3]), int(parts[5])
                if cat in (0, 11) or w <= 0 or h <= 0:  # 0=ignored region, 11=others
                    continue
                sizes.append((w, h))
    return np.array(sizes)


def report(name: str, wh: np.ndarray, strides=(4, 8, 16, 32)) -> None:
    w, h = wh[:, 0], wh[:, 1]
    area = w * h
    small = area < 32**2  # COCO small-object convention
    print(f"\n=== {name} ===")
    print(f"instances: {len(wh)} | small (area<32^2 px): {small.sum()} ({100 * small.mean():.1f}%)")
    for s in strides:
        ws, hs = w[small] / s, h[small] / s
        collapse_area = ((ws * hs) < 1).mean() * 100
        print(f"  stride {s:>2}: {collapse_area:.1f}% of small objects collapse to < 1 feature cell")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dota-root", default="/Users/osias/Documents/PHD/CODE/datasets/DOTAv1.5")
    parser.add_argument(
        "--visdrone-root", default="/Users/osias/Documents/PHD/CODE/datasets/VisDrone/VisDrone2019-DET-train"
    )
    args = parser.parse_args()

    report("DOTA v1.5 (train)", dota_instance_sizes(args.dota_root))
    report("VisDrone (train)", visdrone_instance_sizes(args.visdrone_root))

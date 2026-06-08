# dota_to_coco.py

import os
import json
import cv2
from pathlib import Path

# Catégories DOTA v1.5
DOTA_CATEGORIES = [
    "plane", "ship", "storage-tank", "baseball-diamond",
    "tennis-court", "basketball-court", "ground-track-field",
    "harbor", "bridge", "large-vehicle", "small-vehicle",
    "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"
]

CAT2ID = {cat: i + 1 for i, cat in enumerate(DOTA_CATEGORIES)}


def parse_dota_annotation(txt_path):
    """Parse un fichier annotation DOTA.
    Format : x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
    On convertit le quadrilatère en bbox horizontal (xmin, ymin, xmax, ymax).
    """
    objects = []
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("imagesource") or line.startswith("gsd"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        coords = list(map(float, parts[:8]))
        category = parts[8].lower().replace(" ", "-")
        if category not in CAT2ID:
            continue

        # Bounding box horizontal depuis les 4 coins du quadrilatère
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, ymin = min(xs), min(ys)
        xmax, ymax = max(xs), max(ys)
        w = xmax - xmin
        h = ymax - ymin

        if w <= 0 or h <= 0:
            continue

        objects.append({
            "category": category,
            "bbox": [xmin, ymin, w, h],   # format COCO [x, y, w, h]
            "area": w * h,
        })
    return objects


def dota_to_coco(images_dir, labels_dir, output_json):
    """Convertit un split DOTA entier en fichier COCO JSON."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    coco = {
        "info": {"description": "DOTA v1.5 converted to COCO format"},
        "categories": [{"id": v, "name": k} for k, v in CAT2ID.items()],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    img_id = 1

    img_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    print(f"Processing {len(img_files)} images...")

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        txt_path = labels_dir / (img_path.stem + ".txt")
        if not txt_path.exists():
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        for obj in parse_dota_annotation(txt_path):
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CAT2ID[obj["category"]],
                "bbox": obj["bbox"],
                "area": obj["area"],
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f)

    print(f"Saved {output_json} — {img_id-1} images, {ann_id-1} annotations")


# ── Utilisation ───────────────────────────────────────────────────────────────
dota_to_coco(
    images_dir="/kaggle/input/dota-v15/train/images",
    labels_dir="/kaggle/input/dota-v15/train/labelTxt",
    output_json="/kaggle/working/dota_train.json",
)

dota_to_coco(
    images_dir="/kaggle/input/dota-v15/val/images",
    labels_dir="/kaggle/input/dota-v15/val/labelTxt",
    output_json="/kaggle/working/dota_val.json",
)
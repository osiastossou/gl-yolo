#!/usr/bin/env python3
"""Découpe DOTAv1.5 en tuiles 1024px (gap 200) pour éviter l'OOM RAM système pendant
l'entraînement (TaskAlignedAssigner bascule sur CPU dès qu'une image a trop d'instances,
ce qui fait exploser la RAM sur les images natives DOTA, qui font jusqu'à 4000x4000px).

Usage (sur la VM, après un git pull) :
    python split_dota_dataset.py
"""

from pathlib import Path

from ultralytics.data.split_dota import split_trainval

DATASETS_DIR = Path.home() / "datasets"

split_trainval(
    data_root=str(DATASETS_DIR / "DOTAv1.5"),
    save_dir=str(DATASETS_DIR / "DOTAv1.5-split"),
    crop_size=1024,
    gap=200,
)
print("OK: split terminé ->", DATASETS_DIR / "DOTAv1.5-split")

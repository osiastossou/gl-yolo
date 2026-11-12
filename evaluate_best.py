#!/usr/bin/env python3
"""Final evaluation of every best.pt on a fixed split, with accuracy + FPS.

results.csv (see analyze_runs.py) has per-epoch metrics recorded during
training, but never records speed. This script finds every
runs/<name><imgsz>/weights/best.pt (optionally filtered by --datasets/--models),
and for each one runs model.val(batch=1) to get, in one pass over the same
split for every architecture:
  - precision / recall / mAP50 / mAP50-95
  - per-image speed (preprocess/inference/postprocess, batch=1 so it's a true
    per-image latency, not amortized over a batch) converted to FPS

Default split is "val" (consistent across MVA/DT/VD, and the only split with
public labels for DOTA -- its official "test" is submission-only, no local
ground truth). Pass --split test to evaluate on the held-out test split
instead (works for MVA and VisDrone's test-dev, not for DOTA).

Examples:
    python evaluate_best.py                                   # every best.pt found, on val
    python evaluate_best.py --datasets MVA --models Y11,Y12,Y11T
    python evaluate_best.py --device cpu                       # force CPU (e.g. on a Mac without CUDA)
    python evaluate_best.py --split test                       # evaluate on test instead of val
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from run_gpu import MODEL_SWEEP
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
DATASET_YAML = {d["dataset_name"]: d["yaml"] for d in MODEL_SWEEP}

# "Y11T-MVA1280" -> ("Y11T", "MVA", 1280)
NAME_RE = re.compile(r"^([A-Za-z0-9]+?)-([A-Za-z]+)(\d+)$")


def parse_run_name(folder_name):
    m = NAME_RE.match(folder_name)
    if not m:
        return folder_name, "?", None
    model, dataset, imgsz = m.groups()
    return model, dataset, int(imgsz)


def find_runs(runs_dir, datasets=None, models=None):
    runs = []
    for best in sorted(runs_dir.glob("*/weights/best.pt")):
        folder = best.parent.parent.name
        model, dataset, imgsz = parse_run_name(folder)
        if dataset not in DATASET_YAML:
            continue
        if datasets and dataset not in datasets:
            continue
        if models and model not in models:
            continue
        runs.append({"run": folder, "model": model, "dataset": dataset, "imgsz": imgsz, "best": best})
    return runs


def evaluate(run, split, device, project_dir):
    data_yaml = DATASET_YAML[run["dataset"]]
    print(f"\n=== {run['run']} ({run['dataset']}, {run['model']}) — split={split} ===")
    model = YOLO(str(run["best"]))
    results = model.val(
        data=data_yaml,
        imgsz=run["imgsz"],
        batch=1,  # per-image speed -> a correct FPS, not one amortized over a batch
        device=device,
        split=split,
        conf=0.001,
        iou=0.6,
        plots=False,
        verbose=False,
        project=str(project_dir),
        name=run["run"],
        exist_ok=True,
    )
    box = results.box
    speed = results.speed  # ms/image: preprocess, inference, postprocess
    total_ms = speed.get("preprocess", 0) + speed["inference"] + speed.get("postprocess", 0)
    return {
        "run": run["run"],
        "model": run["model"],
        "dataset": run["dataset"],
        "precision": box.mp,
        "recall": box.mr,
        "mAP50": box.map50,
        "mAP50-95": box.map,
        "preprocess_ms": speed.get("preprocess", 0),
        "inference_ms": speed["inference"],
        "postprocess_ms": speed.get("postprocess", 0),
        "FPS_inference": 1000.0 / speed["inference"] if speed["inference"] else float("nan"),
        "FPS_total": 1000.0 / total_ms if total_ms else float("nan"),
    }


def plot_comparison(df, outdir, split):
    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset].sort_values("model")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        bars = axes[0].bar(sub["model"], sub["mAP50-95"], color="#2E86AB")
        axes[0].bar_label(bars, fmt="%.3f")
        axes[0].set_title(f"mAP50-95 ({split}) — {dataset}")
        axes[0].grid(axis="y", alpha=0.3)

        bars2 = axes[1].bar(sub["model"], sub["FPS_inference"], color="#E08E45")
        axes[1].bar_label(bars2, fmt="%.1f")
        axes[1].set_title(f"FPS (inference, batch=1) — {dataset}")
        axes[1].grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out = outdir / f"eval_{split}_{dataset}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=None, help="Filtre dataset, ex: 'MVA' ou 'MVA,DT'")
    parser.add_argument("--models", type=str, default=None, help="Filtre modèle, ex: 'Y11,Y12,Y11T'")
    parser.add_argument("--split", type=str, default="val", choices=["test", "val"], help="Split à évaluer (défaut: val)")
    parser.add_argument("--device", type=str, default="0", help="Device ultralytics ('0', 'cpu', 'mps', ...). Défaut: '0'")
    parser.add_argument("--outdir", type=str, default="analysis", help="Dossier de sortie (défaut: analysis)")
    args = parser.parse_args()

    datasets = set(args.datasets.split(",")) if args.datasets else None
    models = set(args.models.split(",")) if args.models else None

    runs = find_runs(RUNS_DIR, datasets, models)
    if not runs:
        print("Aucun best.pt trouvé sous runs/ (vérifie --datasets/--models, et que l'entraînement a déjà sauvegardé un best.pt).")
        return

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [evaluate(r, args.split, args.device, outdir / f"eval_{args.split}_runs") for r in runs]
    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + df.to_string(index=False))

    csv_path = outdir / f"eval_{args.split}_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRésumé -> {csv_path}")

    plot_comparison(df, outdir, args.split)


if __name__ == "__main__":
    main()

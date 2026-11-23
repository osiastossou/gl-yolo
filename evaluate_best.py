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
    python evaluate_best.py --plot-only                        # re-plot from an existing eval_<split>_summary.csv, no GPU needed
    python evaluate_best.py --runs-dir runs-vincent            # evaluate every best.pt under a different runs/ folder
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from run_gpu import MODEL_SWEEP
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
DATASET_YAML = {d["dataset_name"]: d["yaml"] for d in MODEL_SWEEP}

# "Y11T-MVA1280" -> ("Y11T", "MVA", 1280) ; "Y11-NWD-MVA1280" -> ("Y11-NWD", "MVA", 1280)
# Model group is greedy and allows embedded hyphens, so only the trailing "-Dataset<imgsz>" is anchored.
NAME_RE = re.compile(r"^(.+)-([A-Za-z]+)(\d+)$")


def get_device():
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU (mirrors train.py's get_device)."""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _grouped_bars(ax, x, series, fmt):
    """Plot len(series) bars per x position, side by side, each bar_label'd."""
    n = len(series)
    width = 0.8 / n
    for i, (col, label, color) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], col, width, label=label, color=color)
        ax.bar_label(bars, fmt=fmt, fontsize=8)


def plot_grouped(df, outdir, split):
    """3 grouped-bar PNGs per dataset: P/R/F1 together, mAP50/mAP50-95 together, FPS together."""
    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset].sort_values("model")
        models = sub["model"].tolist()
        x = range(len(models))

        fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.8), 5))
        _grouped_bars(ax, x, [
            (sub["precision"], "Precision", "#2E86AB"),
            (sub["recall"], "Recall", "#E08E45"),
            (sub["F1"], "F1", "#6BBF59"),
        ], fmt="%.3f")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_title(f"Precision / Recall / F1 ({split}) — {dataset}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = outdir / f"eval_PRF1_{split}_{dataset}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")

        fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.8), 5))
        _grouped_bars(ax, x, [
            (sub["mAP50"], "mAP50", "#2E86AB"),
            (sub["mAP50-95"], "mAP50-95", "#E08E45"),
        ], fmt="%.3f")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_title(f"mAP50 / mAP50-95 ({split}) — {dataset}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = outdir / f"eval_mAP_{split}_{dataset}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")

        fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.8), 5))
        _grouped_bars(ax, x, [
            (sub["FPS_inference"], "FPS (inférence)", "#2E86AB"),
            (sub["FPS_total"], "FPS (pipeline total)", "#E08E45"),
        ], fmt="%.1f")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_title(f"FPS, batch=1 ({split}) — {dataset}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = outdir / f"eval_FPS_{split}_{dataset}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=None, help="Filtre dataset, ex: 'MVA' ou 'MVA,DT'")
    parser.add_argument("--models", type=str, default=None, help="Filtre modèle, ex: 'Y11,Y12,Y11T'")
    parser.add_argument("--split", type=str, default="val", choices=["test", "val"], help="Split à évaluer (défaut: val)")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device ultralytics ('0', 'cpu', 'mps', ...). Défaut: auto-détection (CUDA > MPS > CPU).",
    )
    parser.add_argument("--outdir", type=str, default=None, help="Dossier de sortie (défaut: <runs-dir>/analysis)")
    parser.add_argument(
        "--runs-dir", type=str, default="runs",
        help="Dossier contenant les <name><imgsz>/weights/best.pt à évaluer (défaut: runs)",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Ne ré-évalue rien : relit <runs-dir>/analysis/eval_<split>_summary.csv déjà généré et regénère juste les graphiques (pas besoin de GPU)",
    )
    args = parser.parse_args()

    datasets = set(args.datasets.split(",")) if args.datasets else None
    models = set(args.models.split(",")) if args.models else None

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir

    # Default outdir lives inside runs_dir itself (runs/analysis, runs-vincent/analysis, ...)
    # instead of a single top-level analysis/ shared by every runs-dir -- keeps each sweep's
    # eval results bundled with its own results (and swept up by run_gpu.py's runs.zip).
    outdir = Path(args.outdir) if args.outdir else runs_dir / "analysis"
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"eval_{args.split}_summary.csv"

    if args.plot_only:
        if not csv_path.exists():
            print(f"Aucun résumé trouvé: {csv_path} (lance d'abord evaluate_best.py sans --plot-only pour le générer).")
            return
        df = pd.read_csv(csv_path)
        if datasets:
            df = df[df["dataset"].isin(datasets)]
        if models:
            df = df[df["model"].isin(models)]
        if df.empty:
            print("Aucune ligne après filtre --datasets/--models.")
            return
    else:
        device = args.device if args.device is not None else get_device()
        print(f"Device: {device}")

        runs = find_runs(runs_dir, datasets, models)
        if not runs:
            print(f"Aucun best.pt trouvé sous {runs_dir}/ (vérifie --runs-dir/--datasets/--models, et que l'entraînement a déjà sauvegardé un best.pt).")
            return

        rows = []
        failed = []
        for r in runs:
            try:
                rows.append(evaluate(r, args.split, device, outdir / f"eval_{args.split}_runs"))
            except Exception as e:
                print(f"[skip] {r['run']}: {e}")
                failed.append(r["run"])
        if failed:
            print(f"\n{len(failed)} run(s) non évalué(s) (dataset manquant localement ?): {', '.join(failed)}")
        if not rows:
            print("Aucun run évalué avec succès.")
            return
        df = pd.DataFrame(rows)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print("\n" + df.to_string(index=False))

        df.to_csv(csv_path, index=False)
        print(f"\nRésumé -> {csv_path}")

    df["F1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"])

    print("\nGraphiques:")
    plot_grouped(df, outdir, args.split)


if __name__ == "__main__":
    main()

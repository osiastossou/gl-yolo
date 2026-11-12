#!/usr/bin/env python3
"""Plot training curves and compare architectures from runs/*/results.csv.

Reads every runs/<name><imgsz>/results.csv produced by train.py (via run_gpu.py),
parses the architecture/dataset out of the folder name (e.g. "Y11T-MVA1280" ->
model=Y11T, dataset=MVA, imgsz=1280), and renders loss curves, metric curves,
cumulative/per-epoch training time, and a final-epoch comparison bar chart per
dataset, plus a text summary table (with total time and avg min/epoch).

Examples:
    python analyze_runs.py                              # all runs found under runs/
    python analyze_runs.py --datasets MVA                # only MVA runs
    python analyze_runs.py --models Y11,Y12,Y11T          # only these architectures
    python analyze_runs.py --outdir analysis              # where to save the PNGs/summary.csv
    python analyze_runs.py --watch 300                    # re-render every 300s (live sweep)
"""

import argparse
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"

METRICS = {
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}
LOSSES = {
    "train/box_loss": "Box loss (train)",
    "train/cls_loss": "Cls loss (train)",
    "train/dfl_loss": "DFL loss (train)",
    "val/box_loss": "Box loss (val)",
    "val/cls_loss": "Cls loss (val)",
    "val/dfl_loss": "DFL loss (val)",
}

# "Y11T-MVA1280" -> ("Y11T", "MVA", 1280)
NAME_RE = re.compile(r"^([A-Za-z0-9]+?)-([A-Za-z]+)(\d+)$")


def parse_run_name(folder_name):
    m = NAME_RE.match(folder_name)
    if not m:
        return folder_name, "?", None
    model, dataset, imgsz = m.groups()
    return model, dataset, int(imgsz)


def load_runs(runs_dir, datasets=None, models=None):
    runs = {}
    for csv_path in sorted(runs_dir.glob("*/results.csv")):
        folder = csv_path.parent.name
        model, dataset, imgsz = parse_run_name(folder)
        if datasets and dataset not in datasets:
            continue
        if models and model not in models:
            continue
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if df.empty:
            continue
        runs[folder] = {"model": model, "dataset": dataset, "imgsz": imgsz, "df": df}
    return runs


def _legend_outside(fig, axes):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6))


def plot_grid(runs, outdir, columns, suptitle_prefix, filename_prefix, ncols):
    for dataset in sorted({r["dataset"] for r in runs.values()}):
        nrows = -(-len(columns) // ncols)  # ceil
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), sharex=True)
        fig.suptitle(f"{suptitle_prefix} — {dataset}")
        for ax, (col, label) in zip(axes.flat, columns.items()):
            for folder, r in sorted(runs.items()):
                if r["dataset"] != dataset or col not in r["df"].columns:
                    continue
                ax.plot(r["df"]["epoch"], r["df"][col], label=r["model"])
            ax.set_title(label)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        for ax in axes.flat[len(columns):]:
            ax.axis("off")
        _legend_outside(fig, axes)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        out = outdir / f"{filename_prefix}_{dataset}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")


def plot_comparison_bars(runs, outdir, metric="metrics/mAP50-95(B)", label="mAP50-95"):
    for dataset in sorted({r["dataset"] for r in runs.values()}):
        names, values = [], []
        for folder, r in sorted(runs.items()):
            if r["dataset"] != dataset or metric not in r["df"].columns:
                continue
            names.append(r["model"])
            values.append(r["df"][metric].iloc[-1])
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
        bars = ax.bar(names, values, color="#2E86AB")
        ax.bar_label(bars, fmt="%.3f")
        ax.set_title(f"{label} (dernière epoch enregistrée) — {dataset}")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = outdir / f"compare_{label.replace('/', '_')}_{dataset}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")


def plot_time(runs, outdir):
    for dataset in sorted({r["dataset"] for r in runs.values()}):
        relevant = {f: r for f, r in runs.items() if r["dataset"] == dataset and "time" in r["df"].columns}
        if not relevant:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for folder, r in sorted(relevant.items()):
            df = r["df"]
            axes[0].plot(df["epoch"], df["time"] / 3600, label=r["model"])
            per_epoch_min = df["time"].diff().fillna(df["time"].iloc[0]) / 60
            axes[1].plot(df["epoch"], per_epoch_min, label=r["model"])
        axes[0].set_title(f"Temps cumulé — {dataset}")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("heures")
        axes[0].grid(alpha=0.3)
        axes[1].set_title(f"Durée par epoch — {dataset}")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("minutes")
        axes[1].grid(alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6))
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        out = outdir / f"time_{dataset}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")


def build_summary(runs):
    rows = []
    for folder, r in sorted(runs.items()):
        df = r["df"]
        if "metrics/mAP50-95(B)" in df.columns:
            best_idx = df["metrics/mAP50-95(B)"].idxmax()
        else:
            best_idx = df.index[-1]
        epochs_done = int(df["epoch"].iloc[-1]) + 1
        row = {
            "run": folder,
            "model": r["model"],
            "dataset": r["dataset"],
            "epochs_done": epochs_done,
            "best_epoch": int(df["epoch"].iloc[best_idx]) + 1,
            "precision": df["metrics/precision(B)"].iloc[best_idx],
            "recall": df["metrics/recall(B)"].iloc[best_idx],
            "mAP50": df["metrics/mAP50(B)"].iloc[best_idx],
            "mAP50-95": df["metrics/mAP50-95(B)"].iloc[best_idx],
        }
        if "time" in df.columns:
            row["total_time_h"] = df["time"].iloc[-1] / 3600
            row["min_per_epoch"] = (df["time"].iloc[-1] / epochs_done) / 60
        rows.append(row)
    return pd.DataFrame(rows)


def render(args):
    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = set(args.datasets.split(",")) if args.datasets else None
    models = set(args.models.split(",")) if args.models else None

    runs = load_runs(RUNS_DIR, datasets, models)
    if not runs:
        print("Aucun run trouvé sous runs/ (vérifie --datasets/--models, et que results.csv a déjà au moins une ligne).")
        return

    print(f"{len(runs)} run(s): {', '.join(sorted(runs))}\n")

    summary = build_summary(runs)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary.to_string(index=False))
    summary.to_csv(outdir / "summary.csv", index=False)
    print(f"\nRésumé -> {outdir / 'summary.csv'}")

    print("\nGraphiques:")
    plot_grid(runs, outdir, LOSSES, "Losses", "losses", ncols=3)
    plot_grid(runs, outdir, METRICS, "Metrics", "metrics", ncols=2)
    plot_time(runs, outdir)
    plot_comparison_bars(runs, outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=None, help="Filtre dataset, ex: 'MVA' ou 'MVA,DT'")
    parser.add_argument("--models", type=str, default=None, help="Filtre modèle, ex: 'Y11,Y12,Y11T'")
    parser.add_argument("--outdir", type=str, default="analysis", help="Dossier de sortie des graphiques (défaut: analysis)")
    parser.add_argument("--watch", type=int, default=0, help="Re-génère toutes les N secondes au lieu d'une seule fois (défaut: 0)")
    args = parser.parse_args()

    if args.watch > 0:
        print(f"Mode watch: régénération toutes les {args.watch}s (Ctrl+C pour arrêter)\n")
        while True:
            render(args)
            print(f"\n--- prochaine mise à jour dans {args.watch}s ---\n")
            time.sleep(args.watch)
    else:
        render(args)


if __name__ == "__main__":
    main()

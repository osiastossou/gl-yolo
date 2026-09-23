"""Bulk-launch train.py across every (dataset, architecture) combination.

For each dataset entry in MODEL_SWEEP, trains every architecture listed under
`folder_model`, reproducing the exact command style:

    python train.py -v 11 -imgsz 1280 -model '<model_yaml>' -data '<data_yaml>' \
        -outname 'runs/<name><imgsz>' -epochs <epochs>

Runs execute sequentially, each as its own subprocess, so a crash in one run
does not take down the rest of the sweep. Once the sweep is done, the whole
`runs/` results folder is zipped up (e.g. to grab it from Colab or upload it
to Drive in one shot).

Example:
    python run_gpu.py -epochs 200 -imgsz 1280
    python run_gpu.py -epochs 200 -imgsz 1280 --dry-run   # print the commands only
    python run_gpu.py -epochs 200 -imgsz 1280 --no-zip    # skip the final zip
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "train.py"

# Each dataset entry sweeps every architecture in `folder_model` over it.
# `folder_model` maps an output name (used for -outname and the results
# folder) to the model YAML to train. The YOLO version passed via `-v` is
# inferred from the name (Y12* -> 12, everything else -> 11): -v only
# selects which pretrained checkpoint (yolo11n.pt / yolo12n.pt) warm-starts
# the custom architecture.
MODEL_SWEEP = [
    {
        "dataset_name": "DT",
        "yaml": "ultralytics/cfg/datasets/DOTAv1.5.yaml",
        "folder_model": {
            "Y11-DT": "ultralytics/cfg/models/11/yolo11.yaml",
            "Y12-DT": "ultralytics/cfg/models/12/yolo12.yaml",
            "Y11GL-DT": "ultralytics/cfg/models/11/yolo11-sppf-gl.yaml",
            "Y11GLCCP-DT": "ultralytics/cfg/models/11/yolo11-gl-ccp.yaml",
            "Y11CCP-DT": "ultralytics/cfg/models/11/yolo11-ccp.yaml",
            "Y11SCC-DT": "ultralytics/cfg/models/11/yolo11-gl-sppfcspc-cbam.yaml",
            "Y11T-DT": "ultralytics/cfg/models/11/yolo11-tosa.yaml",
        },
    },
    {
        "dataset_name": "VD",
        "yaml": "ultralytics/cfg/datasets/VisDrone.yaml",
        "folder_model": {
            "Y11-VD": "ultralytics/cfg/models/11/yolo11.yaml",
            "Y12-VD": "ultralytics/cfg/models/12/yolo12.yaml",
            "Y11GL-VD": "ultralytics/cfg/models/11/yolo11-sppf-gl.yaml",
            "Y11GLCCP-VD": "ultralytics/cfg/models/11/yolo11-gl-ccp.yaml",
            "Y11CCP-VD": "ultralytics/cfg/models/11/yolo11-ccp.yaml",
            "Y11SCC-VD": "ultralytics/cfg/models/11/yolo11-gl-sppfcspc-cbam.yaml",
            "Y11T-VD": "ultralytics/cfg/models/11/yolo11-tosa.yaml",
        },
    },
]


def yolo_version(name: str) -> int:
    """Infer -v (11 or 12) from the output name prefix, e.g. 'Y12-DT' -> 12."""
    return 12 if name.startswith("Y12") else 11


def build_command(model_yaml, data_yaml, outname, epochs, imgsz, version):
    """Build the `python train.py ...` argv for one run."""
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "-v", str(version),
        "-imgsz", str(imgsz),
        "-model", model_yaml,
        "-data", data_yaml,
        "-outname", outname,
        "-epochs", str(epochs),
    ]


def run_sweep(sweep, epochs, imgsz, skip_existing=True, dry_run=False):
    """Run train.py once per (dataset, architecture) pair in `sweep`.

    Args:
        sweep: List of dataset entries as in MODEL_SWEEP.
        epochs: Epochs per run.
        imgsz: Image size per run.
        skip_existing: Skip a run if `runs/<name><imgsz>/weights/best.pt` already exists
            (lets you resume an interrupted sweep without redoing finished runs).
        dry_run: Print the commands instead of executing them.

    Returns:
        List of names whose run failed (non-zero exit code).
    """
    failures = []
    for dataset in sweep:
        data_yaml = dataset["yaml"]
        for name, model_yaml in dataset["folder_model"].items():
            if model_yaml is None:
                print(f"[skip] {name}: pas de YAML renseigné.")
                continue

            outname = f"runs/{name}{imgsz}"
            best_weights = REPO_ROOT / outname / "weights" / "best.pt"
            if skip_existing and best_weights.exists():
                print(f"[skip] {name}: {best_weights} existe déjà.")
                continue

            version = yolo_version(name)
            cmd = build_command(model_yaml, data_yaml, outname, epochs, imgsz, version)
            print(f"\n=== {name} ({dataset['dataset_name']}) ===")
            print(" ".join(cmd))
            if dry_run:
                continue

            result = subprocess.run(cmd, cwd=REPO_ROOT)
            if result.returncode != 0:
                print(f"[FAIL] {name} a échoué (code {result.returncode}).")
                failures.append(name)

    print("\n=== Résumé ===")
    if failures:
        print(f"{len(failures)} run(s) en échec: {', '.join(failures)}")
    else:
        print("Tous les runs se sont terminés sans erreur (ou ont été sautés).")
    return failures


def zip_runs_dir(zip_name="runs.zip"):
    """Zip the `runs/` results directory into `zip_name` at the repo root.

    Returns the path to the archive, or None if there was no `runs/` folder to zip.
    """
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        print(f"[zip] {runs_dir} n'existe pas, rien à zipper.")
        return None

    zip_path = REPO_ROOT / zip_name
    base_name = str(zip_path.with_suffix(""))  # shutil appends the .zip extension itself
    archive = shutil.make_archive(base_name, "zip", root_dir=REPO_ROOT, base_dir="runs")
    print(f"[zip] {runs_dir} -> {archive}")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lance train.py pour toutes les architectures du sweep.")
    parser.add_argument("-epochs", type=int, default=200, help="Nombre d'epochs par run (défaut: 200)")
    parser.add_argument("-imgsz", type=int, default=1280, help="Taille d'image (défaut: 1280)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Relancer même si un best.pt existe déjà")
    parser.add_argument("--dry-run", action="store_true", help="Afficher les commandes sans les exécuter")
    parser.add_argument("--zip-name", type=str, default="runs.zip", help="Nom de l'archive du dossier runs (défaut: runs.zip)")
    parser.add_argument("--no-zip", action="store_true", help="Ne pas zipper le dossier runs à la fin du sweep")
    args = parser.parse_args()

    failed = run_sweep(
        MODEL_SWEEP,
        epochs=args.epochs,
        imgsz=args.imgsz,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )

    if not args.dry_run and not args.no_zip:
        zip_runs_dir(args.zip_name)

    sys.exit(1 if failed else 0)

"""Bulk-launch train.py across every (dataset, architecture) combination.

For each dataset entry in MODEL_SWEEP, trains every architecture listed under
`folder_model`, reproducing the exact command style:

    python train.py -v 11 -imgsz 1280 -model '<model_yaml>' -data '<data_yaml>' \
        -outname 'runs/<name><imgsz>' -epochs <epochs>

Runs are pulled from a shared job queue by one worker thread per GPU, each
worker pinning its subprocess to its own GPU via `CUDA_VISIBLE_DEVICES` (so
`train.py` needs no changes: its device-autodetect sees a single visible
GPU as device 0). A crash in one run does not take down the rest of the
sweep. Each run's console output is redirected to its own file under
`logs/` — with several GPUs training concurrently, interleaving their
progress bars on a shared stdout would just corrupt the display. Once the
sweep is done, the whole `runs/` results folder is zipped up (e.g. to grab
it from Colab or upload it to Drive in one shot).

By default this is a good neighbor on a shared machine: it only picks GPUs
that are actually idle (`detect_free_gpus`, via `nvidia-smi` utilization/free
-memory thresholds -- skips any GPU someone else is already using), and runs
each job `nice`d so it yields CPU to other processes/services on the box.

Example:
    python run_gpu.py -epochs 200 -imgsz 1280                   # auto-detects idle GPUs only
    python run_gpu.py -epochs 200 -imgsz 1280 --gpus 0,1,3       # pin to specific GPUs (bypasses the idle check)
    python run_gpu.py -epochs 200 -imgsz 1280 --gpu-mem-frac 0.5 # cap VRAM per run on a shared GPU
    python run_gpu.py -epochs 200 -imgsz 1280 --dry-run          # print the commands only
    python run_gpu.py -epochs 200 -imgsz 1280 --no-zip           # skip the final zip
    python run_gpu.py -epochs 200 -imgsz 1280 --datasets MVA     # only one dataset's 8 architectures
    python run_gpu.py -epochs 200 -imgsz 1280 --models Y11,Y12,Y11T  # only these architectures, every selected dataset
"""

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "train.py"
LOGS_DIR = REPO_ROOT / "logs"

# Each dataset entry sweeps every architecture in `folder_model` over it.
# `folder_model` maps an output name (used for -outname and the results
# folder) to the model YAML to train. The YOLO version passed via `-v` is
# inferred from the name (Y12* -> 12, everything else -> 11): -v only
# selects which pretrained checkpoint (yolo11n.pt / yolo12n.pt) warm-starts
# the custom architecture.
MODEL_SWEEP = [
    {
        "dataset_name": "MVA",
        "yaml": "ultralytics/cfg/datasets/MVA.yaml",
        "folder_model": {
            "Y11-MVA": "ultralytics/cfg/models/11/yolo11.yaml",
            "Y12-MVA": "ultralytics/cfg/models/12/yolo12.yaml",
            "Y11GL-MVA": "ultralytics/cfg/models/11/yolo11-sppf-gl.yaml",
            "Y11GLCCP-MVA": "ultralytics/cfg/models/11/yolo11-gl-ccp.yaml",
            "Y11CCP-MVA": "ultralytics/cfg/models/11/yolo11-ccp.yaml",
            "Y11SCC-MVA": "ultralytics/cfg/models/11/yolo11-gl-sppfcspc-cbam.yaml",
            "Y11T-MVA": "ultralytics/cfg/models/11/yolo11-tosa.yaml",
            "Y11COMBO-MVA": "ultralytics/cfg/models/11/yolo11-tosa-scc-carafe.yaml",
        },
    },
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
            "Y11COMBO-DT": "ultralytics/cfg/models/11/yolo11-tosa-scc-carafe.yaml",
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
            "Y11COMBO-VD": "ultralytics/cfg/models/11/yolo11-tosa-scc-carafe.yaml",
        },
    },
]


def yolo_version(name: str) -> int:
    """Infer -v (11 or 12) from the output name prefix, e.g. 'Y12-DT' -> 12."""
    return 12 if name.startswith("Y12") else 11


def build_command(model_yaml, data_yaml, outname, epochs, imgsz, version, nice=None, gpu_mem_frac=None, resume=False):
    """Build the `python train.py ...` argv for one run, optionally niced, VRAM-capped and resumed."""
    cmd = []
    if nice is not None:
        cmd += ["nice", "-n", str(nice)]  # lower CPU scheduling priority: yield to other processes
    cmd += [
        sys.executable,
        "-u",  # unbuffered stdout/stderr: log files fill in real time instead of only on flush/exit
        str(TRAIN_SCRIPT),
        "-v", str(version),
        "-imgsz", str(imgsz),
        "-model", model_yaml,
        "-data", data_yaml,
        "-outname", outname,
        "-epochs", str(epochs),
    ]
    if gpu_mem_frac is not None:
        cmd += ["-gpu-mem-frac", str(gpu_mem_frac)]
    if resume:
        cmd += ["-resume", "True"]
    return cmd


def detect_gpu_count():
    """Best-effort GPU count via torch; falls back to 1 worker (CPU/MPS/no torch)."""
    try:
        import torch

        return max(torch.cuda.device_count(), 1)
    except Exception:
        return 1


def detect_free_gpus(max_util=20, min_free_mb=2048):
    """Pick only GPUs that are actually idle, instead of blindly grabbing every GPU index.

    Queries `nvidia-smi` for per-GPU utilization and free memory, and excludes any GPU already
    in use by someone else's process above the given thresholds -- this is the fix for launching
    a sweep on a shared machine and accidentally starving other people's jobs/services on GPUs
    that were already busy. Falls back to `detect_gpu_count()` (every CUDA GPU, or a single
    CPU/MPS worker) when `nvidia-smi` isn't available, e.g. on this Mac.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return list(range(detect_gpu_count()))

    free, busy = [], []
    for line in out.strip().splitlines():
        idx, used, total, util = (int(x.strip()) for x in line.split(","))
        (free if (total - used) >= min_free_mb and util <= max_util else busy).append((idx, total - used, util))

    if busy:
        print("[gpu] ignorée(s) car déjà utilisée(s): " + ", ".join(f"GPU{i} ({u}% util, {f}MB libre)" for i, f, u in busy))
    if not free:
        print("[gpu] aucune GPU libre sous les seuils actuels -- rien ne sera lancé sur cette machine partagée.")
    return [i for i, _, _ in free]


def collect_jobs(sweep, imgsz, skip_existing):
    """Flatten MODEL_SWEEP into a job list, skipping finished runs and resuming interrupted ones.

    A run whose `last.pt` exists but not `best.pt` was cut short mid-training (e.g. a Colab
    session disconnect) -- it's queued with `resume=True` to continue from that checkpoint
    instead of silently restarting from epoch 0 and throwing away everything already trained.
    """
    jobs = []
    for dataset in sweep:
        data_yaml = dataset["yaml"]
        for name, model_yaml in dataset["folder_model"].items():
            if model_yaml is None:
                print(f"[skip] {name}: pas de YAML renseigné.")
                continue

            outname = f"runs/{name}{imgsz}"
            weights_dir = REPO_ROOT / outname / "weights"
            best_weights = weights_dir / "best.pt"
            if skip_existing and best_weights.exists():
                print(f"[skip] {name}: {best_weights} existe déjà.")
                continue

            resume = skip_existing and (weights_dir / "last.pt").exists()
            if resume:
                print(f"[resume] {name}: reprise depuis {weights_dir / 'last.pt'}.")

            jobs.append(
                {
                    "name": name,
                    "dataset_name": dataset["dataset_name"],
                    "model_yaml": model_yaml,
                    "data_yaml": data_yaml,
                    "outname": outname,
                    "resume": resume,
                }
            )
    return jobs


def run_job(job, epochs, imgsz, gpu_id, dry_run, nice=None, gpu_mem_frac=None):
    """Run one job pinned to `gpu_id`, logging its output to `logs/<name><imgsz>.log`.

    Returns True on success (or in dry-run mode), False if the subprocess failed.
    """
    version = yolo_version(job["name"])
    cmd = build_command(
        job["model_yaml"], job["data_yaml"], job["outname"], epochs, imgsz, version, nice, gpu_mem_frac, job["resume"]
    )
    tag = f"{job['name']} ({job['dataset_name']})"
    print(f"[GPU {gpu_id}] démarre {tag}: {' '.join(cmd)}")
    if dry_run:
        return True

    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"{job['name']}{imgsz}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # restrict this subprocess to a single physical GPU

    with open(log_path, "a" if job["resume"] else "w") as log_file:
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    print(f"[GPU {gpu_id}] {tag}: {'OK' if ok else f'FAIL (code {result.returncode})'} — log: {log_path}")
    return ok


def run_sweep(sweep, epochs, imgsz, gpu_ids, skip_existing=True, dry_run=False, nice=None, gpu_mem_frac=None):
    """Run every (dataset, architecture) pair in `sweep`, spread across `gpu_ids`.

    One worker thread per GPU pulls jobs from a shared queue (so a GPU that finishes
    an early, cheap model picks up the next pending job instead of sitting idle while
    other GPUs are still busy on heavier models).

    Args:
        sweep: List of dataset entries as in MODEL_SWEEP.
        epochs: Epochs per run.
        imgsz: Image size per run.
        gpu_ids: List of physical GPU indices to use, e.g. [0, 1, 2, 3]. A single
            entry (e.g. [0], the CPU/MPS default) just runs everything sequentially.
        skip_existing: Skip a run if `runs/<name><imgsz>/weights/best.pt` already exists
            (lets you resume an interrupted sweep without redoing finished runs).
        dry_run: Print the commands instead of executing them.

    Returns:
        List of names whose run failed (non-zero exit code).
    """
    jobs = collect_jobs(sweep, imgsz, skip_existing)
    if not jobs:
        print("Rien à exécuter.")
        return []

    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    failures = []
    lock = threading.Lock()

    def worker(gpu_id):
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                if not run_job(job, epochs, imgsz, gpu_id, dry_run, nice, gpu_mem_frac):
                    with lock:
                        failures.append(job["name"])
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=True) for gpu_id in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

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
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="IDs GPU séparés par des virgules (ex: '0,1,2,3'). Défaut: auto-détection des GPU réellement "
        "libres (via nvidia-smi, cf. --max-gpu-util/--min-free-mem-mb) pour ne pas empiéter sur des GPU "
        "déjà utilisées par d'autres (1 seul worker si aucune GPU CUDA, ex. sur Mac/MPS ou CPU).",
    )
    parser.add_argument(
        "--max-gpu-util", type=int, default=20,
        help="Utilisation GPU (%%) max tolérée pour considérer une GPU comme libre (défaut: 20). Ignoré si --gpus.",
    )
    parser.add_argument(
        "--min-free-mem-mb", type=int, default=2048,
        help="Mémoire GPU libre (MB) minimale requise pour considérer une GPU comme libre (défaut: 2048). Ignoré si --gpus.",
    )
    parser.add_argument(
        "--nice", type=int, default=15,
        help="Priorité CPU du sous-processus train.py (0-19, plus haut = moins prioritaire = cède la main plus "
        "facilement aux autres processus de la machine). Défaut: 15. Mettre 0 pour désactiver.",
    )
    parser.add_argument(
        "--gpu-mem-frac", type=float, default=None,
        help="Limite la fraction de VRAM utilisable par run (0-1), pour ne pas monopoliser une GPU partagée. "
        "Défaut: aucune limite.",
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Filtre par dataset_name séparés par des virgules (ex: 'DT' ou 'DT,VD'). Défaut: tous.",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Filtre par préfixe de modèle séparés par des virgules (ex: 'Y11,Y12,Y11T' pour prioriser les "
        "baselines + TOSA). Préfixes valides: Y11, Y12, Y11GL, Y11GLCCP, Y11CCP, Y11SCC, Y11T, Y11COMBO. "
        "Défaut: tous.",
    )
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")] if args.gpus else detect_free_gpus(args.max_gpu_util, args.min_free_mem_mb)
    print(f"GPU(s) utilisée(s): {gpu_ids}")
    if not gpu_ids:
        sys.exit(1)

    sweep = MODEL_SWEEP
    if args.datasets:
        wanted = set(args.datasets.split(","))
        sweep = [d for d in MODEL_SWEEP if d["dataset_name"] in wanted]
        print(f"Datasets sélectionnés: {[d['dataset_name'] for d in sweep]}")

    if args.models:
        wanted_models = set(args.models.split(","))
        sweep = [
            {**d, "folder_model": {n: y for n, y in d["folder_model"].items() if n.split("-")[0] in wanted_models}}
            for d in sweep
        ]
        print(f"Modèles sélectionnés: {sorted(wanted_models)}")

    failed = run_sweep(
        sweep,
        epochs=args.epochs,
        imgsz=args.imgsz,
        gpu_ids=gpu_ids,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
        nice=args.nice or None,
        gpu_mem_frac=args.gpu_mem_frac,
    )

    if not args.dry_run and not args.no_zip:
        zip_runs_dir(args.zip_name)

    sys.exit(1 if failed else 0)

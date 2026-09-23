"""Train a custom YOLO model built from a YAML architecture definition.

Example:
    python train.py -model ultralytics/cfg/models/11/yolo11-gl-ccp.yaml \
        -data ultralytics/cfg/datasets/DOTAv1.5.yaml -outname runs/yolo11-gl-ccp -epochs 10
"""

import argparse
import os

import torch

from ultralytics import YOLO

# Base pre-trained weights to warm-start a custom architecture from, keyed by YOLO major version.
PRETRAINED_WEIGHTS = {11: "yolo11n.pt", 12: "yolo12n.pt"}


def get_device():
    """Pick the best available device: CUDA > MPS (Apple Silicon) > TPU (XLA) > CPU."""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except ImportError:
        return "cpu"


def str2bool(value: str) -> bool:
    """Parse a CLI string as a boolean (argparse's `type=bool` is broken: bool("False") is True)."""
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    if value.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Valeur booléenne attendue, reçu: {value!r}")


def main(modelpath, data, outname, epochs, imgsz=640, v=11, pretrained=True, resume=False):
    """Train (or resume) a YOLO model built from `modelpath`, on `data`, saving under `outname`."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = get_device()

    if resume:
        print("Reprise de l'entraînement à partir du dernier checkpoint.")
        model = YOLO(f"{outname}/weights/last.pt")
    else:
        print(f"Chargement de l'architecture depuis {modelpath}")
        model = YOLO(modelpath)
        if pretrained:
            base_weights = PRETRAINED_WEIGHTS.get(v)
            if base_weights is None:
                raise ValueError(f"Pas de poids pré-entraînés connus pour YOLO v{v}.")
            print(f"Chargement des poids pré-entraînés {base_weights}")
            model = model.load(base_weights)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print(f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB VRAM")

    # `outname` encode project/name, ex: "runs/yolo11-gl-ccp" -> project="runs", name="yolo11-gl-ccp".
    name = os.path.basename(outname)
    project = os.path.dirname(outname)

    print(f"Début de l'entraînement ({epochs} epochs, imgsz={imgsz})...")
    results = model.train(
        data=data,
        epochs=epochs,
        # ── Résolution & batch ──────────────────────────────────────────
        imgsz=imgsz,
        batch=16,  # fixe — pas d'auto-batch, pour garantir l'équité entre modèles comparés
        # ── Device ───────────────────────────────────────────────────────
        device=device,
        amp=True,
        # ── Optimiseur ───────────────────────────────────────────────────
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,  # LR final = lr0 × lrf = 0.0001
        momentum=0.937,
        weight_decay=0.0005,
        # ── Scheduler ────────────────────────────────────────────────────
        cos_lr=True,
        # ── Poids de la loss ─────────────────────────────────────────────
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # ── Sauvegarde & suivi ───────────────────────────────────────────
        save=True,
        save_period=50,  # checkpoint toutes les 50 epochs
        plots=True,
        exist_ok=True,
        seed=42,
        resume=resume,
        workers=2,
        name=name,
        project=project,
    )
    print("Entraînement terminé.")

    print("Début de la validation...")
    metrics = model.val()
    print("Métriques de validation :")
    print(metrics)

    return results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle YOLO à partir d'une architecture YAML custom.")
    parser.add_argument("-model", type=str, required=True, help="Chemin vers le YAML d'architecture YOLO")
    parser.add_argument("-data", type=str, required=True, help="Chemin vers le YAML du dataset")
    parser.add_argument("-outname", type=str, required=True, help="project/name de sortie (ex: runs/yolo11-gl-ccp)")
    parser.add_argument("-epochs", type=int, required=True, help="Nombre d'epochs")
    parser.add_argument("-imgsz", type=int, default=640, help="Taille d'image d'entraînement (défaut: 640)")
    parser.add_argument("-v", type=int, default=11, choices=[11, 12], help="Version YOLO des poids pré-entraînés (défaut: 11)")
    parser.add_argument("-p", type=str2bool, default=True, help="Charger des poids pré-entraînés (défaut: True)")
    parser.add_argument("-resume", type=str2bool, default=False, help="Reprendre depuis le dernier checkpoint (défaut: False)")

    args = parser.parse_args()
    main(args.model, args.data, args.outname, args.epochs, args.imgsz, args.v, args.p, args.resume)

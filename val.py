"""
YOLOv11 — Load Model & Run Validation.
======================================
Usage:
    python yolo11_validation.py
    python yolo11_validation.py --model yolo11n.pt --data coco.yaml --imgsz 640
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — modifiez ici selon votre setup
# ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "yolo11n.pt"  # yolo11n / s / m / l / x
DEFAULT_DATA = "coco128.yaml"  # votre dataset YAML
DEFAULT_IMGSZ = 640  # taille d'entrée
DEFAULT_BATCH = 16  # batch size
DEFAULT_DEVICE = "cpu"  # "cpu" | "cuda" | "0" | "0,1"
DEFAULT_SPLIT = "val"  # "val" | "test" | "train"
DEFAULT_CONF = 0.001  # seuil confiance (bas pour val complète)
DEFAULT_IOU = 0.6  # seuil IoU NMS


# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 Validation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--save-json", action="store_true", help="Sauvegarder les résultats au format COCO JSON")
    parser.add_argument("--plots", action="store_true", help="Générer les courbes P/R et matrices de confusion")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
def load_model(model_path: str) -> YOLO:
    """Charge un modèle YOLOv11. - Si le fichier .pt existe localement → charge depuis le disque - Sinon → télécharge
    automatiquement depuis Ultralytics Hub.
    """
    path = Path(model_path)

    if path.exists():
        print(f"[✓] Chargement du modèle local : {path}")
    else:
        print(f"[↓] Modèle non trouvé localement — téléchargement : {model_path}")

    model = YOLO(model_path)

    # Infos du modèle
    print(f"\n{'─' * 50}")
    print(f"  Modèle    : {model_path}")
    print(f"  Tâche     : {model.task}")
    print(f"  Classes   : {model.names}")
    print(f"  Nb classes: {len(model.names)}")
    print(f"{'─' * 50}\n")

    return model


# ──────────────────────────────────────────────────────────────
def run_validation(model: YOLO, args) -> dict:
    """Lance la validation et retourne les métriques."""
    print("[▶] Lancement de la validation...")
    print(f"    Dataset  : {args.data}")
    print(f"    Split    : {args.split}")
    print(f"    Image sz : {args.imgsz}")
    print(f"    Batch    : {args.batch}")
    print(f"    Device   : {args.device}")
    print(f"    Conf     : {args.conf}")
    print(f"    IoU      : {args.iou}\n")

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        plots=args.plots,
        verbose=True,
    )

    return results


# ──────────────────────────────────────────────────────────────
def print_metrics(results) -> None:
    """Affiche un résumé clair des métriques de validation."""
    box = results.box  # métriques de détection

    print(f"\n{'═' * 50}")
    print("  RÉSULTATS DE VALIDATION — YOLOv11")
    print(f"{'═' * 50}")
    print(f"  Precision (P)   : {box.mp:.4f}   ({box.mp * 100:.2f}%)")
    print(f"  Recall    (R)   : {box.mr:.4f}   ({box.mr * 100:.2f}%)")
    print(f"  mAP@50          : {box.map50:.4f}   ({box.map50 * 100:.2f}%)")
    print(f"  mAP@50-95       : {box.map:.4f}   ({box.map * 100:.2f}%)")
    print(f"{'─' * 50}")

    # Métriques par classe
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        print("\n  Métriques par classe :")
        print(f"  {'Classe':<20} {'AP@50':>8} {'AP50-95':>10}")
        print(f"  {'─' * 40}")
        for i, cls_idx in enumerate(box.ap_class_index):
            cls_name = results.names[cls_idx]
            ap50 = box.ap50[i] if hasattr(box, "ap50") else 0
            ap = box.ap[i] if hasattr(box, "ap") else 0
            print(f"  {cls_name:<20} {ap50:>8.4f} {ap:>10.4f}")

    print(f"{'═' * 50}\n")

    # Chemin des résultats sauvegardés
    if hasattr(results, "save_dir") and results.save_dir:
        print(f"[✓] Résultats sauvegardés dans : {results.save_dir}\n")


# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    model = load_model(args.model)
    results = run_validation(model, args)
    print_metrics(results)


if __name__ == "__main__":
    main()


# python val.py --model models_best/yolo11-MVA-100.pt --data data.yaml --imgsz 640 --batch 16 --device cuda --plots

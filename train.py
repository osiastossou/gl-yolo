# train.py

# /Users/osias/Documents/PHD/CODE/data/data.yaml
import os

import torch

from ultralytics import YOLO


def main(modelpath, data, outname, epochs):
    """Function principale pour entraîner le modèle YOLOv11 customisé."""
    # S'assurer que le script s'exécute dans le bon répertoire
    # pour qu'il puisse trouver les autres fichiers.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # --- 1. CHARGEMENT DU MODÈLE DE BASE ---
    # On charge le modèle pré-entrainé standard. La librairie va correctement
    # parser le nom 'yolov11n.pt' et définir l'échelle ('n') du modèle.
    # C'est cette étape qui résout l'erreur 'UnboundLocalError'.
    print("Chargement du modèle de base (modelpath)")
    model = YOLO(modelpath)
    print("Modèle de base chargé.")

    model = model.load("yolo12n.pt")

    # --- 2. ENTRAÎNEMENT AVEC CONFIGURATION CUSTOMISÉE ---
    # On lance l'entraînement en passant notre architecture customisée
    # via l'argument 'cfg'. La librairie va alors reconstruire le graphe
    # du modèle en se basant sur notre YAML, tout en conservant le contexte
    # (comme l'échelle) correctement défini.
    # !! Remplacez 'path/to/your/dataset.yaml' par le chemin de votre fichier de configuration de données !!
    # Ce fichier décrit les chemins vers vos images d'entraînement/validation et les noms des classes.
    print("Début de l'entraînement avec l'architecture yolov11n-cbam.yaml...")
    model.train(
        # cfg='yolov11n-cbam.yaml',  # Spécifie notre architecture customisée ici
        # data='/Users/osias/Documents/PHD/CODE/data/data.yaml',         # EXAMPLE: à remplacer par votre fichier de données
        data=data,
        epochs=epochs,
        imgsz=640,
        batch=-1,
        device=0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        amp=True,  # 0 pour le premier GPU, ou 'cpu'
        name=outname,  # Nom de l'expérience
    )
    print("Entraînement terminé.")

    # --- 3. VALIDATION (Optionnel mais recommandé) ---
    # Évaluer les performances du modèle entraîné sur le jeu de validation.
    print("Début de la validation...")
    metrics = model.val()
    print("Métriques de validation :")
    print(metrics)


if __name__ == "__main__":
    import argparse

    # Cette instruction est cruciale pour l'entraînement multi-GPU et sur certains systèmes.
    parser = argparse.ArgumentParser(description="Training script with custom arguments")

    parser.add_argument("-model", type=str, required=True, help="Path to YOLO model YAML file")
    parser.add_argument("-data", type=str, required=True, help="Path to dataset .yaml file")
    parser.add_argument("-outname", type=str, required=True, help="Experiment output name")
    parser.add_argument("-epochs", type=int, required=True, help="Experiment output epochs")

    args = parser.parse_args()

    # Appel de main avec les arguments
    main(args.model, args.data, args.outname, args.epochs)

    # python main.py -model 'Yolo11CBAM/yolov11n-cbam.yaml' -data '/Users/osias/Documents/PHD/CODE/birds_data/data.yaml' -outname 'outname' -epochs 100

    # python train.py -model 'Yolo11CBAM/yolov11n-transformer.yaml' -data '/Users/osias/Documents/PHD/CODE/mva2023_sod4bird_dataset/data.yaml' -outname 'outname' -epochs 100

    # python main.py -model 'Yolo11CBAM/yolov11n-gld-cbam.yaml' -data '/Users/osias/Documents/PHD/CODE/birds_data/data.yaml' -outname 'outname' -epochs 100

    # python train.py -model 'ultralytics/cfg/models/12/yolo12.yaml' -data '/Users/osias/Documents/PHD/ultralytics-osias/ultralytics/cfg/datasets/DOTAv1.5.yaml' -outname 'outname' -epochs 100

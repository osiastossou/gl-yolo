# train.py

# /Users/osias/Documents/PHD/CODE/data/data.yaml
import os,sys
import torch

from ultralytics import YOLO






def main(modelpath, data, outname,epochs,imgsz=640):
    """
    Fonction principale pour entraîner le modèle YOLOv11 customisé.
    """
    # S'assurer que le script s'exécute dans le bon répertoire
    # pour qu'il puisse trouver les autres fichiers.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # --- 1. CHARGEMENT DU MODÈLE DE BASE ---
    # On charge le modèle pré-entrainé standard. La librairie va correctement
    # parser le nom 'yolov11n.pt' et définir l'échelle ('n') du modèle.
    # C'est cette étape qui résout l'erreur 'UnboundLocalError'.
    print(f"Chargement du modèle de base (modelpath)")
    model = YOLO(modelpath) 
    print("Modèle de base chargé.")

    model = model.load("yolo11n.pt")

    # --- 2. ENTRAÎNEMENT AVEC CONFIGURATION CUSTOMISÉE ---
    # On lance l'entraînement en passant notre architecture customisée
    # via l'argument 'cfg'. La librairie va alors reconstruire le graphe
    # du modèle en se basant sur notre YAML, tout en conservant le contexte
    # (comme l'échelle) correctement défini.
    # !! Remplacez 'path/to/your/dataset.yaml' par le chemin de votre fichier de configuration de données !!
    # Ce fichier décrit les chemins vers vos images d'entraînement/validation et les noms des classes.
    print("Début de l'entraînement avec l'architecture yolov11n-cbam.yaml...")
    #results = model.train(
        # cfg='yolov11n-cbam.yaml',  # Spécifie notre architecture customisée ici
        #data='/Users/osias/Documents/PHD/CODE/data/data.yaml',         # EXEMPLE: à remplacer par votre fichier de données
        #data=data,
        #epochs=epochs,
        #imgsz=imgsz,
        #batch=-1,
        #device=0 if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        #amp=True,                # 0 pour le premier GPU, ou 'cpu'
        #name=outname, # Nom de l'expérience
    #)

    results = model.train(
        data=data,
        epochs=epochs,

        # ── Résolution & batch ────────────────────────────────────────────────
        imgsz=imgsz,
        batch=16,  # Fixe — pas auto-batch pour garantir l'équité entre modèles

        # ── Device ────────────────────────────────────────────────────────────
        device=0 if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        amp=True,

        # ── Optimiseur ────────────────────────────────────────────────────────
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,  # LR final = lr0 × lrf = 0.0001
        momentum=0.937,
        weight_decay=0.0005,

        # ── Warm-up ───────────────────────────────────────────────────────────
        warmup_epochs=5.0,  # Plus long pour SPD/GLCAB (branches à initialiser)
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ── Scheduler ─────────────────────────────────────────────────────────
        cos_lr=True,

        # ── Augmentation ──────────────────────────────────────────────────────
        mosaic=1.0,
        close_mosaic=30,  # Désactive mosaic aux 30 dernières epochs sur 600
        copy_paste=0.3,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # ── Loss weights ──────────────────────────────────────────────────────
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # ── Sauvegarde & monitoring ───────────────────────────────────────────
        #patience=100,  # Sur 600 epochs, laisser 100 epochs sans amélioration
        save=True,
        save_period=50,  # Checkpoint toutes les 50 epochs
        plots=True,
        exist_ok=True,

        name=outname,
    )

    print("Entraînement terminé.")

    # --- 3. VALIDATION (Optionnel mais recommandé) ---
    # Évaluer les performances du modèle entraîné sur le jeu de validation.
    print("Début de la validation...")
    metrics = model.val()
    print("Métriques de validation :")
    print(metrics)

if __name__ == '__main__':
    import argparse

    # Cette instruction est cruciale pour l'entraînement multi-GPU et sur certains systèmes.
    parser = argparse.ArgumentParser(description="Training script with custom arguments")

    parser.add_argument('-model', type=str, required=True, help="Path to YOLO model YAML file")
    parser.add_argument('-data', type=str, required=True, help="Path to dataset .yaml file")
    parser.add_argument('-outname', type=str, required=True, help="Experiment output name")
    parser.add_argument('-epochs', type=int, required=True, help="Experiment output epochs")
    parser.add_argument('-imgsz', type=int, default=640, help="Image size for training (default: 640)")

    args = parser.parse_args()

    # Appel de main avec les arguments
    main(args.model, args.data, args.outname, args.epochs, args.imgsz)


    #python main.py -model 'Yolo11CBAM/yolov11n-cbam.yaml' -data '/Users/osias/Documents/PHD/CODE/birds_data/data.yaml' -outname 'outname' -epochs 100

    #python train.py -model 'Yolo11CBAM/yolov11n-transformer.yaml' -data '/Users/osias/Documents/PHD/CODE/mva2023_sod4bird_dataset/data.yaml' -outname 'outname' -epochs 100


    #python main.py -model 'Yolo11CBAM/yolov11n-gld-cbam.yaml' -data '/Users/osias/Documents/PHD/CODE/birds_data/data.yaml' -outname 'outname' -epochs 100

    
    #python train.py -model 'ultralytics/cfg/models/12/yolo12.yaml' -data '/Users/osias/Documents/PHD/ultralytics-osias/ultralytics/cfg/datasets/DOTAv1.5.yaml' -outname 'outname' -epochs 100

#python train.py -model 'ultralytics/cfg/models/12/yolo12-adapt-a2c2f.yaml' -data 'ultralytics/cfg/datasets/DOTAv1.5.yaml' -outname 'yolo12-glcab-full-10' -epochs 10






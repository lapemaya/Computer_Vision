import os
import argparse
from ultralytics import YOLO
import torch

def main(data_dir, model_name='yolov8n.pt', epochs=50, imgsz=640):
    # Verifica GPU disponibile
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Training su: {device}")

    # Check if dataset exists
    data_yaml = os.path.join(data_dir, 'data.yaml')
    if not os.path.exists(data_yaml):
        print(f"data.yaml not found in {data_dir}. Please provide a valid YOLO dataset.")
        return

    # Carica il modello
    print(f"📦 Caricamento modello {model_name}...")
    model = YOLO(model_name)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Configurazione training
    print("\n⚙️  Configurazione training:")
    print(f"   Dataset: {data_dir}")
    print(f"   Modello: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: 16")
    print(f"   Device: {device}")

    # Avvia il training
    print("\n🚀 Avvio training...\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project='../runs/train',
        name='yolo_Generale',
        exist_ok=True,
        device=device,
        batch=16,  # Batch size ottimizzato

        # Ottimizzazione
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,  # Rotazione leggera
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,

        # Early stopping e salvataggio
        patience=50,
        save_period=10,
        val=True,
        plots=True
    )

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETATO!")
    print("="*60)

    # Risultati
    print(f"\n📊 Risultati salvati in: ../runs/train/yolo_Generale")
    print(f"💾 Best weights: ../runs/train/yolo_Generale/weights/best.pt")
    print(f"💾 Last weights: ../runs/train/yolo_Generale/weights/last.pt")

    # Valida il modello
    print("\n🔍 Validazione del modello...")
    metrics = model.val()

    print("\n" + "="*60)
    print("📊 METRICHE FINALI")
    print("="*60)

    # metrics.box contiene array numpy, prendiamo la media
    if hasattr(metrics.box, 'p') and metrics.box.p is not None:
        precision = metrics.box.p.mean() if hasattr(metrics.box.p, 'mean') else metrics.box.p
        print(f"Precision: {precision:.3f}")
    else:
        print("Precision: N/A")

    if hasattr(metrics.box, 'r') and metrics.box.r is not None:
        recall = metrics.box.r.mean() if hasattr(metrics.box.r, 'mean') else metrics.box.r
        print(f"Recall: {recall:.3f}")
    else:
        print("Recall: N/A")

    if hasattr(metrics.box, 'map50'):
        print(f"mAP50: {metrics.box.map50:.3f}")
    else:
        print("mAP50: N/A")

    if hasattr(metrics.box, 'map'):
        print(f"mAP50-95: {metrics.box.map:.3f}")
    else:
        print("mAP50-95: N/A")

    print("\n✨ Training completato con successo!")
    print(f"🎯 Usa il modello: ../runs/train/yolo_Generale/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on datasetGenerale folder")
    parser.add_argument('--data_dir', type=str, default='datasetGenerale', help='Path to dataset folder')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    args = parser.parse_args()

    main(args.data_dir, args.model, args.epochs, args.imgsz)
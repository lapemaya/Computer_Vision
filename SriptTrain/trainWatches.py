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
    print(f"   Classi: Watches")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: 8")
    print(f"   Fraction: 0.5 (50% del dataset)")

    # Avvia il training
    print("\n🚀 Avvio training...\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project='runs/train',
        name='yolo_watches',
        exist_ok=True,
        device=device,
        batch=8,  # Reduce batch size to help with CUDA OOM
        fraction=0.5   # usa solo il 50% del dataset
    )

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETATO!")
    print("="*60)

    # Risultati
    print(f"\n📊 Risultati salvati in: runs/train/yolo_watches")
    print(f"📈 mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"📈 mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"💾 Best weights: runs/train/yolo_watches/weights/best.pt")
    print(f"💾 Last weights: runs/train/yolo_watches/weights/last.pt")

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
    print(f"🎯 Usa il modello: runs/train/yolo_watches/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on datasetWatches folder")
    parser.add_argument('--data_dir', type=str, default='datasetWatches', help='Path to dataset folder')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    args = parser.parse_args()

    main(args.data_dir, args.model, args.epochs, args.imgsz)
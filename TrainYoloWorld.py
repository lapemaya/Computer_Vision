"""
Script per trainare YOLOv8-World X con il dataset generale
"""

from ultralytics import YOLOWorld
import torch

# Verifica GPU disponibile
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Training su: {device}")

# Carica il modello YOLOv8-World X pre-trained
print("📦 Caricamento modello YOLOv8-World X...")
model = YOLOWorld('yolov8s-world.pt')

# Configurazione training
print("\n⚙️  Configurazione training:")
print("   Dataset: datasetGenerale")
print("   Modello: YOLOv8-World X")
print("   Classi: Wallet, Keys, Glasses, Remote, Pen")

# Parametri di training
EPOCHS = 50  # Aumentato da 1 a 50
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 50  # Aumentato da 20 a 50

print(f"   Epochs: {EPOCHS}")
print(f"   Image size: {IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Patience: {PATIENCE}")

# Avvia il training
print("\n🚀 Avvio training...\n")

results = model.train(
    data='datasetGenerale/data.yaml',
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    device=device,
    project='runs/train',
    name='yoloworld_generale',
    exist_ok=True,

    # Parametri ottimizzazione
    optimizer='AdamW',
    lr0=0.001,  # Learning rate iniziale
    lrf=0.01,   # Learning rate finale (frazione di lr0)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,  # Aumentato da 3 a 5
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # Data augmentation ottimizzata
    hsv_h=0.015,  # Hue augmentation
    hsv_s=0.7,    # Saturation augmentation
    hsv_v=0.4,    # Value augmentation
    degrees=10.0,  # Aumentato da 0.0
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,  # Aumentato da 0.0
    copy_paste=0.0,

    # Altri parametri
    verbose=True,
    save=True,
    save_period=10,  # Salva checkpoint ogni 10 epochs
    cache=False,
    plots=True,
    val=True,
)

print("\n" + "="*60)
print("✅ TRAINING COMPLETATO!")
print("="*60)

# Risultati
print(f"\n📊 Risultati salvati in: runs/train/yoloworld_generale")
print(f"📈 mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"📈 mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"💾 Best weights: runs/train/yoloworld_generale/weights/best.pt")
print(f"💾 Last weights: runs/train/yoloworld_generale/weights/last.pt")

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
print(f"🎯 Usa il modello: runs/train/yoloworld_generale/weights/best.pt")

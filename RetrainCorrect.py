
"""
Script per retrainare TUTTI i modelli prioritari con parametri FORZATI
Questo garantisce che gli epochs corretti vengano usati
"""

import subprocess
import time
import sys
from datetime import datetime, timedelta

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

print("╔" + "="*78 + "╗")
print("║" + " "*15 + "RETRAINING MODELLI CON PARAMETRI FORZATI" + " "*23 + "║")
print("╚" + "="*78 + "╝")

training_configs = [
    {
        'name': 'YOLO Generale',
        'script': 'SriptTrain/trainGenerale.py',
        'epochs': 50,
        'priority': 'MEDIA',
        'prev_map': 0.829
    },
    {
        'name': 'YOLO Glasses', 
        'script': 'SriptTrain/trainGlasses.py',
        'epochs': 120,
        'priority': 'ALTA',
        'prev_map': 0.630
    },
    {
        'name': 'YOLO Pen',
        'script': 'SriptTrain/trainpenne.py', 
        'epochs': 70,
        'priority': 'MEDIA',
        'prev_map': 0.813
    }
]

print(f"\n⏰ Inizio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
print("📊 Modelli da trainare:\n")

for config in training_configs:
    print(f"   • {config['name']}: {config['epochs']} epochs (mAP50-95 precedente: {config['prev_map']:.3f})")

print(f"\n⏱️  Tempo stimato totale: ~60-80 minuti\n")

response = input("Vuoi procedere? (s/n): ")
if response.lower() != 's':
    print("❌ Training annullato")
    sys.exit(0)

results = []
total_start = time.time()

for i, config in enumerate(training_configs, 1):
    print("\n" + "="*80)
    print(f"🚀 [{i}/{len(training_configs)}] Training {config['name']}")
    print(f"   Epochs: {config['epochs']} | Priorità: {config['priority']}")
    print("="*80 + "\n")
    
    start = time.time()
    
    # Esegui con parametri FORZATI via command line
    result = subprocess.run(
        [
            sys.executable,
            config['script'],
            '--epochs', str(config['epochs']),
            '--imgsz', '640',
            '--model', 'yolov8n.pt'
        ],
        text=True
    )
    
    elapsed = time.time() - start
    success = result.returncode == 0
    
    results.append({
        'name': config['name'],
        'epochs': config['epochs'],
        'time': elapsed,
        'success': success,
        'prev_map': config['prev_map']
    })
    
    status = "✅ OK" if success else "❌ ERRORE"
    print(f"\n{status} - Tempo: {format_time(elapsed)}\n")

total_time = time.time() - total_start

# Riepilogo
print("\n" + "╔" + "="*78 + "╗")
print("║" + " "*30 + "RIEPILOGO" + " "*38 + "║")
print("╚" + "="*78 + "╝\n")

print(f"⏰ Fine: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
print(f"{'Modello':<20} {'Epochs':<10} {'Tempo':<15} {'mAP Prev':<12} {'Stato':<10}")
print("-" * 80)

successful = sum(1 for r in results if r['success'])

for r in results:
    status = "✅ OK" if r['success'] else "❌ ERRORE"
    print(f"{r['name']:<20} {r['epochs']:<10} {format_time(r['time']):<15} {r['prev_map']:.3f}{'':<8} {status:<10}")

print("-" * 80)
print(f"{'TOTALE':<20} {'':<10} {format_time(total_time):<15}")
print("-" * 80)

print(f"\n📊 Risultati: {successful}/{len(results)} completati\n")

if successful == len(results):
    print("🎉 TRAINING COMPLETATO CON SUCCESSO!")
    print("\n💡 Prossimi passi:")
    print("   1. Controlla le metriche in runs/train/*/")
    print("   2. Confronta con i valori precedenti")
    print("   3. Usa i best weights per l'inferenza\n")
else:
    print("⚠️  Alcuni training sono falliti. Controlla i log.\n")

# Salva log
log_file = f"retrain_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_file, 'w') as f:
    f.write(f"RETRAINING LOG - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    for r in results:
        status = "COMPLETATO" if r['success'] else "FALLITO"
        f.write(f"Modello: {r['name']}\n")
        f.write(f"Epochs: {r['epochs']}\n")
        f.write(f"Tempo: {format_time(r['time'])}\n")
        f.write(f"mAP50-95 precedente: {r['prev_map']:.3f}\n")
        f.write(f"Stato: {status}\n")
        f.write("-"*80 + "\n\n")
    f.write(f"Tempo totale: {format_time(total_time)}\n")

print(f"💾 Log salvato in: {log_file}\n")
"""
Script per retrainare SOLO YOLO Glasses (il più debole)
con parametri FORZATI e corretti
"""

import subprocess
import time
from datetime import datetime, timedelta

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

print("="*80)
print("🎯 RETRAINING YOLO GLASSES CON PARAMETRI OTTIMIZZATI")
print("="*80)
print(f"\n⏰ Inizio: {datetime.now().strftime('%H:%M:%S')}")
print("\n📊 Configurazione:")
print("   • Modello: YOLO Glasses")
print("   • Epochs: 120 (FORZATI)")
print("   • Image size: 640")
print("   • Batch: 16")
print("   • Augmentation: AGGRESSIVA")
print("   • Tempo stimato: ~25-30 minuti\n")

input("⏸️  Premi INVIO per iniziare il training o CTRL+C per annullare...")

start = time.time()

# Esegui con parametri ESPLICITI
result = subprocess.run(
    [
        'python', 
        'SriptTrain/trainGlasses.py',
        '--epochs', '120',  # FORZA 120 epochs
        '--imgsz', '640',
        '--model', 'yolov8n.pt'
    ],
    text=True
)

elapsed = time.time() - start

print("\n" + "="*80)
if result.returncode == 0:
    print("✅ TRAINING COMPLETATO CON SUCCESSO!")
else:
    print("❌ ERRORE DURANTE IL TRAINING")
print(f"⏱️  Tempo totale: {format_time(elapsed)}")
print(f"⏰ Fine: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

print("\n📁 Risultati salvati in: runs/train/yolo_Glasses/")
print("💾 Best weights: runs/train/yolo_Glasses/weights/best.pt\n")


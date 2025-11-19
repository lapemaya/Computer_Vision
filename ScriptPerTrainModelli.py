"""
Script per trainare tutti i modelli YOLO
Esegue in sequenza:
- YOLO Generale (multi-classe)
- YOLO Keys
- YOLO Remote
- YOLO Glasses
- YOLO Wallet
- YOLO Pen
- YOLOv8-World X

Stampa i tempi di training per ogni modello
"""

import subprocess
import time
import sys
from datetime import datetime, timedelta
import os
import re

def format_time(seconds):
    """Formatta i secondi in formato leggibile"""
    return str(timedelta(seconds=int(seconds)))

def extract_metrics_from_output(output):
    """
    Estrae le metriche di training dall'output dello script

    Args:
        output: Output testuale dello script

    Returns:
        dict: Dizionario con le metriche estratte
    """
    metrics = {
        'precision': 'N/A',
        'recall': 'N/A',
        'map50': 'N/A',
        'map50_95': 'N/A'
    }

    # Pattern per estrarre le metriche
    patterns = {
        'precision': r'Precision:\s*([\d.]+|N/A)',
        'recall': r'Recall:\s*([\d.]+|N/A)',
        'map50': r'mAP50:\s*([\d.]+|N/A)',
        'map50_95': r'mAP50-95:\s*([\d.]+|N/A)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = match.group(1)

    return metrics

def run_training_script(script_path, script_name, description):
    """
    Esegue uno script di training e misura il tempo

    Args:
        script_path: Path dello script da eseguire
        script_name: Nome dello script (per il log)
        description: Descrizione del modello

    Returns:
        tempo_training: Tempo impiegato in secondi
        success: True se il training è andato a buon fine
        metrics: Dizionario con le metriche di training
    """
    print("\n" + "="*80)
    print(f"🚀 AVVIO TRAINING: {description}")
    print(f"📄 Script: {script_path}")
    print(f"⏰ Ora inizio: {datetime.now().strftime('%H:%M:%S')}")
    print("="*80 + "\n")

    start_time = time.time()

    metrics = {
        'precision': 'N/A',
        'recall': 'N/A',
        'map50': 'N/A',
        'map50_95': 'N/A'
    }

    # File temporaneo per salvare l'output
    output_file = f"temp_output_{script_name.replace(' ', '_')}.txt"

    try:
        # Esegui lo script con output in tempo reale e salva anche in un file
        with open(output_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            # Leggi l'output linea per linea in tempo reale
            for line in process.stdout:
                print(line, end='')  # Stampa in tempo reale
                f.write(line)  # Salva in file
                output_lines.append(line)

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)

            # Estrai le metriche dall'output salvato
            full_output = ''.join(output_lines)
            metrics = extract_metrics_from_output(full_output)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n" + "="*80)
        print(f"✅ TRAINING COMPLETATO: {description}")
        print(f"⏱️  Tempo impiegato: {format_time(elapsed_time)}")
        print(f"⏰ Ora fine: {datetime.now().strftime('%H:%M:%S')}")
        print("="*80 + "\n")

        # Rimuovi il file temporaneo
        if os.path.exists(output_file):
            os.remove(output_file)

        return elapsed_time, True, metrics

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Prova comunque a estrarre le metriche dall'output parziale
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                output = f.read()
                metrics = extract_metrics_from_output(output)
            os.remove(output_file)

        print("\n" + "="*80)
        print(f"❌ ERRORE DURANTE IL TRAINING: {description}")
        print(f"⏱️  Tempo trascorso prima dell'errore: {format_time(elapsed_time)}")
        print(f"💥 Codice errore: {e.returncode}")
        print("="*80 + "\n")

        return elapsed_time, False, metrics

    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Rimuovi il file temporaneo
        if os.path.exists(output_file):
            os.remove(output_file)

        print("\n" + "="*80)
        print(f"❌ ERRORE IMPREVISTO: {description}")
        print(f"⏱️  Tempo trascorso: {format_time(elapsed_time)}")
        print(f"💥 Errore: {e}")
        print("="*80 + "\n")

        return elapsed_time, False, metrics

def main():
    """Esegue tutti i training in sequenza"""

    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TRAINING AUTOMATICO TUTTI I MODELLI" + " "*23 + "║")
    print("╚" + "="*78 + "╝")
    print(f"\n🕐 Inizio training globale: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")

    # Definisci tutti i training da eseguire
    training_jobs = [
        {
            'script': 'SriptTrain/trainGenerale.py',
            'name': 'YOLO Generale',
            'description': 'YOLO Generale (Wallet, Keys, Glasses, Remote, Pen)'
        },
        # {
        #     'script': 'SriptTrain/trainyolokeys.py',
        #     'name': 'YOLO Keys',
        #     'description': 'YOLO Keys (specializzato)'
        # },
        # {
        #     'script': 'SriptTrain/trainRemote.py',
        #     'name': 'YOLO Remote',
        #     'description': 'YOLO Remote (specializzato)'
        # },
        {
            'script': 'SriptTrain/trainGlasses.py',
            'name': 'YOLO Glasses',
            'description': 'YOLO Glasses (specializzato)'
        },
        # {
        #     'script': 'SriptTrain/trainwallet.py',
        #     'name': 'YOLO Wallet',
        #     'description': 'YOLO Wallet (specializzato)'
        # },
        {
            'script': 'SriptTrain/trainpenne.py',
            'name': 'YOLO Pen',
            'description': 'YOLO Pen (specializzato)'
        },
        # {
        #     'script': 'TrainYoloWorld.py',
        #     'name': 'YOLOv8-World',
        #     'description': 'YOLOv8-World X (modello zero-shot)'
        # }
    ]

    # Verifica che tutti gli script esistano
    print("🔍 Verifica script di training...\n")
    missing_scripts = []
    for job in training_jobs:
        if not os.path.exists(job['script']):
            print(f"⚠️  Script mancante: {job['script']}")
            missing_scripts.append(job['script'])
        else:
            print(f"✅ Trovato: {job['script']}")

    if missing_scripts:
        print(f"\n❌ Impossibile procedere: {len(missing_scripts)} script mancanti")
        return

    print("\n✅ Tutti gli script sono presenti\n")

    # Esegui tutti i training
    results = []
    total_start_time = time.time()

    for i, job in enumerate(training_jobs, 1):
        print(f"\n📊 Progresso: {i}/{len(training_jobs)} modelli")

        elapsed, success, metrics = run_training_script(
            job['script'],
            job['name'],
            job['description']
        )

        results.append({
            'name': job['name'],
            'description': job['description'],
            'time': elapsed,
            'success': success,
            'metrics': metrics
        })

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Stampa il riepilogo finale
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*28 + "RIEPILOGO TRAINING" + " "*32 + "║")
    print("╚" + "="*78 + "╝\n")

    print(f"⏰ Ora fine: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    print(f"{'Modello':<25} {'Tempo':<15} {'Stato':<10}")
    print("-" * 80)

    successful = 0
    failed = 0

    for result in results:
        status = "✅ OK" if result['success'] else "❌ ERRORE"
        if result['success']:
            successful += 1
        else:
            failed += 1

        print(f"{result['name']:<25} {format_time(result['time']):<15} {status:<10}")

    print("-" * 80)
    print(f"{'TEMPO TOTALE':<25} {format_time(total_time):<15}")
    print("-" * 80)
    print(f"\n📊 Risultati: {successful} completati ✅ | {failed} falliti ❌")

    if failed == 0:
        print("\n🎉 TUTTI I TRAINING COMPLETATI CON SUCCESSO! 🎉\n")
    else:
        print(f"\n⚠️  {failed} training falliti. Controlla i log sopra per i dettagli.\n")

    # Salva il log in un file
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"{'TRAINING LOG - ' + datetime.now().strftime('%d/%m/%Y %H:%M:%S'):^100}\n")
        f.write("="*100 + "\n\n")

        # Riepilogo generale
        f.write(f"Tempo totale: {format_time(total_time)}\n")
        f.write(f"Modelli completati: {successful}/{len(results)}\n")
        f.write(f"Modelli falliti: {failed}/{len(results)}\n\n")

        f.write("="*100 + "\n")
        f.write(f"{'DETTAGLIO TRAINING':^100}\n")
        f.write("="*100 + "\n\n")

        for result in results:
            status = "✅ COMPLETATO" if result['success'] else "❌ FALLITO"
            f.write(f"Modello: {result['name']}\n")
            f.write(f"Descrizione: {result['description']}\n")
            f.write(f"Stato: {status}\n")
            f.write(f"Tempo: {format_time(result['time'])}\n")

            # Metriche di training
            f.write(f"\n📊 Metriche di Training:\n")
            f.write(f"   • Precision:     {result['metrics']['precision']}\n")
            f.write(f"   • Recall:        {result['metrics']['recall']}\n")
            f.write(f"   • mAP50:         {result['metrics']['map50']}\n")
            f.write(f"   • mAP50-95:      {result['metrics']['map50_95']}\n")
            f.write("\n" + "-"*100 + "\n\n")

        f.write("="*100 + "\n")
        f.write(f"{'RIEPILOGO STATISTICHE':^100}\n")
        f.write("="*100 + "\n\n")

        # Tabella riassuntiva
        f.write(f"{'Modello':<25} {'Tempo':<12} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12} {'Stato':<10}\n")
        f.write("-"*100 + "\n")

        for result in results:
            status = "OK" if result['success'] else "ERRORE"
            f.write(f"{result['name']:<25} "
                   f"{format_time(result['time']):<12} "
                   f"{result['metrics']['precision']:<12} "
                   f"{result['metrics']['recall']:<12} "
                   f"{result['metrics']['map50']:<12} "
                   f"{result['metrics']['map50_95']:<12} "
                   f"{status:<10}\n")

        f.write("-"*100 + "\n")
        f.write(f"{'TOTALE':<25} {format_time(total_time):<12}\n")
        f.write("="*100 + "\n")

    print(f"💾 Log dettagliato salvato in: {log_filename}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrotto dall'utente (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Errore critico: {e}")
        sys.exit(1)

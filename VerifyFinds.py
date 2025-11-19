from ultralytics import YOLO
import cv2
from pathlib import Path
import json
import os
import numpy as np
import torch

# Mappa tra le classi rilevate e i modelli corrispondenti
CLASS_TO_MODEL = {
    'Keys': 'runs/train/yolo_Keys/weights/best.pt',
    'keys': 'runs/train/yolo_Keys/weights/best.pt',
    'key': 'runs/train/yolo_Keys/weights/best.pt',
    'chiave': 'runs/train/yolo_Keys/weights/best.pt',
    'chiavi': 'runs/train/yolo_Keys/weights/best.pt',
    'Pen': 'runs/train/yolo_Pen/weights/best.pt',
    'Wallet': 'runs/train/yolo_Wallet/weights/best.pt',
    'wallet': 'runs/train/yolo_Wallet/weights/best.pt',
    'watch': 'runs/train/yolo_watches/weights/best.pt',
    'watches': 'runs/train/yolo_watches/weights/best.pt',
    'Glasses': 'runs/train/yolo_Glasses/weights/best.pt',
    "glasses":'runs/train/yolo_Glasses/weights/best.pt'
    # Aggiungi altre mappature se necessario
}

def load_bbox_data(bbox_file):
    """Legge i dati del bounding box dal file txt"""
    data = {}
    with open(bbox_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                data[key] = value
    return data

def calculate_distance(bbox1_data, bbox2_data):
    """
    Calcola la distanza tra i centri di due bounding box usando coordinate normalizzate
    """
    x1_norm = float(bbox1_data.get('x1_norm', '0'))
    y1_norm = float(bbox1_data.get('y1_norm', '0'))
    w1_norm = float(bbox1_data.get('width_norm', '0'))
    h1_norm = float(bbox1_data.get('height_norm', '0'))

    x2_norm = float(bbox2_data.get('x1_norm', '0'))
    y2_norm = float(bbox2_data.get('y1_norm', '0'))
    w2_norm = float(bbox2_data.get('width_norm', '0'))
    h2_norm = float(bbox2_data.get('height_norm', '0'))

    # Calcola i centri dei bounding box
    center1_x = x1_norm + w1_norm / 2
    center1_y = y1_norm + h1_norm / 2

    center2_x = x2_norm + w2_norm / 2
    center2_y = y2_norm + h2_norm / 2

    # Distanza euclidea tra i centri
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

    return distance

def verify_detections(segment_dir="runs/segment", output_dir="runs/verification", conf_threshold=0.8, delete_negatives=True, distance_threshold=0.1):
    """
    Verifica le detection usando i modelli specifici per ogni classe
    e opzionalmente elimina i file relativi ai negativi e ai duplicati troppo vicini

    distance_threshold: soglia di distanza normalizzata (0-1) sotto la quale due detection sono considerate troppo vicine
    """
    # Determina il device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  VerifyFinds usando: {device}")

    segment_path = Path(segment_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dizionario per tenere traccia dei modelli caricati
    loaded_models = {}

    # Report dei risultati
    verification_report = {
        'total_detections': 0,
        'verified': 0,
        'failed': 0,
        'too_close': 0,
        'deleted_files': 0,
        'by_class': {}
    }

    # Itera su ogni classe nella cartella segment
    for class_dir in segment_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\n{'='*60}")
        print(f"🔍 Verificando classe: {class_name}")
        print(f"{'='*60}")

        # Trova il modello corrispondente
        model_path = CLASS_TO_MODEL.get(class_name)
        print(class_name)
        if not model_path:
            print(f"⚠️  Nessun modello trovato per la classe '{class_name}'. Uso modello generale.")
            model_path = 'runs/train/yolo_Generale/weights/best.pt'

        # Carica il modello se non è già stato caricato
        if model_path not in loaded_models:
            print(f"📦 Caricamento modello: {model_path}")
            loaded_models[model_path] = YOLO(model_path)
            loaded_models[model_path].to(device)

        model = loaded_models[model_path]

        # Directory per questa classe
        frames_dir = class_dir / "frames"
        bboxes_dir = class_dir / "bboxes"
        crops_dir = class_dir / "crops"

        if not frames_dir.exists() or not bboxes_dir.exists():
            print(f"⚠️  Directory frames o bboxes non trovata per {class_name}")
            continue

        # Inizializza statistiche per questa classe
        verification_report['by_class'][class_name] = {
            'total': 0,
            'verified': 0,
            'failed': 0,
            'too_close': 0,
            'deleted': 0,
            'details': []
        }

        # Prima passa: raccogli tutti i bbox per controllare le distanze
        all_bboxes = []
        bbox_files = sorted(bboxes_dir.glob("*.txt"))

        for bbox_path in bbox_files:
            bbox_data = load_bbox_data(bbox_path)
            bbox_data['file_path'] = bbox_path
            all_bboxes.append(bbox_data)

        # Controlla quali detection sono troppo vicine
        too_close_indices = set()
        # Traccia quali indici devono essere mantenuti (hanno vinto il confronto)
        keep_indices = set(range(len(all_bboxes)))

        for i in range(len(all_bboxes)):
            # Salta se questo indice è già stato marcato per eliminazione
            if i not in keep_indices:
                continue

            for j in range(i + 1, len(all_bboxes)):
                # Salta se questo indice è già stato marcato per eliminazione
                if j not in keep_indices:
                    continue

                distance = calculate_distance(all_bboxes[i], all_bboxes[j])
                if distance < distance_threshold:
                    # Mantieni quella con confidence più alta, elimina l'altra
                    conf_i = float(all_bboxes[i].get('Confidence', '0'))
                    conf_j = float(all_bboxes[j].get('Confidence', '0'))

                    if conf_i < conf_j:
                        # Elimina i, mantieni j
                        too_close_indices.add(i)
                        keep_indices.discard(i)
                        print(f"⚠️  Detection troppo vicine! Distanza: {distance:.3f} - Mantengo detection {j} (conf {conf_j:.2f}), elimino {i} (conf {conf_i:.2f})")
                    else:
                        # Elimina j, mantieni i
                        too_close_indices.add(j)
                        keep_indices.discard(j)
                        print(f"⚠️  Detection troppo vicine! Distanza: {distance:.3f} - Mantengo detection {i} (conf {conf_i:.2f}), elimino {j} (conf {conf_j:.2f})")

        # Processa ogni frame
        frame_files = sorted(frames_dir.glob("*.jpg"))
        print(f"📊 Trovati {len(frame_files)} frame da verificare")

        for frame_idx, frame_path in enumerate(frame_files):
            # Trova il corrispondente file bbox
            bbox_filename = frame_path.stem.replace('_full', '') + '.txt'
            bbox_path = bboxes_dir / bbox_filename

            # Trova il corrispondente file crop
            crop_filename = frame_path.stem.replace('_full', '') + '.jpg'
            crop_path = crops_dir / crop_filename

            if not bbox_path.exists():
                print(f"⚠️  File bbox non trovato: {bbox_filename}")
                continue

            # Carica i dati del bbox originale
            bbox_data = load_bbox_data(bbox_path)
            original_conf = float(bbox_data.get('Confidence', '0'))

            verification_report['total_detections'] += 1
            verification_report['by_class'][class_name]['total'] += 1

            # Controlla se questa detection è troppo vicina ad altre
            is_too_close = False
            for idx, bbox in enumerate(all_bboxes):
                if bbox['file_path'] == bbox_path and idx in too_close_indices:
                    is_too_close = True
                    break

            if is_too_close:
                verification_report['too_close'] += 1
                verification_report['by_class'][class_name]['too_close'] += 1

                if delete_negatives:
                    files_to_delete = [frame_path, bbox_path]
                    if crop_path.exists():
                        files_to_delete.append(crop_path)

                    for file_path in files_to_delete:
                        try:
                            os.remove(file_path)
                            verification_report['deleted_files'] += 1
                            verification_report['by_class'][class_name]['deleted'] += 1
                        except Exception as e:
                            print(f"⚠️  Errore nell'eliminazione di {file_path.name}: {e}")

                    print(f"📏 TROPPO VICINO - {frame_path.name} (orig_conf: {original_conf:.2f}) → 🗑️  FILE ELIMINATI")
                else:
                    print(f"📏 TROPPO VICINO - {frame_path.name} (orig_conf: {original_conf:.2f})")
                continue

            # Carica il frame
            frame = cv2.imread(str(frame_path))

            # Esegui la predizione con il modello specifico
            results = model.predict(frame, conf=conf_threshold, verbose=False)

            # Verifica semplicemente se l'oggetto è presente nel frame
            verified = False
            best_conf = 0
            num_detections = 0

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # L'oggetto è stato rilevato nel frame
                verified = True
                num_detections = len(results[0].boxes)
                # Trova la confidence massima tra tutte le detection
                for box in results[0].boxes:
                    conf = box.conf[0].cpu().numpy()
                    best_conf = max(best_conf, conf)

            # Aggiorna statistiche
            status = "✅ VERIFICATO" if verified else "❌ NON VERIFICATO"
            if verified:
                verification_report['verified'] += 1
                verification_report['by_class'][class_name]['verified'] += 1
            else:
                verification_report['failed'] += 1
                verification_report['by_class'][class_name]['failed'] += 1

                # Elimina i file relativi ai negativi
                if delete_negatives:
                    files_to_delete = [frame_path, bbox_path]
                    if crop_path.exists():
                        files_to_delete.append(crop_path)

                    for file_path in files_to_delete:
                        try:
                            os.remove(file_path)
                            verification_report['deleted_files'] += 1
                            verification_report['by_class'][class_name]['deleted'] += 1
                        except Exception as e:
                            print(f"⚠️  Errore nell'eliminazione di {file_path.name}: {e}")

                print(f"{status} - {frame_path.name} (orig_conf: {original_conf:.2f})")
                continue

            detail = {
                'file': frame_path.name,
                'verified': verified,
                'original_confidence': float(original_conf),
                'num_objects_found': int(num_detections),
                'best_confidence': float(best_conf)
            }
            verification_report['by_class'][class_name]['details'].append(detail)

            print(f"{status} - {frame_path.name} (orig_conf: {original_conf:.2f}, objects_found: {num_detections}, best_conf: {best_conf:.2f})")

    # Salva il report in JSON
    report_path = output_path / "verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(verification_report, f, indent=2)

    # Stampa riepilogo
    print(f"\n{'='*60}")
    print(f"📊 RIEPILOGO VERIFICA")
    print(f"{'='*60}")
    print(f"Totale detection: {verification_report['total_detections']}")
    print(f"✅ Verificate: {verification_report['verified']} ({verification_report['verified']/max(verification_report['total_detections'],1)*100:.1f}%)")
    print(f"❌ Non verificate: {verification_report['failed']} ({verification_report['failed']/max(verification_report['total_detections'],1)*100:.1f}%)")
    print(f"📏 Troppo vicine: {verification_report['too_close']} ({verification_report['too_close']/max(verification_report['total_detections'],1)*100:.1f}%)")
    if delete_negatives:
        print(f"🗑️  File eliminati: {verification_report['deleted_files']}")

    print(f"\n📋 Dettaglio per classe:")
    for class_name, stats in verification_report['by_class'].items():
        total = stats['total']
        verified = stats['verified']
        failed = stats['failed']
        too_close = stats.get('too_close', 0)
        deleted = stats.get('deleted', 0)
        success_rate = (verified / max(total, 1)) * 100
        print(f"  {class_name}:")
        print(f"    Totale: {total}")
        print(f"    ✅ Verificate: {verified} ({success_rate:.1f}%)")
        print(f"    ❌ Non verificate: {failed}")
        print(f"    📏 Troppo vicine: {too_close}")
        if delete_negatives:
            print(f"    🗑️  File eliminati: {deleted}")

    print(f"\n💾 Report salvato in: {report_path}")

    return verification_report



if __name__ == "__main__":
    verify_detections(
        segment_dir="runs/segment",
        output_dir="runs/verification",
        conf_threshold=0.8,  # Soglia di confidenza per la verifica
        delete_negatives=True,  # Elimina i file relativi ai negativi
        distance_threshold=0.2  # Distanza normalizzata sotto la quale due detection sono considerate troppo vicine (0.1 = 10% della dimensione dell'immagine)
    )

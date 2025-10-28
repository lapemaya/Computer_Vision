"""
Script per salvare nel database gli oggetti rilevati da runs/segment/
"""

from pathlib import Path
from datetime import datetime
from dataset import ObjectDatabase
from DinoFeatureExtraction import extract_dino_features_rotations


def parse_bbox_file(bbox_path):
    """
    Legge il file bbox e restituisce i dati del bounding box

    Args:
        bbox_path: percorso del file .txt con i dati bbox

    Returns:
        dict con i dati bbox o None
    """
    try:
        with open(bbox_path, 'r') as f:
            lines = f.readlines()

        data = {}
        bbox = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith('---'):
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                # Mappa i campi
                if key == 'frame':
                    data['frame'] = int(value)
                elif key == 'object id':
                    data['object_id'] = int(value)
                elif key == 'class':
                    data['class'] = value
                elif key == 'confidence':
                    data['confidence'] = float(value)
                elif key == 'x1':
                    bbox['x1'] = int(value)
                elif key == 'y1':
                    bbox['y1'] = int(value)
                elif key == 'x2':
                    bbox['x2'] = int(value)
                elif key == 'y2':
                    bbox['y2'] = int(value)
                elif key == 'width':
                    bbox['width'] = int(value)
                elif key == 'height':
                    bbox['height'] = int(value)

        if bbox:
            data['bbox'] = bbox

        return data
    except Exception as e:
        print(f"⚠️ Errore nel leggere {bbox_path}: {e}")
        return None


def process_segment_results(segment_dir="runs/segment", db_path="detections.db",
                            feature_dir="features_db", crops_dir="crops_db",
                            similarity_threshold=0.85, delete_after_import=True,rotations=8):
    print("🚀 Avvio salvataggio oggetti nel database...")
    print(f"   Cartella sorgente: {segment_dir}")
    print(f"   Database: {db_path}")
    print(f"   Soglia similarità: {similarity_threshold}")
    print(f"   Elimina file dopo importazione: {delete_after_import}")
    """
    Processa tutti i risultati di segmentazione e li aggiunge al database

    Args:
        segment_dir: directory con i risultati di segmentazione
        db_path: percorso del database SQLite
        feature_dir: directory per salvare le feature
        crops_dir: directory per salvare le immagini croppate
        similarity_threshold: soglia di similarità per evitare duplicati
        delete_after_import: se True, elimina i file dopo l'importazione
    """
    segment_path = Path(segment_dir)

    if not segment_path.exists():
        print(f"❌ Directory {segment_dir} non trovata")
        return

    # Inizializza il database
    print(f"📊 Inizializzazione database: {db_path}")
    db = ObjectDatabase(db_path=db_path, feature_dir=feature_dir, crops_dir=crops_dir)

    # Statistiche
    stats = {
        'total_processed': 0,
        'total_added': 0,
        'total_skipped': 0,
        'total_files_deleted': 0,
        'by_class': {}
    }

    # Lista per tracciare i file da eliminare
    files_to_delete = []

    # Processa ogni classe
    for class_dir in sorted(segment_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\n{'='*60}")
        print(f"📁 Processando classe: {class_name}")
        print(f"{'='*60}")

        crops_dir_path = class_dir / "crops"
        bboxes_dir_path = class_dir / "bboxes"
        frames_dir_path = class_dir / "frames"

        if not crops_dir_path.exists():
            print(f"⚠️ Cartella crops non trovata per {class_name}")
            continue

        # Inizializza statistiche per questa classe
        stats['by_class'][class_name] = {
            'processed': 0,
            'added': 0,
            'skipped': 0,
            'files_deleted': 0
        }

        # Processa ogni crop
        for crop_file in sorted(crops_dir_path.glob("*.jpg")):
            stats['total_processed'] += 1
            stats['by_class'][class_name]['processed'] += 1

            # Costruisci i percorsi dei file correlati
            base_name = crop_file.stem  # es: frame00045_obj00_Keys
            bbox_file = bboxes_dir_path / f"{base_name}.txt"
            frame_file = frames_dir_path / f"{base_name}_full.jpg"

            # Aggiungi i file alla lista di eliminazione
            files_to_delete.append(crop_file)
            if bbox_file.exists():
                files_to_delete.append(bbox_file)
            if frame_file.exists():
                files_to_delete.append(frame_file)

            print(f"\n  🔍 Processando: {crop_file.name}")

            # Estrai features con DINOv2 CON ROTAZIONI MULTIPLE (8 rotazioni ogni 45°)
            try:
                print(f"     Estrazione features DINOv2 con rotation invariance...")
                features = extract_dino_features_rotations(str(crop_file), show=False,num_rotations=rotations)
                print(f"     ✓ Estratte {len(features)} rotazioni")
            except Exception as e:
                print(f"     ❌ Errore nell'estrazione features: {e}")
                stats['total_skipped'] += 1
                stats['by_class'][class_name]['skipped'] += 1
                continue

            # Leggi bbox se esiste
            bbox_data = None
            confidence = None
            if bbox_file.exists():
                bbox_info = parse_bbox_file(bbox_file)
                if bbox_info:
                    bbox_data = bbox_info.get('bbox')
                    confidence = bbox_info.get('confidence')

            # Percorso frame completo
            full_frame_path = str(frame_file) if frame_file.exists() else None

            # Aggiungi al database
            try:
                object_id = db.add_object(
                    class_name=class_name,
                    features=features,
                    detection_datetime=datetime.now(),
                    confidence=confidence,
                    image_path=str(crop_file),
                    bbox=bbox_data,
                    similarity_threshold=similarity_threshold,

                )

                if object_id:
                    # Controlla se è stato aggiunto o era duplicato
                    # (add_object ritorna l'ID esistente se trova un duplicato)
                    stats['total_added'] += 1
                    stats['by_class'][class_name]['added'] += 1
                else:
                    stats['total_skipped'] += 1
                    stats['by_class'][class_name]['skipped'] += 1

            except Exception as e:
                print(f"     ❌ Errore nell'aggiunta al database: {e}")
                stats['total_skipped'] += 1
                stats['by_class'][class_name]['skipped'] += 1

    # Elimina i file processati se richiesto
    if delete_after_import:
        print(f"\n{'='*60}")
        print(f"🗑️  ELIMINAZIONE FILE PROCESSATI")
        print(f"{'='*60}")

        for file_path in files_to_delete:
            try:
                if file_path.exists():
                    file_path.unlink()
                    stats['total_files_deleted'] += 1

                    # Aggiorna stats per classe
                    for class_name in stats['by_class'].keys():
                        if class_name in str(file_path):
                            stats['by_class'][class_name]['files_deleted'] += 1
                            break
            except Exception as e:
                print(f"⚠️ Errore nell'eliminare {file_path}: {e}")

        print(f"✓ Eliminati {stats['total_files_deleted']} file da runs/segment/")

        # Elimina cartelle vuote
        for class_dir in sorted(segment_path.iterdir()):
            if not class_dir.is_dir():
                continue

            for subdir_name in ['crops', 'bboxes', 'frames']:
                subdir = class_dir / subdir_name
                if subdir.exists() and subdir.is_dir():
                    try:
                        # Controlla se è vuota
                        if not any(subdir.iterdir()):
                            subdir.rmdir()
                            print(f"✓ Eliminata cartella vuota: {subdir}")
                    except Exception as e:
                        print(f"⚠️ Errore nell'eliminare {subdir}: {e}")

    # Stampa statistiche finali
    print(f"\n{'='*60}")
    print(f"📊 STATISTICHE FINALI")
    print(f"{'='*60}")
    print(f"Oggetti processati:  {stats['total_processed']}")
    print(f"Oggetti aggiunti:    {stats['total_added']}")
    print(f"Oggetti saltati:     {stats['total_skipped']}")
    if delete_after_import:
        print(f"File eliminati:      {stats['total_files_deleted']}")

    print(f"\n📋 Dettaglio per classe:")
    for class_name, class_stats in stats['by_class'].items():
        print(f"  {class_name}:")
        print(f"    - Processati: {class_stats['processed']}")
        print(f"    - Aggiunti:   {class_stats['added']}")
        print(f"    - Saltati:    {class_stats['skipped']}")
        if delete_after_import:
            print(f"    - File eliminati: {class_stats['files_deleted']}")

    # Esporta il database in JSON
    export_path = "database_export.json"
    db.export_to_json(export_path)

    # Mostra le classi nel database
    print(f"\n📊 Classi nel database:")
    for cls in db.get_all_classes():
        print(f"  {cls['class_name']}: {cls['object_count']} oggetti")

    db.close()
    print(f"\n✅ Processo completato!")


if __name__ == "__main__":
    # Configurazione
    SEGMENT_DIR = "runs/segment"
    DB_PATH = "detections.db"
    FEATURE_DIR = "features_db"
    CROPS_DIR = "crops_db"
    SIMILARITY_THRESHOLD = 0.85  # Soglia per evitare duplicati
    DELETE_AFTER_IMPORT = True  # Elimina i file dopo l'importazione

    process_segment_results(
        segment_dir=SEGMENT_DIR,
        db_path=DB_PATH,
        feature_dir=FEATURE_DIR,
        crops_dir=CROPS_DIR,
        similarity_threshold=SIMILARITY_THRESHOLD,
        delete_after_import=DELETE_AFTER_IMPORT
    )

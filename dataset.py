import json
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
import shutil


class ObjectDatabase:
    """Database per gestire classi e oggetti rilevati con feature DINOv2"""

    def __init__(self, db_path="detections.db", feature_dir="features_db", crops_dir="crops_db"):
        """
        Inizializza il database

        Args:
            db_path: percorso del file database SQLite
            feature_dir: directory dove salvare le feature numpy
            crops_dir: directory dove salvare le immagini croppate
        """
        self.db_path = db_path
        self.feature_dir = Path(feature_dir)
        self.crops_dir = Path(crops_dir)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self._init_database()

    def _init_database(self):
        """Crea le tabelle del database"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Tabella classi
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS classes (
                class_id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Tabella oggetti
        # Uso TEXT per date/ora perche' salviamo ISO strings
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS objects (
                object_id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id INTEGER NOT NULL,
                feature_path TEXT NOT NULL,
                crop_image_path TEXT,
                detection_date TEXT NOT NULL,
                detection_time TEXT NOT NULL,
                confidence REAL,
                image_path TEXT,
                bbox_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (class_id) REFERENCES classes (class_id)
            )
        ''')

        # Migrazione: aggiungi crop_image_path se non esiste
        try:
            self.cursor.execute("SELECT crop_image_path FROM objects LIMIT 1")
        except sqlite3.OperationalError:
            # La colonna non esiste, aggiungila
            print("⚙️ Aggiornamento schema database: aggiunta colonna crop_image_path")
            self.cursor.execute("ALTER TABLE objects ADD COLUMN crop_image_path TEXT")

        self.conn.commit()

    def add_class(self, class_name):
        """
        Aggiunge una nuova classe

        Args:
            class_name: nome della classe

        Returns:
            class_id della classe aggiunta o esistente
        """
        try:
            self.cursor.execute(
                "INSERT INTO classes (class_name) VALUES (?)",
                (class_name,)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            # Classe già esistente, recupera l'ID
            self.cursor.execute(
                "SELECT class_id FROM classes WHERE class_name = ?",
                (class_name,)
            )
            return self.cursor.fetchone()[0]

    def add_object(self, class_name, features, detection_datetime=None,
                   confidence=None, image_path=None, bbox=None, similarity_threshold=0.85):
        """
        Aggiunge un nuovo oggetto rilevato

        Args:
            class_name: nome della classe
            features: array numpy con le feature DINOv2 OPPURE lista di 8 array (uno per rotazione)
            detection_datetime: datetime del rilevamento (default: now)
            confidence: confidence della detection
            image_path: percorso dell'immagine completa (usato come crop source se fornito)
            bbox: bounding box (x, y, w, h)
            similarity_threshold: soglia di similarità per evitare duplicati

        Returns:
            object_id dell'oggetto aggiunto o None se è un duplicato
        """
        # Ottieni o crea la classe
        crop_image_path=image_path
        class_id = self.add_class(class_name)

        # Gestisci features: può essere un singolo array o una lista di 8 array (rotazioni)
        is_multi_rotation = isinstance(features, list)

        if is_multi_rotation:
            # Features con rotazioni multiple (8 rotazioni)
            # SALVA TUTTE le rotazioni, USA SOLO LA PRIMA (0°) per il confronto
            features_list = [np.asarray(f) for f in features]
            feats_to_save = features_list  # Salva tutte le rotazioni
            feat_for_comparison = features_list[0]  # Usa solo rotazione a 0° per confronto
        else:
            # Singola feature
            feat_single = np.asarray(features)
            features_list = [feat_single]
            feats_to_save = features_list
            feat_for_comparison = feat_single

        # Controllo similarità: confronta la rotazione a 0° del nuovo oggetto
        # con TUTTE le rotazioni degli oggetti esistenti
        try:
            existing_objects = self.get_class_objects(class_name)
        except Exception:
            existing_objects = []

        max_similarity_same_class = 0.0
        if len(existing_objects) > 0:
            for obj in existing_objects:
                try:
                    # Confronta la rotazione a 0° del nuovo oggetto con TUTTE le rotazioni salvate
                    max_similarity = 0.0
                    existing_features_list = obj.get('features_list', [])

                    if not existing_features_list:
                        continue

                    feat_for_comparison_norm = np.linalg.norm(feat_for_comparison)
                    if feat_for_comparison_norm == 0:
                        continue

                    # Confronta con ogni rotazione salvata dell'oggetto esistente
                    for existing_feat in existing_features_list:
                        existing_feat = np.asarray(existing_feat)
                        existing_norm = np.linalg.norm(existing_feat)
                        if existing_norm == 0:
                            continue
                        similarity = float(np.dot(feat_for_comparison, existing_feat) / (feat_for_comparison_norm * existing_norm))
                        max_similarity = max(max_similarity, similarity)

                    # Traccia la similarità massima nella stessa classe
                    max_similarity_same_class = max(max_similarity_same_class, max_similarity)

                except Exception as e:
                    continue

                if max_similarity >= similarity_threshold:
                    print(f"⚠️ Oggetto simile trovato (id={obj['object_id']}, sim={max_similarity:.3f}) nella classe '{class_name}'. Non verrà aggiunto.")
                    return None

        # Nuovo controllo: cerca oggetti simili in TUTTE le classi (cross-class)
        max_similarity_cross_class = 0.0
        if feat_for_comparison is not None:
            try:
                # Confronta la rotazione a 0° del nuovo con tutte le rotazioni degli esistenti
                max_cross_similarity = 0.0
                matched_obj = None

                # Recupera tutti gli oggetti
                self.cursor.execute('''
                    SELECT o.object_id, o.feature_path, c.class_name
                    FROM objects o
                    JOIN classes c ON o.class_id = c.class_id
                ''')
                all_objects = self.cursor.fetchall()

                feat_for_comparison_norm = np.linalg.norm(feat_for_comparison)
                if feat_for_comparison_norm == 0:
                    raise ValueError("Feature norm is zero")

                for obj_row in all_objects:
                    obj_id, feature_path, obj_class_name = obj_row

                    # Ignora oggetti della stessa classe (già gestiti sopra)
                    if obj_class_name == class_name:
                        continue

                    try:
                        # Carica tutte le feature esistenti (8 rotazioni)
                        if Path(feature_path).exists():
                            # Se è un JSON con lista di path, caricali tutti
                            if feature_path.endswith('_rotations.json'):
                                with open(feature_path, 'r') as f:
                                    feature_paths = json.load(f)
                                existing_features_list = []
                                for fp in feature_paths:
                                    if Path(fp).exists():
                                        existing_features_list.append(np.load(fp))
                            else:
                                # File singolo .npy - potrebbe contenere array 2D con 8 rotazioni
                                loaded = np.load(feature_path)
                                if loaded.ndim == 2:
                                    # Array 2D: ogni riga è una rotazione
                                    existing_features_list = [loaded[i] for i in range(loaded.shape[0])]
                                else:
                                    # Array 1D: singola feature
                                    existing_features_list = [loaded]

                            # Confronta con tutte le rotazioni salvate
                            for existing_feat in existing_features_list:
                                existing_norm = np.linalg.norm(existing_feat)
                                if existing_norm == 0:
                                    continue
                                similarity = float(np.dot(feat_for_comparison, existing_feat) / (feat_for_comparison_norm * existing_norm))
                                if similarity > max_cross_similarity:
                                    max_cross_similarity = similarity
                                    matched_obj = (obj_id, obj_class_name)
                    except:
                        continue

                max_similarity_cross_class = max_cross_similarity

                if max_cross_similarity >= similarity_threshold and matched_obj:
                    print(f"⚠️ Oggetto simile trovato in altra classe (id={matched_obj[0]}, class={matched_obj[1]}, sim={max_cross_similarity:.3f}). Non verrà aggiunto.")
                    return None

            except Exception as e:
                pass

        # Usa datetime corrente se non specificato
        if detection_datetime is None:
            detection_datetime = datetime.now()

        # Converti la data e l'ora in stringhe ISO
        detection_date = detection_datetime.date().isoformat()
        detection_time = detection_datetime.time().isoformat()

        # Salva TUTTE le feature (8 rotazioni) in un singolo file numpy 2D
        # Array shape: (num_rotations, feature_dim)
        features_array = np.array(feats_to_save)
        feature_filename = f"class_{class_id}_{detection_datetime.strftime('%Y%m%d_%H%M%S')}.npy"
        feature_path = self.feature_dir / class_name / feature_filename
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, features_array)

        # Salva l'immagine croppata se fornita
        saved_crop_path = None
        if crop_image_path and Path(crop_image_path).exists():
            crop_filename = f"crop_{class_id}_{detection_datetime.strftime('%Y%m%d_%H%M%S')}{Path(crop_image_path).suffix}"
            saved_crop_path = self.crops_dir / class_name / crop_filename
            saved_crop_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(crop_image_path, saved_crop_path)

        # Converti bbox in stringa JSON se presente
        bbox_json = json.dumps(bbox) if bbox is not None else None

        # Inserisci nel database
        self.cursor.execute('''
            INSERT INTO objects 
            (class_id, feature_path, crop_image_path, detection_date, detection_time, 
             confidence, image_path, bbox_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            class_id,
            str(feature_path),
            str(saved_crop_path) if saved_crop_path else None,
            detection_date,
            detection_time,
            confidence,
            image_path,
            bbox_json
        ))

        self.conn.commit()
        object_id = self.cursor.lastrowid

        rotation_info = f" (salvate {len(feats_to_save)} rotazioni)" if is_multi_rotation else ""
        similarity_info = ""
        if max_similarity_same_class > 0 or max_similarity_cross_class > 0:
            similarity_info = f" | Similarità max: stessa classe={max_similarity_same_class:.3f}, cross-classe={max_similarity_cross_class:.3f}"
        print(f"✓ Oggetto {object_id} aggiunto: {class_name} - {detection_datetime}{rotation_info}{similarity_info}")
        return object_id

    def get_class_objects(self, class_name, limit=None):
        """
        Recupera tutti gli oggetti di una classe

        Args:
            class_name: nome della classe
            limit: numero massimo di oggetti da recuperare

        Returns:
            lista di dizionari con i dati degli oggetti
        """
        query = '''
            SELECT o.object_id, o.feature_path, o.crop_image_path, o.detection_date, 
                   o.detection_time, o.confidence, o.image_path, o.bbox_data
            FROM objects o
            JOIN classes c ON o.class_id = c.class_id
            WHERE c.class_name = ?
            ORDER BY o.detection_date DESC, o.detection_time DESC
        '''

        if limit:
            query += f" LIMIT {limit}"

        self.cursor.execute(query, (class_name,))
        rows = self.cursor.fetchall()

        objects = []
        for row in rows:
            feature_path = row[1]

            # Carica le features (può essere array 2D con più rotazioni o array 1D singolo)
            features = None
            features_list = []
            if Path(feature_path).exists():
                loaded = np.load(feature_path)
                if loaded.ndim == 2:
                    # Array 2D: ogni riga è una rotazione
                    features_list = [loaded[i] for i in range(loaded.shape[0])]
                    features = loaded[0]  # Prima rotazione per compatibilità
                else:
                    # Array 1D: singola feature
                    features = loaded
                    features_list = [loaded]

            obj = {
                'object_id': row[0],
                'feature_path': feature_path,
                'features': features,  # Prima rotazione per compatibilità
                'features_list': features_list,  # Tutte le rotazioni
                'crop_image_path': row[2],
                'detection_date': row[3],
                'detection_time': row[4],
                'confidence': row[5],
                'image_path': row[6],
                'bbox': json.loads(row[7]) if row[7] else None
            }
            objects.append(obj)

        return objects

    def get_all_classes(self):
        """Recupera tutte le classi con il conteggio degli oggetti"""
        self.cursor.execute('''
            SELECT c.class_id, c.class_name, COUNT(o.object_id) as object_count
            FROM classes c
            LEFT JOIN objects o ON c.class_id = o.class_id
            GROUP BY c.class_id, c.class_name
            ORDER BY c.class_name
        ''')

        return [
            {'class_id': row[0], 'class_name': row[1], 'object_count': row[2]}
            for row in self.cursor.fetchall()
        ]

    def search_similar_objects(self, features, class_name=None, threshold=0.85, limit=10):
        """
        Cerca oggetti simili nel database usando similarità coseno

        Args:
            features: feature DINOv2 dell'oggetto da cercare
            class_name: filtra per classe (opzionale)
            threshold: soglia di similarità minima
            limit: numero massimo di risultati

        Returns:
            lista di oggetti simili ordinati per similarità
        """
        # Recupera oggetti da confrontare
        if class_name:
            objects = self.get_class_objects(class_name)
        else:
            # Recupera tutti gli oggetti
            self.cursor.execute('''
                SELECT o.object_id, o.feature_path, o.crop_image_path, o.detection_date, 
                       o.detection_time, c.class_name, o.confidence
                FROM objects o
                JOIN classes c ON o.class_id = c.class_id
            ''')
            rows = self.cursor.fetchall()
            objects = [
                {
                    'object_id': row[0],
                    'features': np.load(row[1]),
                    'crop_image_path': row[2],
                    'detection_date': row[3],
                    'detection_time': row[4],
                    'class_name': row[5],
                    'confidence': row[6]
                }
                for row in rows
            ]

        # Calcola similarità
        similar_objects = []
        for obj in objects:
            similarity = np.dot(features, obj['features']) / \
                         (np.linalg.norm(features) * np.linalg.norm(obj['features']))
            print
            if similarity >= threshold:
                obj['similarity'] = float(similarity)
                similar_objects.append(obj)

        # Ordina per similarità decrescente
        similar_objects.sort(key=lambda x: x['similarity'], reverse=True)

        return similar_objects[:limit]

    def export_to_json(self, output_path):
        """Esporta il database in formato JSON"""
        classes = self.get_all_classes()

        export_data = {
            'export_date': datetime.now().isoformat(),
            'classes': []
        }

        for cls in classes:
            objects = self.get_class_objects(cls['class_name'])

            class_data = {
                'class_id': cls['class_id'],
                'class_name': cls['class_name'],
                'object_count': cls['object_count'],
                'objects': [
                    {
                        'object_id': obj['object_id'],
                        # detection_date/time sono salvati come ISO stringhe
                        'detection_date': obj['detection_date'],
                        'detection_time': obj['detection_time'],
                        'confidence': obj['confidence'],
                        'image_path': obj['image_path'],
                        'crop_image_path': obj['crop_image_path'],
                        'bbox': obj['bbox']
                    }
                    for obj in objects
                ]
            }
            export_data['classes'].append(class_data)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Database esportato in: {output_path}")

    def delete_all_objects(self, delete_files=True):
        """
        Elimina tutti gli oggetti dal database mantenendo le classi

        Args:
            delete_files: se True, elimina anche i file delle feature e delle immagini croppate

        Returns:
            numero di oggetti eliminati
        """
        # Conta gli oggetti prima di eliminarli
        self.cursor.execute("SELECT COUNT(*) FROM objects")
        count = self.cursor.fetchone()[0]

        if count == 0:
            print("ℹ️ Nessun oggetto da eliminare")
            return 0

        # Se richiesto, elimina anche i file
        if delete_files:
            print("🗑️ Eliminazione file delle feature e immagini croppate...")

            # Recupera tutti i percorsi dei file dal database
            self.cursor.execute("SELECT feature_path, crop_image_path FROM objects")
            file_paths = self.cursor.fetchall()

            deleted_features = 0
            deleted_crops = 0

            for feature_path, crop_path in file_paths:
                # Elimina file feature
                if feature_path:
                    feature_file = Path(feature_path)
                    if feature_file.exists():
                        try:
                            feature_file.unlink()
                            deleted_features += 1
                        except Exception as e:
                            print(f"⚠️ Errore nell'eliminare {feature_file}: {e}")

                # Elimina file crop
                if crop_path:
                    crop_file = Path(crop_path)
                    if crop_file.exists():
                        try:
                            crop_file.unlink()
                            deleted_crops += 1
                        except Exception as e:
                            print(f"⚠️ Errore nell'eliminare {crop_file}: {e}")

            # ELIMINA TUTTI i file rimanenti nelle cartelle features_db e crops_db
            # (anche quelli eventualmente orfani non referenziati nel DB)
            print("🧹 Pulizia file orfani...")

            # Pulisci features_db - elimina tutti i file .npy e .json
            if self.feature_dir.exists():
                for class_dir in self.feature_dir.iterdir():
                    if class_dir.is_dir():
                        for feature_file in class_dir.glob("*.npy"):
                            try:
                                feature_file.unlink()
                                deleted_features += 1
                                print(f"  ✓ Eliminato file orfano: {feature_file.name}")
                            except Exception as e:
                                print(f"  ⚠️ Errore nell'eliminare {feature_file}: {e}")

                        # Elimina anche eventuali file JSON di rotazioni (vecchio formato)
                        for json_file in class_dir.glob("*.json"):
                            try:
                                json_file.unlink()
                                print(f"  ✓ Eliminato file JSON: {json_file.name}")
                            except Exception as e:
                                print(f"  ⚠️ Errore nell'eliminare {json_file}: {e}")

            # Pulisci crops_db - elimina tutte le immagini
            if self.crops_dir.exists():
                for class_dir in self.crops_dir.iterdir():
                    if class_dir.is_dir():
                        for crop_file in class_dir.glob("*.jpg"):
                            try:
                                crop_file.unlink()
                                deleted_crops += 1
                                print(f"  ✓ Eliminato crop orfano: {crop_file.name}")
                            except Exception as e:
                                print(f"  ⚠️ Errore nell'eliminare {crop_file}: {e}")

                        for crop_file in class_dir.glob("*.png"):
                            try:
                                crop_file.unlink()
                                deleted_crops += 1
                                print(f"  ✓ Eliminato crop orfano: {crop_file.name}")
                            except Exception as e:
                                print(f"  ⚠️ Errore nell'eliminare {crop_file}: {e}")

            # Elimina cartelle vuote
            for class_dir in self.feature_dir.iterdir():
                if class_dir.is_dir() and not any(class_dir.iterdir()):
                    try:
                        class_dir.rmdir()
                        print(f"  ✓ Eliminata cartella vuota: {class_dir.name}")
                    except Exception as e:
                        print(f"  ⚠️ Errore nell'eliminare cartella {class_dir}: {e}")

            for class_dir in self.crops_dir.iterdir():
                if class_dir.is_dir() and not any(class_dir.iterdir()):
                    try:
                        class_dir.rmdir()
                        print(f"  ✓ Eliminata cartella vuota: {class_dir.name}")
                    except Exception as e:
                        print(f"  ⚠️ Errore nell'eliminare cartella {class_dir}: {e}")

            print(f"✓ Eliminati {deleted_features} file di feature e {deleted_crops} immagini croppate")

        # Elimina tutti gli oggetti dal database
        self.cursor.execute("DELETE FROM objects")
        self.conn.commit()

        print(f"✓ Eliminati {count} oggetti dal database. Le classi sono state mantenute.")
        return count

    def close(self):
        """Chiude la connessione al database"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        self.close()


# Esempio d'uso (disattivato): import e codice di esempio sono commentati nel repository
if __name__ == "__main__":
    # Questo modulo è pensato per essere importato; il blocco main è intenzionalmente minimale.
    db = ObjectDatabase(db_path="detections.db", feature_dir="features_db")
    db.delete_all_objects()
    db.export_to_json("database_export.json")
    db.close()

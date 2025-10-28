# 🚁 Progetto Drone - Dataset Creation & Object Retrieval Pipeline

## 📋 Indice
1. [Panoramica del Progetto](#panoramica-del-progetto)
2. [Architettura del Sistema](#architettura-del-sistema)
3. [Tecnologie e Tecniche Utilizzate](#tecnologie-e-tecniche-utilizzate)
4. [Installazione](#installazione)
5. [Pipeline di Creazione Dataset](#pipeline-di-creazione-dataset)
6. [Sistema di Retrieval](#sistema-di-retrieval)
7. [Guida all'Esecuzione](#guida-allesecuzione)
8. [Struttura Output](#struttura-output)
9. [Parametri e Configurazione](#parametri-e-configurazione)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Panoramica del Progetto

Sistema automatizzato per la creazione di dataset di oggetti da video e il successivo retrieval (ricerca) di oggetti specifici all'interno di nuovi video. Il progetto si compone di due componenti principali:

1. **Pipeline di Creazione Dataset** (`pipelineCreazioneDataset.py`): estrae, verifica, deduplica e indicizza automaticamente oggetti da video
2. **Sistema di Retrieval** (`ProvaRetrival.py`): cerca oggetti specifici (già nel database) all'interno di nuovi video

### 🎯 Casi d'Uso
- **Sorveglianza**: identificare oggetti personali (chiavi, portafogli, occhiali) in video di sicurezza
- **Inventory tracking**: tracciare oggetti specifici in ambienti industriali
- **Object matching**: trovare corrispondenze tra oggetti rilevati in momenti diversi

---

## 🏗️ Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 1: SEGMENTAZIONE VIDEO (ProvaScanDaVideo.py)          │
│  - YOLO detection frame-by-frame                            │
│  - Estrazione crops + bounding boxes                        │
│  - Output: runs/segment/{classe}/crops,bboxes,frames       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 2: VERIFICA DETECTIONS (VerifyFinds.py)               │
│  - Filtraggio confidence threshold                          │
│  - Controllo coerenza spaziale/temporale                    │
│  - Rimozione falsi positivi                                 │
│  - Output: runs/verification/                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 3: DEDUPLICAZIONE (DinoFeatureExtraction.py)          │
│  - Estrazione feature DINOv2                                │
│  - Clustering per similarità                                │
│  - Rimozione duplicati                                      │
│  - Output: runs/verification/similarity/                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 4: IMPORT IN DATABASE (SaveInDatabaseFounds.py)       │
│  - Estrazione features con rotazioni multiple               │
│  - Salvataggio in SQLite (detections.db)                    │
│  - Organizzazione crops e features                          │
│  - Output: crops_db/, features_db/, detections.db           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │  DATABASE   │
              └─────────────┘
```

---

## 🔬 Tecnologie e Tecniche Utilizzate

### 1. **Object Detection - YOLOv8**
- **Libreria**: Ultralytics YOLO
- **Modelli utilizzati**:
  - Modello generale: `runs/train/yolo_Generale/weights/best.pt`
  - Modelli specifici per classe: `runs/train/yolo_{classe}/weights/best.pt`
- **Tecnica**: detection frame-by-frame con confidence threshold configurabile
- **Configurazione**:
  - Confidence iniziale: 0.46 (segmentazione)
  - Confidence verifica: 0.75-0.8 (filtraggio)
  - Vid_stride: 1 (analizza ogni frame)

**Perché YOLO?**
- Velocità: real-time processing
- Accuratezza: stato dell'arte per object detection
- Flessibilità: training su classi custom

### 2. **Feature Extraction - DINOv2 (Facebook Research)**
- **Modello**: `dinov2_vitb14` (Vision Transformer Base, patch size 14)
- **Risoluzione input**: 518×518 pixels
- **Output**: embedding di 768 dimensioni (feature vector)
- **Normalizzazione**: ImageNet mean/std (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)

**Perché DINOv2?**
- **Self-supervised learning**: embeddings semantici robusti senza annotazioni
- **Invarianza**: resistente a cambiamenti di illuminazione, prospettiva, scala
- **Discriminatività**: eccellente per distinguere istanze simili della stessa classe

**Preprocessing applicato**:
```python
transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                       std=(0.229, 0.224, 0.225))
])
```

### 3. **Rotation Augmentation per Robustezza**
Durante l'import nel database, ogni crop viene ruotato in N angolazioni diverse (default: 16 rotazioni):
- **Angoli**: 0°, 22.5°, 45°, 67.5°, ..., 337.5° (360°/16)
- **Scopo**: rendere il matching invariante alla rotazione
- **Implementazione**: per ogni crop, vengono estratte ed aggregate 16 feature vectors
- **Aggregazione**: media delle features o concatenazione

**Vantaggi**:
- Riconoscimento robusto anche se l'oggetto appare ruotato nel video di retrieval
- Maggiore accuratezza nel matching

### 4. **Similarity Matching - Cosine Similarity**
- **Metrica**: distanza del coseno tra feature vectors normalizzati
- **Formula**: `similarity = (v1 · v2) / (||v1|| × ||v2||)`
- **Range**: [-1, 1], dove 1 = identici, 0 = ortogonali, -1 = opposti
- **Threshold**: 0.75-0.85 (configurabile)

**Tecnica di deduplicazione**:
1. Estrai features per tutti i crops di una classe
2. Calcola matrice di similarità NxN
3. Identifica coppie con similarità > threshold
4. Rimuovi duplicati mantenendo il crop con confidence maggiore

### 5. **Spatial-Temporal Consistency Check**
Nel modulo `VerifyFinds.py`, le detection vengono verificate analizzando:
- **Coerenza spaziale**: distanza normalizzata tra centri dei bbox in frame consecutivi
- **Threshold distanza**: 0.1-0.2 (coordinate normalizzate)
- **Logica**: oggetti reali non "saltano" drasticamente tra frame consecutivi

**Calcolo distanza**:
```python
center1 = (x1_norm + w1_norm/2, y1_norm + h1_norm/2)
center2 = (x2_norm + w2_norm/2, y2_norm + h2_norm/2)
distance = sqrt((center1[0]-center2[0])² + (center1[1]-center2[1])²)
```

### 6. **Database SQLite + File System Ibrido**
- **Metadata**: SQLite (`detections.db`)
  - Tabella `classes`: anagrafica classi
  - Tabella `objects`: metadata oggetti (confidence, bbox, date/time, path features)
- **Features**: file `.npy` numpy in `features_db/{classe}/`
- **Crops**: immagini in `crops_db/{classe}/`

**Perché approccio ibrido?**
- SQLite: veloce per query su metadata
- File system: efficiente per dati binari grandi (features, immagini)
- Scalabilità: facile migrazione a vector DB (Milvus, Pinecone) se necessario

### 7. **Motion Blur Reduction (opzionale)**
In `DinoFeatureExtraction.py` è presente una funzione di deconvoluzione Wiener:
- **PSF (Point Spread Function)**: simula motion blur lineare
- **Parametri**: lunghezza e angolo del blur
- **Scopo**: migliorare qualità features per immagini sfocate

---

## 🛠️ Installazione

### Prerequisiti
- Python 3.8+ (testato su 3.10-3.13)
- CUDA 11.8+ (opzionale, per GPU acceleration)
- 8GB+ RAM (16GB consigliato)
- 2GB+ spazio disco per modelli

### Step 1: Clona/Prepara il Progetto
```bash
cd /home/lapemaya/PycharmProjects/Progetto\ Drone
```

### Step 2: Crea Ambiente Virtuale
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows
```

### Step 3: Installa Dipendenze

**Metodo 1 - Requirements automatico** (consigliato):
```bash
pip install --upgrade pip
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy scipy pillow sympy matplotlib
```

**Librerie principali richieste**:
- `ultralytics` - Framework YOLO
- `torch` + `torchvision` - PyTorch per DINOv2
- `opencv-python` - Elaborazione video/immagini
- `numpy` - Array e calcoli numerici
- `scipy` - Operazioni scientifiche
- `pillow` - Manipolazione immagini
- `sympy` - Calcoli simbolici (usato per rotazioni)
- `matplotlib` - Visualizzazione (debug)

### Step 4: Download Modelli Pre-trained

**DINOv2** (download automatico al primo uso):
```bash
python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
```

**YOLO Models** - scarica i modelli base:
```bash
# Nella cartella yolomodels/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt
```

**Modelli Custom** - assicurati di avere i tuoi modelli trained in:
- `runs/train/yolo_Generale/weights/best.pt` (modello generale)
- `runs/train/yolo_{Classe}/weights/best.pt` (modelli specifici)

### Step 5: Verifica Installazione
```bash
python3 -c "from ultralytics import YOLO; import torch; print('✓ YOLO OK'); print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA: {torch.cuda.is_available()}')"
```

Output atteso:
```
✓ YOLO OK
✓ PyTorch 2.x.x
✓ CUDA: True  # (o False se CPU-only)
```

---

## 📦 Pipeline di Creazione Dataset

### Descrizione Completa

La pipeline automatizza 4 fasi sequenziali per trasformare un video grezzo in un database pulito e indicizzato.

### File: `pipelineCreazioneDataset.py`

```python
def pipelineCreazioneDataset(
    video_path,                      # Path del video input
    outputscan_dir="runs/segment",   # Directory segmentazione
    db_path="detections.db",         # Database SQLite
    feature_dir="features_db",       # Directory features
    crops_dir="crops_db",            # Directory crops
    similarity_threshold=0.85,       # Soglia deduplicazione
    delete_after_import=True         # Pulisci file temporanei
)
```

### FASE 1: Segmentazione Video (`segment_video`)

**File**: `ProvaScanDaVideo.py`

**Funzionalità**:
- Carica modello YOLO (generale o specifico)
- Processa video frame-by-frame
- Per ogni detection:
  - Salva crop dell'oggetto
  - Salva frame completo con bbox
  - Salva file `.txt` con metadata bbox

**Parametri**:
- `input_path`: percorso video
- `output_dir`: directory output (default: `runs/segment`)
- `confidence`: soglia confidence YOLO (default: 0.46)
- `model_path`: path modello YOLO custom

**Output organizzato per classe**:
```
runs/segment/
├── Keys/
│   ├── crops/          # Crop degli oggetti
│   │   ├── crop_0.jpg
│   │   ├── crop_1.jpg
│   ├── bboxes/         # Metadata bbox
│   │   ├── bbox_0.txt
│   │   ├── bbox_1.txt
│   └── frames/         # Frame completi
│       ├── frame_0.jpg
│       ├── frame_1.jpg
├── Wallet/
└── Glasses/
```

**Formato bbox file** (`bbox_X.txt`):
```
Frame: 42
Object ID: 0
Class: Keys
Confidence: 0.8934
x1: 150
y1: 200
x2: 250
y2: 300
width: 100
height: 100
x1_norm: 0.234
y1_norm: 0.185
width_norm: 0.156
height_norm: 0.139
```

### FASE 2: Verifica Detections (`verify_detections`)

**File**: `VerifyFinds.py`

**Funzionalità**:
1. Per ogni classe, carica modello YOLO specifico
2. Ri-verifica ogni crop con il modello della sua classe
3. Applica filtri:
   - **Confidence threshold**: elimina detection < 0.75
   - **Spatial consistency**: verifica coerenza posizione tra frame
   - **Distance threshold**: elimina "salti" > 0.2 coordinate normalizzate

**Parametri**:
- `segment_dir`: directory con risultati segmentazione
- `output_dir`: directory output verifica
- `conf_threshold`: soglia confidence (default: 0.75)
- `delete_negatives`: elimina file negativi (default: True)
- `distance_threshold`: soglia distanza spaziale (default: 0.2)

**Logica verifica spaziale**:
```
Per ogni coppia di detection consecutive:
  1. Calcola centro bbox normalizzato
  2. Calcola distanza euclidea tra centri
  3. Se distanza > threshold: marca come sospetto
  4. Se confidence bassa E distanza alta: elimina
```

**Output**:
```
runs/verification/
├── positive/           # Detection verificate
├── negative/           # Falsi positivi rimossi
└── logs/              # Log verifiche
```

### FASE 3: Deduplicazione (`verify_all_classes`)

**File**: `DinoFeatureExtraction.py`

**Funzionalità**:
1. Per ogni classe in `runs/segment/`:
   - Estrai features DINOv2 per tutti i crops
   - Costruisci matrice similarità NxN
   - Identifica duplicati (similarity > threshold)
   - Elimina duplicati mantenendo migliore

**Algoritmo deduplicazione**:
```
1. crops = [crop_0, crop_1, ..., crop_N]
2. features = [extract_dino(crop_i) for crop_i in crops]
3. similarity_matrix = cosine_similarity(features, features)
4. duplicates = []
5. For i in range(N):
6.     For j in range(i+1, N):
7.         If similarity_matrix[i][j] > threshold:
8.             duplicates.append((i, j))
9. For (i, j) in duplicates:
10.    keep = i if confidence[i] > confidence[j] else j
11.    delete = j if keep == i else i
12.    remove(crops[delete])
```

**Parametri**:
- `segment_dir`: directory segmentazione
- `similarity_threshold`: soglia similarità (default: 0.75)
- `delete_duplicates`: elimina duplicati (default: True)
- `output_dir`: directory output

**Output**:
```
runs/verification/similarity/
├── Keys/
│   ├── similarity_matrix.npy      # Matrice NxN
│   ├── duplicates_report.json     # Report duplicati
│   └── kept_crops.txt             # Lista crops mantenuti
└── ...
```

### FASE 4: Import in Database (`process_segment_results`)

**File**: `SaveInDatabaseFounds.py`

**Funzionalità**:
1. Scansiona `runs/segment/` per ogni classe
2. Per ogni crop:
   - Estrai features con N rotazioni
   - Verifica non sia duplicato rispetto al DB
   - Salva crop in `crops_db/{classe}/`
   - Salva features in `features_db/{classe}/`
   - Inserisci record in SQLite
3. Opzionalmente elimina file temporanei

**Estrazione features con rotazioni**:
```python
def extract_dino_features_rotations(image_path, num_rotations=16):
    img = Image.open(image_path)
    features_list = []
    angles = [360 * i / num_rotations for i in range(num_rotations)]
    
    for angle in angles:
        rotated_img = img.rotate(angle)
        features = extract_dino_features(rotated_img)
        features_list.append(features)
    
    # Aggrega (media o concatenazione)
    aggregated_features = np.mean(features_list, axis=0)
    return aggregated_features
```

**Controllo duplicati DB**:
```python
# Query tutti gli oggetti della stessa classe
existing_objects = db.get_objects_by_class(class_name)

for obj in existing_objects:
    existing_features = np.load(obj.feature_path)
    similarity = cosine_similarity(new_features, existing_features)
    
    if similarity > similarity_threshold:
        print(f"⚠️ Duplicato trovato! Similarity: {similarity:.3f}")
        skip_import = True
        break
```

**Parametri**:
- `segment_dir`: directory segmentazione
- `db_path`: path database SQLite
- `feature_dir`: directory features
- `crops_dir`: directory crops
- `similarity_threshold`: soglia duplicati (default: 0.85)
- `delete_after_import`: pulisci temp (default: True)
- `rotations`: numero rotazioni (default: 16)

**Schema Database**:

**Tabella `classes`**:
```sql
CREATE TABLE classes (
    class_id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Tabella `objects`**:
```sql
CREATE TABLE objects (
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
```

---

## 🔍 Sistema di Retrieval

### File: `ProvaRetrival.py`

### Descrizione

Cerca un oggetto specifico (già nel database) all'interno di un nuovo video. Workflow:
1. Recupera oggetto target dal DB (tramite `object_id`)
2. Carica le sue features DINOv2
3. Segmenta il video di ricerca con YOLO
4. Verifica detection
5. Per ogni crop: estrai features e confronta con target
6. Se similarity > threshold: MATCH trovato!

### Funzione Principale

```python
def retrive_obj_id(
    object_id,                       # ID oggetto da cercare (DB)
    video_path,                      # Video in cui cercare
    outputscan_dir="runs/retrival",  # Directory temp
    db_path="detections.db",         # Database
    feature_dir="features_db",       # Directory features
    crops_dir="crops_db",            # Directory crops
    rotations=8                      # Rotazioni per matching
)
```

### Workflow Dettagliato

#### 1. Caricamento Target dal DB
```python
db = ObjectDatabase(db_path, feature_dir, crops_dir)
db.cursor.execute('''
    SELECT o.object_id, o.feature_path, o.crop_image_path, c.class_name,
           o.confidence, o.detection_date, o.detection_time
    FROM objects o
    JOIN classes c ON o.class_id = c.class_id
    WHERE o.object_id = ?
''', (object_id,))

result = db.cursor.fetchone()
target_features = np.load(result.feature_path)
class_name = result.class_name
```

#### 2. Selezione Modello YOLO Specifico
```python
class_model_path = Path(f"runs/train/yolo_{class_name}/weights/best.pt")
if class_model_path.exists():
    model = YOLO(str(class_model_path))
else:
    raise FileNotFoundError(f"Modello per {class_name} non trovato!")
```

**Vantaggio**: usa modello specializzato per la classe dell'oggetto target → maggiore recall

#### 3. Segmentazione Video di Ricerca
```python
segment_video(
    input_path=video_path,
    output_dir=outputscan_dir,
    model_path=class_model_path,
    confidence=0.50  # Soglia più permissiva per non perdere match
)
```

#### 4. Verifica Detection
```python
verify_detections(
    segment_dir=outputscan_dir,
    conf_threshold=0.8,        # Più restrittivo per ridurre FP
    distance_threshold=0.1     # Coerenza spaziale stretta
)
```

#### 5. Matching Feature-by-Feature
```python
def process_images_in_folder(folder_path, feature_target, feature_dist=0.9):
    """
    Scansiona tutti i crops nella folder e cerca match
    
    Args:
        folder_path: directory con crops da verificare
        feature_target: features dell'oggetto target
        feature_dist: soglia similarità (default: 0.9)
    
    Returns:
        (found: bool, image_path: str)
    """
    for crop_path in Path(folder_path).rglob("*.jpg"):
        # Estrai features del crop
        crop_features = extract_dino_features(crop_path)
        
        # Calcola similarità
        similarity = cosine_similarity(
            crop_features.reshape(1, -1),
            feature_target.reshape(1, -1)
        )[0][0]
        
        if similarity >= feature_dist:
            return True, str(crop_path)
    
    return False, None
```

#### 6. Visualizzazione Risultato
```python
if found:
    print(f"✓ Object found in video! at {img_path}")
    img = cv2.imread(str(img_path))
    cv2.imshow("Object found", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("✗ Object not found in video. SAD!")
```

#### 7. Pulizia
```python
# Elimina file temporanei
for item in Path(outputscan_dir).iterdir():
    if item.is_dir():
        shutil.rmtree(item)
```

### Parametri di Tuning

| Parametro | Valore Default | Descrizione |
|-----------|----------------|-------------|
| `confidence` (segmentazione) | 0.50 | Soglia iniziale detection (più bassa per recall) |
| `conf_threshold` (verifica) | 0.8 | Soglia verifica (più alta per precision) |
| `distance_threshold` | 0.1 | Coerenza spaziale (bassa = movimento limitato) |
| `feature_dist` | 0.9 | Soglia matching features (alta per evitare FP) |
| `rotations` | 8 | Rotazioni per matching (meno della creazione dataset) |

### Trade-off Precision/Recall

**Aumentare Recall** (trovare più match, rischio falsi positivi):
- Abbassare `confidence` → 0.3-0.4
- Abbassare `conf_threshold` → 0.6-0.7
- Abbassare `feature_dist` → 0.85-0.88

**Aumentare Precision** (evitare falsi positivi, rischio perdere match):
- Aumentare `confidence` → 0.6-0.7
- Aumentare `conf_threshold` → 0.85-0.9
- Aumentare `feature_dist` → 0.92-0.95

---

## 🚀 Guida all'Esecuzione

### Scenario 1: Creare Dataset da Video

**Obiettivo**: estrarre e indicizzare tutti gli oggetti da un video.

#### Passo 1: Prepara il video
```bash
# Copia il video nella cartella video/
cp /path/to/your/video.mp4 video/my_video.mp4
```

#### Passo 2: Esegui la pipeline
```bash
python3 pipelineCreazioneDataset.py
```

**Oppure** modifica il video nel codice:
```python
# Alla fine di pipelineCreazioneDataset.py
if __name__ == "__main__":
    pipelineCreazioneDataset(video_path="video/my_video.mp4")
```

#### Passo 3: Monitora l'esecuzione
Output atteso:
```
INIZIO IL MIO LAVORO
🎥 Segmentazione video...
  ✓ Frame processati: 1523
  ✓ Oggetti rilevati: 87
  ✓ Classi trovate: Keys(23), Wallet(15), Glasses(49)

🔍 Verifica detections...
  ✓ Keys: 23 → 21 (2 falsi positivi rimossi)
  ✓ Wallet: 15 → 14 (1 falso positivo rimosso)
  ✓ Glasses: 49 → 47 (2 falsi positivi rimossi)

🧬 Deduplicazione...
  ✓ Keys: 21 → 5 (16 duplicati rimossi)
  ✓ Wallet: 14 → 3 (11 duplicati rimossi)
  ✓ Glasses: 47 → 8 (39 duplicati rimossi)

💾 Import in database...
  ✓ 16 oggetti importati
  ✓ 0 duplicati skippati (già in DB)

IL MIO LAVORO QUI è FINITO
```

#### Passo 4: Verifica risultati
```bash
# Controlla database
sqlite3 detections.db "SELECT class_name, COUNT(*) FROM objects o JOIN classes c ON o.class_id=c.class_id GROUP BY class_name;"

# Output esempio:
# Keys|5
# Wallet|3
# Glasses|8
```

```bash
# Controlla file salvati
ls -lh crops_db/Keys/
ls -lh features_db/Keys/
```

### Scenario 2: Cercare un Oggetto Specifico

**Obiettivo**: trovare un oggetto (già nel DB) in un nuovo video.

#### Passo 1: Identifica object_id
```bash
# Elenca oggetti nel DB
sqlite3 detections.db "SELECT o.object_id, c.class_name, o.confidence, o.crop_image_path FROM objects o JOIN classes c ON o.class_id=c.class_id;"

# Output esempio:
# 1|Keys|0.8934|crops_db/Keys/crop_0_20251027_131244.jpg
# 2|Wallet|0.9123|crops_db/Wallet/crop_1_20251027_131250.jpg
# ...
```

Oppure visualizza il crop:
```bash
# Mostra immagine oggetto ID 1
python3 -c "import cv2; img=cv2.imread('crops_db/Keys/crop_0_20251027_131244.jpg'); cv2.imshow('Object', img); cv2.waitKey(0)"
```

#### Passo 2: Modifica ProvaRetrival.py
```python
# Alla fine del file ProvaRetrival.py
if __name__ == "__main__":
    retrive_obj_id(
        object_id=1,                      # ID dell'oggetto da cercare
        video_path="video/search_video.mp4",  # Video in cui cercare
        outputscan_dir="runs/retrival",
        db_path="detections.db",
        feature_dir="features_db",
        crops_dir="crops_db",
        rotations=8
    )
```

#### Passo 3: Esegui retrieval
```bash
python3 ProvaRetrival.py
```

Output atteso:
```
✓ Oggetto trovato:
  - ID: 1
  - Classe: Keys
  - Confidence originale: 0.8934
  - Rilevato il: 2025-10-27 alle 13:12:44
  - Immagine: crops_db/Keys/crop_0_20251027_131244.jpg

🤖 Caricamento modello YOLO...
✓ Usando modello specifico per Keys: runs/train/yolo_Keys/weights/best.pt

🎥 Segmentazione video di ricerca...
  ✓ 45 detections trovate

🔍 Verifica detections...
  ✓ 45 → 42 detections confermate

🧬 Matching features...
  ⏳ Crop 1/42: similarity 0.6532
  ⏳ Crop 2/42: similarity 0.7821
  ⏳ Crop 3/42: similarity 0.9234 ✓ MATCH!

Object found in video! at runs/retrival/Keys/crops/crop_2.jpg
[Mostra finestra con immagine del match]

🧹 Pulizia directory runs/retrival...
✓ Pulizia completata!
```

### Scenario 3: Esecuzione Programmmatica

**Creazione dataset**:
```python
from pipelineCreazioneDataset import pipelineCreazioneDataset

# Processa multipli video
videos = ["video/video1.mp4", "video/video2.mp4", "video/video3.mp4"]
for video in videos:
    print(f"\n📹 Processando {video}...")
    pipelineCreazioneDataset(
        video_path=video,
        similarity_threshold=0.88,  # Soglia personalizzata
        delete_after_import=True
    )
```

**Retrieval batch**:
```python
from ProvaRetrival import retrive_obj_id

# Cerca multipli oggetti
object_ids = [1, 2, 3, 5, 8]
search_video = "video/security_footage.mp4"

for obj_id in object_ids:
    print(f"\n🔍 Cercando oggetto ID {obj_id}...")
    retrive_obj_id(
        object_id=obj_id,
        video_path=search_video,
        feature_dist=0.92  # Alta precision
    )
```

### Scenario 4: Training Modelli Custom

Se vuoi trainare modelli YOLO su dataset custom:

```bash
# Esempio: train su dataset chiavi
cd SriptTrain/
python3 trainyolokeys.py
```

Script tipo `trainyolokeys.py`:
```python
from ultralytics import YOLO

# Carica modello base
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='../datasetchiavi/data.yaml',  # Path al dataset
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolo_Keys',
    project='../runs/train'
)

# Valida
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

---

## 📂 Struttura Output

```
Progetto Drone/
├── detections.db                    # 💾 Database SQLite principale
│
├── crops_db/                        # 🖼️ Immagini crops salvate
│   ├── Keys/
│   │   ├── crop_0_20251027_131244.jpg
│   │   ├── crop_1_20251027_131255.jpg
│   │   └── ...
│   ├── Wallet/
│   ├── Glasses/
│   └── ...
│
├── features_db/                     # 🧬 Feature vectors DINOv2
│   ├── Keys/
│   │   ├── features_0_20251027_131244.npy
│   │   ├── features_1_20251027_131255.npy
│   │   └── ...
│   ├── Wallet/
│   └── ...
│
├── runs/
│   ├── segment/                     # 📊 Output segmentazione temporanea
│   │   ├── Keys/
│   │   │   ├── crops/
│   │   │   ├── bboxes/
│   │   │   └── frames/
│   │   └── ...
│   │
│   ├── verification/                # ✅ Output verifica
│   │   ├── positive/
│   │   ├── negative/
│   │   └── similarity/
│   │       ├── Keys/
│   │       │   ├── similarity_matrix.npy
│   │       │   └── duplicates_report.json
│   │       └── ...
│   │
│   ├── retrival/                    # 🔍 Output retrieval temporaneo
│   │   └── [pulito dopo ogni run]
│   │
│   └── train/                       # 🏋️ Modelli YOLO trained
│       ├── yolo_Generale/
│       │   └── weights/best.pt
│       ├── yolo_Keys/
│       │   └── weights/best.pt
│       └── ...
│
├── video/                           # 🎬 Video input
│   ├── 4obj1.mp4
│   ├── walletKeyGlasses.mp4
│   └── ...
│
└── pipeline_output.log              # 📝 Log esecuzione (opzionale)
```

---

## ⚙️ Parametri e Configurazione

### Pipeline Creazione Dataset

| Parametro | Default | Range | Descrizione |
|-----------|---------|-------|-------------|
| `confidence` (segment) | 0.46 | 0.1-0.9 | Soglia detection YOLO iniziale |
| `conf_threshold` (verify) | 0.75 | 0.5-0.95 | Soglia verifica detections |
| `distance_threshold` | 0.2 | 0.05-0.5 | Soglia coerenza spaziale |
| `similarity_threshold` (dedup) | 0.75 | 0.7-0.9 | Soglia deduplicazione features |
| `similarity_threshold` (import) | 0.85 | 0.8-0.95 | Soglia anti-duplicati DB |
| `rotations` | 16 | 4-32 | Numero rotazioni feature extraction |
| `delete_after_import` | True | bool | Elimina file temporanei dopo import |
| `delete_negatives` | True | bool | Elimina falsi positivi dopo verifica |
| `delete_duplicates` | True | bool | Elimina duplicati dopo clustering |

### Sistema Retrieval

| Parametro | Default | Range | Descrizione |
|-----------|---------|-------|-------------|
| `confidence` (segment) | 0.50 | 0.3-0.7 | Soglia detection (più bassa per recall) |
| `conf_threshold` (verify) | 0.8 | 0.7-0.9 | Soglia verifica (più alta per precision) |
| `distance_threshold` | 0.1 | 0.05-0.3 | Coerenza spaziale stretta |
| `feature_dist` | 0.9 | 0.85-0.95 | Soglia matching features |
| `rotations` | 8 | 4-16 | Rotazioni per matching (meno della creazione) |

### Guida al Tuning

**Dataset con molti oggetti simili** (es. chiavi diverse):
```python
pipelineCreazioneDataset(
    video_path="video/many_keys.mp4",
    similarity_threshold=0.75,  # Più bassa per distinguere varianti
    rotations=24               # Più rotazioni per robustezza
)
```

**Dataset con oggetti molto distinti**:
```python
pipelineCreazioneDataset(
    video_path="video/diverse_objects.mp4",
    similarity_threshold=0.90,  # Più alta, meno rischio confusione
    conf_threshold=0.70,        # Più permissiva
    rotations=8                 # Meno rotazioni necessarie
)
```

**Retrieval in video con movimento veloce**:
```python
retrive_obj_id(
    object_id=5,
    video_path="video/fast_motion.mp4",
    confidence=0.35,            # Molto bassa per catturare blur
    distance_threshold=0.3,     # Più permissiva per movimento
    feature_dist=0.88           # Leggermente più bassa
)
```

**Retrieval ad alta precision**:
```python
retrive_obj_id(
    object_id=5,
    video_path="video/search.mp4",
    confidence=0.65,            # Alta per meno FP
    conf_threshold=0.85,        # Molto restrittiva
    feature_dist=0.93,          # Match quasi perfetto
    rotations=16                # Più rotazioni per robustezza
)
```

---

## 🐛 Troubleshooting

### Problema 1: "No module named 'ultralytics'"
**Soluzione**:
```bash
pip install ultralytics
```

### Problema 2: "Model file not found: runs/train/yolo_Generale/weights/best.pt"
**Causa**: modello custom non trainato.
**Soluzione**:
1. Usa modello base: modifica `model_path="yolov8n.pt"` in `ProvaScanDaVideo.py`
2. Oppure train il modello:
```bash
cd SriptTrain/
python3 trainGenerale.py
```

### Problema 3: "CUDA out of memory"
**Soluzione**:
```python
# In DinoFeatureExtraction.py, cambia device
device = "cpu"  # Invece di "cuda"
```

Oppure riduci batch processing (modifica nei file per processare meno immagini contemporaneamente).

### Problema 4: Pipeline molto lenta
**Ottimizzazioni**:
1. **Usa GPU**: verifica `torch.cuda.is_available() == True`
2. **Riduci rotazioni**: `rotations=8` invece di 16
3. **Aumenta confidence iniziale**: meno detection da processare
4. **Usa modello YOLO più leggero**: `yolov8n.pt` invece di `yolov8l.pt`

### Problema 5: Troppi duplicati nel database
**Causa**: `similarity_threshold` troppo bassa.
**Soluzione**:
```python
pipelineCreazioneDataset(
    video_path="video/video.mp4",
    similarity_threshold=0.90  # Aumenta soglia
)
```

### Problema 6: Retrieval non trova match che dovrebbe trovare
**Debug**:
1. Visualizza features target:
```python
import numpy as np
features = np.load("features_db/Keys/features_0_20251027_131244.npy")
print(f"Shape: {features.shape}, Norm: {np.linalg.norm(features)}")
```

2. Test manuale similarità:
```python
from ProvaRetrival import extract_dino_features
from scipy.spatial.distance import cosine

target_feat = np.load("features_db/Keys/features_0_20251027_131244.npy")
test_crop = "runs/retrival/Keys/crops/crop_5.jpg"
test_feat = extract_dino_features(test_crop)

similarity = 1 - cosine(target_feat, test_feat)
print(f"Similarity: {similarity:.4f}")
```

3. Abbassa soglia:
```python
retrive_obj_id(object_id=1, video_path="video/test.mp4", feature_dist=0.85)
```

### Problema 7: "Database is locked"
**Causa**: processo precedente non ha chiuso la connessione.
**Soluzione**:
```bash
# Chiudi tutti i processi Python
pkill -9 python3

# Oppure rimuovi il lock
rm detections.db-journal
```

### Problema 8: Falsi positivi in retrieval
**Soluzione**:
1. Aumenta `feature_dist` → 0.92-0.95
2. Aumenta `conf_threshold` → 0.85-0.9
3. Usa modello YOLO più specifico per la classe
4. Aumenta `rotations` durante la creazione dataset → 24-32

---

## 📊 Metriche e Valutazione

### Valutare Performance Detection
```python
from ultralytics import YOLO

model = YOLO("runs/train/yolo_Keys/weights/best.pt")
metrics = model.val(data="datasetchiavi/data.yaml")

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")
```

### Valutare Deduplicazione
```bash
# Prima della deduplicazione
echo "Crops prima: $(find runs/segment/Keys/crops -name '*.jpg' | wc -l)"

# Dopo deduplicazione
echo "Crops dopo: $(find runs/verification/similarity/Keys -name '*.jpg' | wc -l)"
```

### Valutare Retrieval
Per valutazione sistematica, crea ground truth:
```python
# Test retrieval su N video annotati
results = []
for video, expected_object_id in ground_truth:
    found, _ = retrive_obj_id(expected_object_id, video)
    results.append(found)

accuracy = sum(results) / len(results)
print(f"Retrieval Accuracy: {accuracy:.2%}")
```

---

## 🚀 Prossimi Sviluppi

1. **Vector Database**: migrazione a Milvus/Pinecone per scalabilità
2. **Real-time processing**: pipeline streaming per webcam/drone live
3. **Multi-object tracking**: tracciamento temporale oggetti nei video
4. **Web interface**: UI per visualizzare DB e lanciare retrieval
5. **Distributed processing**: parallelize feature extraction su cluster
6. **Active learning**: re-train modelli con esempi hard-negative

---

## 📚 Riferimenti

- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **DINOv2**: [Facebook Research Paper](https://arxiv.org/abs/2304.07193)
- **Cosine Similarity**: [scikit-learn docs](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity)
- **SQLite**: [Python sqlite3](https://docs.python.org/3/library/sqlite3.html)

---



## 👥 Contributi

Per contribuire al progetto:
1. Fork del repository
2. Crea feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit modifiche (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri Pull Request

---

## 📧 Contatti

Per domande o supporto, contatta: lapo.chiostrini00@gmail.com

---

**Ultimo aggiornamento**: 28 Ottobre 2025


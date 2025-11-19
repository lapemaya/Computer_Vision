from ProvaScanDaVideo import segment_video
from VerifyFinds import verify_detections
import numpy as np
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dataset import ObjectDatabase
from DinoFeatureExtraction import extract_dino_features
import json
from torchvision import transforms
import torch
from PIL import Image
import shutil


#param iniziali: db_path,feature dir db,crops dir db, obj id,video_path,outputscan,
model_name = "dinov2_vitb14"
model = torch.hub.load("facebookresearch/dinov2", model_name)
model.eval()


transform = transforms.Compose([
    transforms.Resize((518, 518)),  # stessa risoluzione usata da DINOv2
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

def retrive_obj_id(object_id,video_path,outputscan_dir="runs/retrival",db_path="detections.db",feature_dir="features_db",crops_dir="crops_db",rotations=8):
    db = ObjectDatabase(db_path=db_path, feature_dir=feature_dir, crops_dir=crops_dir)

    # Recupera l'oggetto
    db.cursor.execute('''
            SELECT o.object_id, o.feature_path, o.crop_image_path, c.class_name, 
                   o.confidence, o.detection_date, o.detection_time
            FROM objects o
            JOIN classes c ON o.class_id = c.class_id
            WHERE o.object_id = ?
        ''', (object_id,))

    result = db.cursor.fetchone()
    if not result:
        print(f"❌ Oggetto con ID {object_id} non trovato nel database!")
        return None

    obj_id, feature_path, crop_path, class_name, confidence, det_date, det_time = result
    target_features = np.load(feature_path)
    db.close()

    print(f"✓ Oggetto trovato:")
    print(f"  - ID: {obj_id}")
    print(f"  - Classe: {class_name}")
    print(f"  - Confidence originale: {confidence:.4f}" if confidence else "  - Confidence: N/A")
    print(f"  - Rilevato il: {det_date} alle {det_time}")
    if crop_path:
        print(f"  - Immagine: {crop_path}")

    # 2. Carica il modello YOLO appropriato
    print(f"\n🤖 Caricamento modello YOLO...")

    # Prova a usare il modello specifico per la classe se esiste
    print(class_name)
    class_model_path = Path(f"runs/train/yolo_{class_name}/weights/best.pt")
    if class_model_path.exists():
        print(f"✓ Usando modello specifico per {class_name}: {class_model_path}")
        model = YOLO(str(class_model_path))
    else:
        print("modello non trovato")
        return None
    "___________________________________________________________\n"
    segment_video(input_path=video_path, output_dir=outputscan_dir, model_path=class_model_path,confidence=0.50)
    verify_detections(segment_dir=outputscan_dir,conf_threshold=0.8,distance_threshold=0.1)
    found,img_path=process_images_in_folder(folder_path=outputscan_dir,feature_target=target_features)
    if found:
        print("Object found in video! at ",img_path)
        imgFound=cv2.imread(str(img_path))
        cv2.imshow("Object found",imgFound)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Object not found in video. SAD!")

    # Pulisci la directory outputscan_dir eliminando tutte le sottocartelle
    print(f"\n🧹 Pulizia directory {outputscan_dir}...")
    output_path = Path(outputscan_dir)
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Eliminata cartella: {item.name}")
        print("✓ Pulizia completata!")

    return None





def process_images_in_folder(folder_path,feature_target,feature_dist=0.85):
    print("entrato")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    folder = Path(folder_path)

    # Cerca in tutte le sottocartelle (class_name/crops/)
    for class_folder in folder.iterdir():
        if class_folder.is_dir():
            crops_folder = class_folder / "crops"
            if crops_folder.exists():
                print(f"📁 Cercando in: {crops_folder}")
                for img_path in crops_folder.glob('*'):
                    if img_path.suffix.lower() in image_extensions:
                        image = cv2.imread(str(img_path))
                        if image is not None:
                            print(f"Processing: {img_path.name}")
                            sim=check2feature(img=image,feature_target=feature_target)
                            print(sim)
                            if sim>feature_dist:
                                print(f"✓ FOUND! Oggetto trovato in {img_path}")
                                return True,img_path
    print("❌ Oggetto non trovato in nessuna immagine")
    return False,None



def check2feature(img, feature_target):
    # Convert OpenCV BGR image to PIL RGB image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Rileva il device del modello
    device = next(model.parameters()).device
    img_t = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model(img_t)

    # feats: tensor [1, feature_dim]
    f1 = feats.squeeze(0).cpu().numpy()

    # Se feature_target ha shape (n, 768), confronta con tutte le rotazioni
    # e restituisci la similarità massima
    if len(feature_target.shape) == 2:
        # feature_target ha shape (num_rotations, feature_dim)
        # Calcola la similarità con ogni rotazione e prendi il massimo
        similarities = []
        for rotation_features in feature_target:
            cosine_sim = np.dot(f1, rotation_features) / (np.linalg.norm(f1) * np.linalg.norm(rotation_features))
            similarities.append(cosine_sim)
        max_similarity = max(similarities)
        print(f"Similarità coseno (max tra {len(feature_target)} rotazioni): {max_similarity:.4f}")
        return max_similarity
    else:
        # feature_target ha shape (feature_dim,) - singolo vettore
        cosine_sim = np.dot(f1, feature_target) / (np.linalg.norm(f1) * np.linalg.norm(feature_target))
        print(f"Similarità coseno: {cosine_sim:.4f}")
        return cosine_sim



if __name__ == "__main__":
    retrive_obj_id(object_id=359,video_path="video/4obj1.mp4")

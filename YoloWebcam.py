from torchvision import transforms
import requests
from io import BytesIO
import numpy as np
from ultralytics import YOLO
import clip
import torch
from PIL import Image
import cv2
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # più veloce
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),  # più veloce
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])


def extract_dino_features_from_bbox(frame, bbox, device="cpu"):
    """
    frame: immagine OpenCV BGR
    bbox: (x1, y1, x2, y2) in pixel (float o int)
    """
    # Converti i valori in interi
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Crop bbox dal frame
    crop = frame[y1:y2, x1:x2]  # y prima, x dopo

    # Converti BGR -> RGB e da NumPy a PIL
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)

    # Preprocess DINO
    img_t = transform(pil_crop).unsqueeze(0).to(device)

    # Estrai feature
    with torch.no_grad():
        feats = model(img_t)

    return feats.squeeze(0).cpu().numpy()
def feature_detection_CLIP(image,bbox):
    global features_tensor  # <-- aggiungi questa riga

    # Carica e preprocessa l'immagine
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    crop = image.crop(bbox)
    image = preprocess(crop).unsqueeze(0).to("cuda")

    # Estrai feature
    with torch.no_grad():
        image_features = modelCLIP.encode_image(image)
    #print(image_features.shape)
    feat = F.normalize(image_features, dim=1)  # se singolo, oppure dim=1 se batch
    feat = feat.squeeze(0).unsqueeze(0)        # shape [1,1024]
    threshold = 1 # sopra questa soglia consideriamo "troppo simile"

    if features_tensor.shape[0] == 0:
        # prima feature, inserisci direttamente
        features_tensor = feat.unsqueeze(0)  # shape [1, 1024]
        print("first")
        print(feat.unsqueeze(0))
    else:
        # calcola cosine similarity con tutte le features già presenti
        sims = F.cosine_similarity(feat.unsqueeze(0), features_tensor, dim=1)  # shape [N]
        
        if torch.max(sims) < threshold:
            # nessuna troppo simile, aggiungi
            features_tensor = torch.cat([features_tensor, feat.unsqueeze(0)], dim=0)
            print("Feature aggiunta, totale:", features_tensor.shape[0])
        else:
            print("Feature troppo simile, non aggiunta.")

    return


modelCLIP, preprocess = clip.load("RN50", device="cuda")
# ======== 1. Carica il modello YOLO ========
# Sostituisci con il percorso del tuo modello (.pt)
MODEL_PATH = "runs/train/yolo_keys/weights/best.pt"
model = YOLO(MODEL_PATH)

# Se hai una GPU, usa 'cuda', altrimenti 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
features_tensor = torch.empty((0, 1024)).to("cuda")  # Inizializza il tensore delle feature CLIP
# ======== 2. Apri la webcam ========
cap = cv2.VideoCapture(0)  # 0 = webcam predefinita
if not cap.isOpened():
    raise RuntimeError("Errore: impossibile aprire la webcam")

print("Premi 'q' per uscire.")
# ======== 3. Loop di acquisizione ========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ======== 4. Fai la detection YOLO ========
    results = model(frame, stream=True, device=device,conf=0.5,verbose=False)
    #print how many results
    # ======== 5. Visualizza i risultati ========
    for r in results:
        #print(len(r.boxes))
        boxes = r.boxes.xyxy.cpu().numpy()     # [x1, y1, x2, y2]
        confs = r.boxes.conf.cpu().numpy()     # confidence
        clss = r.boxes.cls.cpu().numpy().astype(int)  # class IDs
        names = r.names                        # class names dict
        if len(boxes) > 0 and confs[0] > 0.7:
            feature_detection_CLIP(frame,boxes[0])
            #bbox = (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
            #extract_dino_features_from_bbox(frame,bbox,"cuda")

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box.astype(int)
            label = f"{names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostra l’immagine
    cv2.imshow("YOLO Realtime Detection", frame)

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======== 6. Cleanup ========
cap.release()
cv2.destroyAllWindows()

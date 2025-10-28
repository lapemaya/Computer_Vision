import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import cv2
import os

# ======== 1. Carica il modello DINOv2 ========
# Modelli disponibili: "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
model_name = "dinov2_vitb14"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

model = torch.hub.load("facebookresearch/dinov2", model_name)
model.eval()
model = model.to(device)


# ======== 2. Preprocessing ========
transform = transforms.Compose([
    transforms.Resize((518, 518)),  # stessa risoluzione usata da DINOv2
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ======== 2.5 Funzione de-blur con Wiener deconvolution ========
def motion_psf(length, angle_deg):
    """Build a linear motion PSF kernel of given length and angle (degrees).
    Kernel will be centered and normalized.
    """
    import math
    # Kernel size - make odd and proportional to length
    size = int(max(3, np.ceil(length)))
    # ensure odd
    if size % 2 == 0:
        size += 1
    # To allow slanted lines, make a square kernel with side = length if length>3 else 3
    side = max(size, int(length))
    side = max(3, side)
    if side % 2 == 0:
        side += 1
    kernel = np.zeros((side, side), dtype=np.float32)
    # Draw a line through the center with the given angle and length
    center = side // 2
    angle = np.deg2rad(angle_deg)
    dx = int(np.cos(angle) * (length / 2.0))
    dy = int(np.sin(angle) * (length / 2.0))
    x1, y1 = center - dx, center - dy
    x2, y2 = center + dx, center + dy
    cv2.line(kernel, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=1)
    s = kernel.sum()
    if s != 0:
        kernel /= s
    else:
        # fallback to delta
        kernel[center, center] = 1.0
    return kernel


def wiener_deconv_channel(channel, kernel, K=0.01):
    """Apply Wiener deconvolution to a single-channel image using given kernel.
    channel: 2D numpy float32 (0..1)
    kernel: 2D numpy float32
    K: float Wiener regularization
    Returns deconvolved channel clipped to 0..1
    """
    # pad kernel to image size
    img_shape = channel.shape
    # Move kernel to top-left by shifting center
    pad = np.zeros(img_shape, dtype=np.float32)
    kh, kw = kernel.shape
    pad[:kh, :kw] = kernel
    # shift kernel to center (so that its center is at (0,0) in freq domain)
    pad = np.roll(np.roll(pad, -kh//2, axis=0), -kw//2, axis=1)

    # FFTs
    G = np.fft.fft2(channel)
    H = np.fft.fft2(pad)
    H_conj = np.conj(H)
    denom = (H * H_conj) + K
    # avoid divide-by-zero
    denom = np.where(np.abs(denom) == 0, 1e-8, denom)
    F_est = (H_conj * G) / denom
    f = np.fft.ifft2(F_est)
    f = np.real(f)
    # clip
    f = np.clip(f, 0, 1)
    return f.astype(np.float32)


def wiener_deconv_color(img_bgr, kernel, K=0.01):
    """Apply Wiener deconvolution on color image per-channel.
    Input: BGR uint8 image (0..255). Returns uint8 BGR image.
    """
    img = img_bgr.astype(np.float32) / 255.0
    chans = cv2.split(img)
    out_chans = []
    for c in chans:
        out = wiener_deconv_channel(c, kernel, K=K)
        out_chans.append(out)
    merged = cv2.merge(out_chans)
    merged = (np.clip(merged, 0, 1) * 255.0).astype(np.uint8)
    return merged


def apply_deblur(image, blur_length=15, blur_angle=0, K=0.01):
    """
    Applica un filtro di Wiener deconvolution per rimuovere motion blur

    Args:
        image: PIL Image
        blur_length: lunghezza del motion blur in pixel (default 15)
        blur_angle: angolo del motion blur in gradi (default 0)
        K: costante di regolarizzazione Wiener (default 0.01)

    Returns:
        PIL Image de-blurred
    """
    # Converti in numpy array per OpenCV (RGB)
    img_array = np.array(image)

    # Crea il kernel PSF per il motion blur
    kernel = motion_psf(blur_length, blur_angle)

    # Applica Wiener deconvolution
    deblurred = wiener_deconv_color(img_array, kernel, K=K)

    # Converti di nuovo in PIL Image
    result = Image.fromarray(deblurred)

    # Applica un leggero sharpening finale
    enhancer = ImageEnhance.Sharpness(result)
    sharpened = enhancer.enhance(1.2)

    return sharpened

# ======== 3. Funzione per estrarre feature da un bounding box ========
def extract_dino_features(image_path, bbox=None, show=True, save_path=None):
    """
    image_path: percorso o URL dell'immagine
    bbox: (x, y, w, h) - can be absolute pixels or relative floats in [0,1]
    show: if True, display the cropped image
    save_path: if provided, save the cropped image to this path
    """
    # Carica immagine
    if image_path.startswith("http"):
        image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    # Applica de-blur prima di qualsiasi altra operazione
    #image = apply_deblur(image)
    #image.show()

    if( bbox is not None):
        # Ritaglia la regione
        x, y, w, h = bbox

        # Get image dimensions
        img_w, img_h = image.size

        # If bbox appears relative (all values between 0 and 1), convert to absolute
        if all(0.0 <= float(v) <= 1.0 for v in (x, y, w, h)):
            x = int(x * img_w)
            y = int(y * img_h)
            w = int(w * img_w)
            h = int(h * img_h)
        else:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

        # Clamp to image bounds
        x = max(0, min(x, img_w))
        y = max(0, min(y, img_h))
        w = max(0, min(w, img_w - x))
        h = max(0, min(h, img_h - y))

        print(f"Image size: {img_w}x{img_h}, Crop box: x={x}, y={y}, w={w}, h={h}")

        image = image.crop((x-w/2, y-h/2,x+w/2 ,y+h/2 ))

    # Optionally save the cropped image
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        image.save(save_path)
        print(f"Cropped image saved to: {save_path}")

    # Optionally show the cropped image
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.array(image))
        plt.title(f"Cropped from {image_path}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Preprocess - usa il device globale dove è stato caricato il modello
    img_t = transform(image).unsqueeze(0).to(device)

    # Estrai feature
    with torch.no_grad():
        feats = model(img_t)

    # feats: tensor [1, feature_dim]
    return feats.squeeze(0).cpu().numpy()


def extract_dino_features_rotations(image_path, bbox=None, show=False, save_path=None, num_rotations=16):
    """
    Estrae features DINOv2 con rotation invariance usando multiple rotazioni

    Args:
        image_path: percorso o URL dell'immagine
        bbox: (x, y, w, h) - bounding box opzionale
        show: se True, mostra l'immagine croppata
        save_path: percorso dove salvare l'immagine croppata
        num_rotations: numero di rotazioni da generare (default 8 = ogni 45°)

    Returns:
        list di numpy arrays, uno per ogni rotazione (angoli: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    """
    # Carica immagine
    if image_path.startswith("http"):
        image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    if bbox is not None:
        # Ritaglia la regione
        x, y, w, h = bbox
        img_w, img_h = image.size

        # If bbox appears relative (all values between 0 and 1), convert to absolute
        if all(0.0 <= float(v) <= 1.0 for v in (x, y, w, h)):
            x = int(x * img_w)
            y = int(y * img_h)
            w = int(w * img_w)
            h = int(h * img_h)
        else:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

        # Clamp to image bounds
        x = max(0, min(x, img_w))
        y = max(0, min(y, img_h))
        w = max(0, min(w, img_w - x))
        h = max(0, min(h, img_h - y))

        image = image.crop((x-w/2, y-h/2, x+w/2, y+h/2))

    # Salva l'immagine originale se richiesto
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        image.save(save_path)
        print(f"Cropped image saved to: {save_path}")

    # Mostra l'immagine se richiesto
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.array(image))
        plt.title(f"Cropped from {image_path}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Genera rotazioni ed estrai features per ciascuna
    rotation_features = []
    angles = [i * (360 / num_rotations) for i in range(num_rotations)]  # 0°, 45°, 90°, ..., 315°

    print(f"🔄 Estrazione features con {num_rotations} rotazioni...")

    for angle in angles:
        # Ruota l'immagine
        rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)

        # Preprocess - usa il device globale dove è stato caricato il modello
        img_t = transform(rotated_image).unsqueeze(0).to(device)

        # Estrai feature
        with torch.no_grad():
            feats = model(img_t)

        features_array = feats.squeeze(0).cpu().numpy()
        rotation_features.append(features_array)
        print(f"  ✓ Rotazione {angle:.0f}°: feature shape {features_array.shape}")

    return rotation_features


def check2images(img1,img2,bbox1=None,bbox2=None, show=False):
    f1 = extract_dino_features(img1, bbox=bbox1, show=show)
    f2 = extract_dino_features(img2, bbox=bbox2, show=show)

    # Somiglianza coseno
    cosine_sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
    print("Similarità coseno:", cosine_sim)
    return cosine_sim


def verify_crops_similarity(crops_dir, similarity_threshold=0.85, delete_duplicates=True, output_report=None):
    """
    Verifica la similarità tra tutti i crop in una directory usando DINOv2

    Args:
        crops_dir: directory contenente i crop da verificare
        similarity_threshold: soglia di similarità (0-1) sopra la quale i crop sono considerati duplicati
        delete_duplicates: se True, elimina i crop con confidence più bassa
        output_report: percorso del file JSON dove salvare il report (opzionale)

    Returns:
        report: dizionario con i risultati della verifica
    """
    crops_path = Path(crops_dir)

    if not crops_path.exists():
        print(f"❌ Directory non trovata: {crops_dir}")
        return None

    # Trova tutti i crop
    crop_files = sorted(crops_path.glob("*.jpg"))

    if len(crop_files) == 0:
        print(f"❌ Nessun crop trovato in {crops_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"🔬 Analisi similarità DINOv2")
    print(f"{'='*60}")
    print(f"📁 Directory: {crops_dir}")
    print(f"📊 Crop trovati: {len(crop_files)}")
    print(f"🎯 Soglia similarità: {similarity_threshold}")
    print(f"\n🔬 Estrazione feature DINOv2...")

    # Estrai feature per tutti i crop
    crop_features = []
    for crop_path in crop_files:
        feats = extract_dino_features(str(crop_path), show=False)

        # Prova a trovare il file bbox corrispondente per ottenere la confidence
        bbox_path = crop_path.parent.parent / "bboxes" / (crop_path.stem + '.txt')
        confidence = 0.0

        if bbox_path.exists():
            try:
                with open(bbox_path, 'r') as f:
                    for line in f:
                        if 'Confidence:' in line:
                            confidence = float(line.split(':')[1].strip())
                            break
            except:
                pass

        crop_features.append({
            'path': crop_path,
            'features': feats,
            'confidence': confidence,
            'name': crop_path.name
        })
        print(f"  ✓ {crop_path.name} (conf: {confidence:.2f})")

    # Trova crop troppo simili
    print(f"\n🔍 Confronto {len(crop_features)} crop per similarità...")

    duplicates = []
    too_similar_indices = set()

    for i in range(len(crop_features)):
        if crop_features[i]['features'] is None:
            continue
        for j in range(i + 1, len(crop_features)):
            if crop_features[j]['features'] is None:
                continue

            # Calcola similarità coseno
            similarity = np.dot(crop_features[i]['features'], crop_features[j]['features']) / \
                        (np.linalg.norm(crop_features[i]['features']) * np.linalg.norm(crop_features[j]['features']))

            if similarity > similarity_threshold:
                # Trova quale ha confidence più bassa
                conf_i = crop_features[i]['confidence']
                conf_j = crop_features[j]['confidence']

                if conf_i < conf_j:
                    to_keep = j
                    to_remove = i
                else:
                    to_keep = i
                    to_remove = j

                too_similar_indices.add(to_remove)

                duplicates.append({
                    'image1': crop_features[i]['name'],
                    'image2': crop_features[j]['name'],
                    'similarity': float(similarity),
                    'conf1': float(conf_i),
                    'conf2': float(conf_j),
                    'kept': crop_features[to_keep]['name'],
                    'removed': crop_features[to_remove]['name']
                })

                print(f"🎨 Similarità: {similarity:.3f} - {crop_features[i]['name']} (conf {conf_i:.2f}) ↔️ {crop_features[j]['name']} (conf {conf_j:.2f})")
                print(f"   → Mantengo: {crop_features[to_keep]['name']}, Rimuovo: {crop_features[to_remove]['name']}")

    # Elimina duplicati se richiesto
    deleted_files = []
    if delete_duplicates and len(too_similar_indices) > 0:
        print(f"\n🗑️  Eliminazione duplicati...")
        for idx in too_similar_indices:
            crop_path = crop_features[idx]['path']

            # Trova anche frame e bbox corrispondenti
            frame_path = crop_path.parent.parent / "frames" / (crop_path.stem + '_full.jpg')
            bbox_path = crop_path.parent.parent / "bboxes" / (crop_path.stem + '.txt')

            files_to_delete = [crop_path]
            if frame_path.exists():
                files_to_delete.append(frame_path)
            if bbox_path.exists():
                files_to_delete.append(bbox_path)

            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    print(f"  ✓ Eliminato: {file_path.name}")
                except Exception as e:
                    print(f"  ❌ Errore nell'eliminazione di {file_path.name}: {e}")

    # Crea report
    report = {
        'directory': str(crops_dir),
        'total_crops': len(crop_files),
        'similarity_threshold': similarity_threshold,
        'duplicates_found': len(duplicates),
        'unique_crops': len(crop_files) - len(too_similar_indices),
        'deleted_files': len(deleted_files),
        'duplicates': duplicates,
        'deleted_files_list': deleted_files
    }

    # Salva report se richiesto
    if output_report:
        output_path = Path(output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report salvato in: {output_report}")

    # Stampa riepilogo
    print(f"\n{'='*60}")
    print(f"📊 RIEPILOGO")
    print(f"{'='*60}")
    print(f"Totale crop: {report['total_crops']}")
    print(f"Duplicati trovati: {report['duplicates_found']}")
    print(f"Crop unici: {report['unique_crops']}")
    if delete_duplicates:
        print(f"File eliminati: {report['deleted_files']}")

    return report


def verify_all_classes(segment_dir="runs/segment", similarity_threshold=0.85, delete_duplicates=True, output_dir="runs/verification"):
    """
    Verifica la similarità per tutte le classi in runs/segment
    """
    segment_path = Path(segment_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not segment_path.exists():
        print(f"❌ Directory non trovata: {segment_dir}")
        return

    print(f"\n{'='*60}")
    print(f"🚀 VERIFICA SIMILARITÀ PER TUTTE LE CLASSI")
    print(f"{'='*60}")

    all_reports = {}

    # Itera su ogni classe
    for class_dir in sorted(segment_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        crops_dir = class_dir / "crops"

        if not crops_dir.exists():
            print(f"\n⚠️  Nessuna directory crops per {class_name}")
            continue

        # Verifica similarità per questa classe
        report = verify_crops_similarity(
            crops_dir=str(crops_dir),
            similarity_threshold=similarity_threshold,
            delete_duplicates=delete_duplicates,
            output_report=str(output_path / f"{class_name}_similarity_report.json")
        )

        if report:
            all_reports[class_name] = report

    # Salva report globale
    global_report_path = output_path / "global_similarity_report.json"
    with open(global_report_path, 'w') as f:
        json.dump(all_reports, f, indent=2)

    # Stampa riepilogo globale
    print(f"\n{'='*60}")
    print(f"📊 RIEPILOGO GLOBALE")
    print(f"{'='*60}")

    total_crops = sum(r['total_crops'] for r in all_reports.values())
    total_duplicates = sum(r['duplicates_found'] for r in all_reports.values())
    total_unique = sum(r['unique_crops'] for r in all_reports.values())
    total_deleted = sum(r['deleted_files'] for r in all_reports.values())

    print(f"Classi analizzate: {len(all_reports)}")
    print(f"Totale crop: {total_crops}")
    print(f"Duplicati trovati: {total_duplicates}")
    print(f"Crop unici: {total_unique}")
    if delete_duplicates:
        print(f"File eliminati: {total_deleted}")

    print(f"\n📋 Dettaglio per classe:")
    for class_name, report in all_reports.items():
        print(f"  {class_name}:")
        print(f"    Crop: {report['total_crops']}")
        print(f"    Duplicati: {report['duplicates_found']}")
        print(f"    Unici: {report['unique_crops']}")
        if delete_duplicates:
            print(f"    Eliminati: {report['deleted_files']}")

    print(f"\n💾 Report globale salvato in: {global_report_path}")

    return all_reports


# ======== 4. Esempio d'uso ========
if __name__ == "__main__":
    # Esempio 1: Confronta due immagini specifiche
    #check2images("runs/segment/Remote/crops/frame00105_obj01_Remote.jpg",
    #             "runs/segment/Remote/crops/frame00090_obj00_Remote.jpg")

    # Esempio 2: Verifica una singola classe
    #verify_crops_similarity(
    #    crops_dir="runs/segment/Remote/crops",
    #    similarity_threshold=0.85,
    #    delete_duplicates=True,
    #    output_report="runs/verification/Remote_similarity.json"
    #)

    # Esempio 3: Verifica tutte le classi
    verify_all_classes(
        segment_dir="runs/segment",
        similarity_threshold=0.85,
        delete_duplicates=True,
        output_dir="runs/verification/similarity"
    )

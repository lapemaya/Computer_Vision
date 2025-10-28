from ultralytics import YOLO
import cv2
from pathlib import Path

def segment_video(input_path, output_dir="runs/segment", confidence=0.25,model_path="runs/train/yolo_Generale/weights/best.pt"):
    # Carica il modello YOLOv8 per segmentation (puoi scegliere n, s, m, l, x)
    model = YOLO(model_path)
    # Apri il video
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    object_count = 0
    class_counts = {}  # Contatore per classe
    base_dir = Path(output_dir)

    # Predizione sul video
    results = model.predict(
        source=input_path,   # video input
        save=True,           # salva video con output
        save_frames=False,   # non salvare i singoli frame
        vid_stride=1,        # analizza ogni frame
        conf=confidence,            # soglia di confidenza
        stream=True          # usa streaming per processare frame per frame
    )

    # Processa ogni frame
    for result in results:
        frame_count += 1

        # Ottieni il frame originale
        frame = result.orig_img

        # Itera su ogni detection nel frame
        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                object_count += 1

                # Estrai coordinate del bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]

                # Conta gli oggetti per classe
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1

                # Crea directory per questa classe se non esistono
                crop_dir = base_dir / class_name / "crops"
                bbox_dir = base_dir / class_name / "bboxes"
                frame_dir = base_dir / class_name / "frames"
                crop_dir.mkdir(parents=True, exist_ok=True)
                bbox_dir.mkdir(parents=True, exist_ok=True)
                frame_dir.mkdir(parents=True, exist_ok=True)

                # Converti in interi
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calcola width e height
                w = x2 - x1
                h = y2 - y1

                # Calcola coordinate normalizzate (0-1)
                img_h, img_w = frame.shape[:2]
                x1_norm = x1 / img_w
                y1_norm = y1 / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # Crop l'oggetto rilevato
                crop = frame[y1:y2, x1:x2]

                # Nome file univoco
                filename = f"frame{frame_count:05d}_obj{i:02d}_{class_name}"

                # Salva l'immagine croppata
                crop_path = crop_dir / f"{filename}.jpg"
                cv2.imwrite(str(crop_path), crop)

                # Salva il frame completo (non tagliato)
                frame_path = frame_dir / f"{filename}_full.jpg"
                cv2.imwrite(str(frame_path), frame)

                # Salva i dati del bounding box in un file txt
                bbox_path = bbox_dir / f"{filename}.txt"
                with open(bbox_path, 'w') as f:
                    f.write(f"Frame: {frame_count}\n")
                    f.write(f"Object ID: {object_count}\n")
                    f.write(f"Class: {class_name}\n")
                    f.write(f"Confidence: {conf:.4f}\n")
                    f.write(f"\n--- Absolute Coordinates (pixels) ---\n")
                    f.write(f"x1: {x1}\n")
                    f.write(f"y1: {y1}\n")
                    f.write(f"x2: {x2}\n")
                    f.write(f"y2: {y2}\n")
                    f.write(f"width: {w}\n")
                    f.write(f"height: {h}\n")
                    f.write(f"\n--- Normalized Coordinates (0-1) ---\n")
                    f.write(f"x1_norm: {x1_norm:.6f}\n")
                    f.write(f"y1_norm: {y1_norm:.6f}\n")
                    f.write(f"width_norm: {w_norm:.6f}\n")
                    f.write(f"height_norm: {h_norm:.6f}\n")
                    f.write(f"\n--- Image Dimensions ---\n")
                    f.write(f"frame_width: {img_w}\n")
                    f.write(f"frame_height: {img_h}\n")

                print(f"✓ Salvato oggetto {object_count}: {class_name} (conf: {conf:.2f}) - Frame {frame_count}")

    cap.release()

    print(f"\n✅ Video elaborato!")
    print(f"📁 File salvati in: {base_dir}/")
    print(f"🎯 Totale oggetti rilevati: {object_count}")
    print(f"🎬 Totale frame processati: {frame_count}")
    print(f"\n📊 Distribuzione per classe:")
    for class_name, count in sorted(class_counts.items()):
        print(f"   - {class_name}: {count} oggetti")

if __name__ == "__main__":
    segment_video("video/tantioggetti.mp4",confidence=0.50)
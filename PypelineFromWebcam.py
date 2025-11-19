import cv2
from datetime import datetime
from pathlib import Path
from pipelineCreazioneDataset import pipelineCreazioneDataset
import time

def record_from_webcam(output_path="video/webcam_recording.mp4", duration=None, fps=60.0, resolution=(1280, 720)):
    """
    Registra un video dalla webcam e lo salva.

    Args:
        output_path: Percorso dove salvare il video
        duration: Durata in secondi (None = continua finché si preme 'q')
        fps: Frame per secondo
        resolution: Risoluzione del video (width, height)

    Returns:
        str: Percorso del video salvato
    """
    # Crea la directory se non esiste
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Errore: impossibile aprire la webcam")
        return None

    # Imposta la risoluzione
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Ottieni le dimensioni reali del frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Webcam aperta con risoluzione: {width}x{height}")

    # Definisci il codec e crea il VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🔴 Registrazione iniziata...")
    if duration:
        print(f"⏱️  Durata: {duration} secondi")
    else:
        print("⏱️  Premi 'q' per fermare la registrazione")

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Errore nella lettura del frame")
            break

        # Scrivi il frame nel video
        out.write(frame)
        frame_count += 1

        # Mostra il frame con indicatore di registrazione
        display_frame = frame.copy()
        elapsed = time.time() - start_time

        # Aggiungi indicatore REC
        cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(display_frame, "REC", (50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Aggiungi timer
        cv2.putText(display_frame, f"Time: {elapsed:.1f}s", (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Registrazione Webcam - Premi Q per fermare', display_frame)

        # Verifica se fermare la registrazione
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️  Registrazione fermata dall'utente")
            break

        # Ferma dopo la durata specificata
        if duration and elapsed >= duration:
            print(f"\n⏹️  Registrazione completata: {duration} secondi")
            break

    # Rilascia le risorse
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Video salvato: {output_path}")
    print(f"📊 Frame registrati: {frame_count}")
    print(f"⏱️  Durata totale: {time.time() - start_time:.2f} secondi")

    return output_path


def webcam_to_dataset_pipeline(
    video_output_path=None,
    duration=None,
    fps=60.0,
    resolution=(1920, 1080),
    outputscan_dir="runs/segment",
    db_path="detections.db",
    feature_dir="features_db",
    crops_dir="crops_db",
    similarity_threshold=0.85,
    delete_after_import=True
):
    """
    Pipeline completa: registra dalla webcam e applica la pipeline di creazione dataset.

    Args:
        video_output_path: Percorso dove salvare il video (default: auto-generato con timestamp)
        duration: Durata registrazione in secondi (None = manuale con 'q')
        fps: Frame per secondo
        resolution: Risoluzione del video
        outputscan_dir: Directory output per segmentazione
        db_path: Percorso database
        feature_dir: Directory features
        crops_dir: Directory crops
        similarity_threshold: Soglia di similarità
        delete_after_import: Elimina file dopo import
    """
    print("="*60)
    print("🚀 AVVIO PIPELINE: WEBCAM → DATASET")
    print("="*60)

    # Genera nome file automatico se non specificato
    if video_output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_output_path = f"video/webcam_{timestamp}.mp4"

    # FASE 1: Registrazione dalla webcam
    print("\n" + "="*60)
    print("📹 FASE 1: REGISTRAZIONE DALLA WEBCAM")
    print("="*60)

    video_path = record_from_webcam(
        output_path=video_output_path,
        duration=duration,
        fps=fps,
        resolution=resolution
    )

    if video_path is None:
        print("❌ Errore durante la registrazione del video")
        return None

    # FASE 2: Applicazione pipeline creazione dataset
    print("\n" + "="*60)
    print("⚙️  FASE 2: APPLICAZIONE PIPELINE CREAZIONE DATASET")
    print("="*60)

    try:
        pipelineCreazioneDataset(
            video_path=video_path,
            outputscan_dir=outputscan_dir,
            db_path=db_path,
            feature_dir=feature_dir,
            crops_dir=crops_dir,
            similarity_threshold=similarity_threshold,
            delete_after_import=delete_after_import,
            confidence=0.10,

        )

        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETATA CON SUCCESSO!")
        print("="*60)
        print(f"📁 Video salvato in: {video_path}")
        print(f"📁 Risultati in: {outputscan_dir}/")
        # Il path del video annotato viene stampato dalla pipeline stessa

        return video_path

    except Exception as e:
        print(f"\n❌ Errore durante la pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Esempio 1: Registrazione manuale (premi 'q' per fermare)
    print("Avvio registrazione dalla webcam...")
    print("Mostra gli oggetti alla webcam e premi 'q' quando hai finito")

    webcam_to_dataset_pipeline(
        duration=None,  # Registrazione manuale
        fps=60.0,
        resolution=(1280, 720),
        similarity_threshold=0.85,
        delete_after_import=True
    )

    # Esempio 2: Registrazione automatica di 30 secondi
    # webcam_to_dataset_pipeline(
    #     duration=30,  # 30 secondi
    #     fps=30.0,
    #     resolution=(1280, 720),
    #     similarity_threshold=0.85,
    #     delete_after_import=True
    # )

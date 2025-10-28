import cv2
from ultralytics import YOLO

# ⚠️ PASSO 1: Configurazione del Modello
# Sostituisci questo percorso con il file dei pesi del tuo modello addestrato (best.pt).
# Se hai usato lo script di addestramento precedente, sarà in una cartella simile a questa:
MODEL_PATH = 'runs/train/yolo_Keys/weights/best.pt'

# PASSO 2: Configurazione della Webcam
# 0 è l'indice predefinito per la webcam principale del sistema.
# Se hai più webcam, potresti dover usare 1, 2, ecc.
WEBCAM_INDEX = 0

# ----------------------------------------------------------------------

def run_webcam_detection(model_path, webcam_index):
    """
    Carica il modello YOLO e avvia il rilevamento degli oggetti in tempo reale dalla webcam.
    """
    try:
        # Carica il tuo modello addestrato
        model = YOLO(model_path)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il modello da {model_path}.")
        print("Assicurati che il percorso sia corretto e che il file esista.")
        print(f"Dettagli: {e}")
        return

    # Inizializza la cattura video dalla webcam
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"ERRORE: Impossibile aprire la webcam all'indice {webcam_index}.")
        return

    print("--- Avvio Rilevamento in Tempo Reale (Premi 'q' per uscire) ---")

    while True:
        # Cattura frame per frame
        ret, frame = cap.read()
        if not ret:
            print("Errore nella ricezione del frame. Uscita...")
            break

        # Esegui l'inferenza (rilevamento) sul frame
        # L'argomento 'stream=True' attiva la modalità generatore per l'inferenza in tempo reale,
        # leggermente più veloce.
        results = model(frame, conf=0.5,stream=True, verbose=False)

        # Disegna i bounding box sul frame
        for r in results:
            # Il metodo .plot() disegna automaticamente box e etichette sul frame
            annotated_frame = r.plot()
        
        # Mostra il frame con le annotazioni
        cv2.imshow('YOLO Webcam Detection - Chiavi', annotated_frame)

        # Interrompi il ciclo se viene premuto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascio della cattura e chiusura delle finestre
    cap.release()
    cv2.destroyAllWindows()
    print("--- Rilevamento interrotto. ---")

if __name__ == "__main__":
    run_webcam_detection(MODEL_PATH, WEBCAM_INDEX)
"""
VideoResultsViewer.py - Visualizza i risultati della pipeline con bounding box e ID database
"""

import cv2
import numpy as np
from pathlib import Path
from dataset import ObjectDatabase
import json
from datetime import datetime


class VideoResultsViewer:
    """Classe per visualizzare i risultati della detection su video con ID database"""
    
    def __init__(self, db_path="detections.db", crops_dir="crops_db"):
        self.db = ObjectDatabase(db_path=db_path, feature_dir="features_db", crops_dir=crops_dir)
        self.colors = {}
        
    def get_class_color(self, class_name):
        """Ottiene un colore univoco per ogni classe"""
        if class_name not in self.colors:
            # Genera colore basato sull'hash del nome classe
            np.random.seed(hash(class_name) % 2**32)
            self.colors[class_name] = tuple(map(int, np.random.randint(50, 255, 3)))
        return self.colors[class_name]
    
    def show_detection_results(self, video_path, mapping_file="detection_mapping.json", 
                              output_path=None, show_live=True, fps_override=None):
        """
        Mostra il video con i bounding box e gli ID del database
        
        Args:
            video_path: Percorso del video originale
            mapping_file: File JSON con il mapping frame->detections->db_id
            output_path: Se specificato, salva il video annotato
            show_live: Se True, mostra il video in tempo reale
            fps_override: FPS per il video di output (None = usa FPS originale)
        """
        print(f"🎬 Avvio visualizzazione risultati...")
        print(f"   Video: {video_path}")
        print(f"   Mapping: {mapping_file}")
        print(f"   Output: {output_path}")
        print(f"   Show live: {show_live}")

        # Carica il mapping
        if not Path(mapping_file).exists():
            print(f"❌ File mapping non trovato: {mapping_file}")
            print("   Assicurati di aver eseguito la pipeline con save_mapping=True")
            return False

        try:
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            print(f"✅ Mapping caricato: {len(mapping)} frame con detection")
        except Exception as e:
            print(f"❌ Errore nel caricare mapping: {e}")
            return False

        # Verifica video
        if not Path(video_path).exists():
            print(f"❌ Video non trovato: {video_path}")
            return False

        # Apri il video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Impossibile aprire il video: {video_path}")
            return False

        # Proprietà video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video aperto correttamente")
        print(f"   Risoluzione: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Frame totali: {total_frames}")
        
        # Setup writer se richiesto
        writer = None
        if output_path:
            # Crea la directory se non esiste
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory creata: {output_dir}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                print(f"❌ Impossibile creare writer per: {output_path}")
                cap.release()
                return False

            print(f"✅ Writer video creato per: {output_path}")

        frame_idx = 0
        frames_written = 0
        detections_drawn = 0

        print(f"\n▶️  Elaborazione video...")
        if show_live:
            print("   Premi 'q' per uscire, 'p' per pausa, SPACE per frame singolo")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
            
            # Crea copia per annotazioni
            annotated_frame = frame.copy()
            
            # Cerca detections per questo frame
            frame_key = str(frame_idx)
            if frame_key in mapping:
                detections = mapping[frame_key]
                
                for det in detections:
                    db_id = det.get('db_id')
                    class_name = det.get('class_name')
                    bbox = det.get('bbox')
                    confidence = det.get('confidence', 0)
                    
                    if not bbox or db_id is None:
                        continue
                    
                    # Estrai coordinate
                    x1 = int(bbox.get('x1', 0))
                    y1 = int(bbox.get('y1', 0))
                    x2 = int(bbox.get('x2', 0))
                    y2 = int(bbox.get('y2', 0))
                    
                    # Colore per la classe
                    color = self.get_class_color(class_name)
                    
                    # Disegna bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepara label
                    label = f"ID:{db_id} {class_name}"
                    conf_text = f"{confidence:.2f}" if confidence > 0 else ""
                    
                    # Background per il testo
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    conf_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    # Rettangolo background per label principale
                    cv2.rectangle(annotated_frame, 
                                (x1, y1 - label_size[1] - 8), 
                                (x1 + label_size[0] + 4, y1), 
                                color, -1)
                    
                    # Testo label
                    cv2.putText(annotated_frame, label, 
                              (x1 + 2, y1 - 4),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Confidence in basso
                    if conf_text:
                        cv2.rectangle(annotated_frame,
                                    (x1, y2),
                                    (x1 + conf_size[0] + 4, y2 + conf_size[1] + 4),
                                    color, -1)
                        cv2.putText(annotated_frame, conf_text,
                                  (x1 + 2, y2 + conf_size[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    detections_drawn += 1

            # Info frame
            info_text = f"Frame: {frame_idx}/{total_frames}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Salva frame se richiesto
            if writer:
                writer.write(annotated_frame)
                frames_written += 1

            # Mostra frame se richiesto
            if show_live:
                cv2.imshow('Risultati Detection - q:esci p:pausa SPACE:step', annotated_frame)
                
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):
                    print("\n⏹️  Interrotto dall'utente")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️  Pausa' if paused else '▶️  Riprendi'}")
                elif key == ord(' '):
                    paused = True
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            print(f"✅ Writer rilasciato")
        if show_live:
            cv2.destroyAllWindows()
        
        print(f"\n✅ Elaborazione completata!")
        print(f"   Frame elaborati: {frame_idx}")
        print(f"   Frame salvati: {frames_written}")
        print(f"   Bounding box disegnati: {detections_drawn}")

        if output_path:
            if Path(output_path).exists():
                size_mb = Path(output_path).stat().st_size / (1024*1024)
                print(f"💾 Video salvato: {output_path}")
                print(f"   Dimensione: {size_mb:.2f} MB")

                if size_mb < 0.1:
                    print(f"⚠️  ATTENZIONE: File molto piccolo, potrebbe essere vuoto")
                if detections_drawn == 0:
                    print(f"⚠️  ATTENZIONE: Nessun bounding box disegnato!")
                    print(f"   Verifica che il mapping contenga db_id validi")
            else:
                print(f"❌ ERRORE: File non creato: {output_path}")
                return False

        return True

    def create_results_summary_video(self, video_path, output_path="results_summary.mp4"):
        """
        Crea un video riassuntivo con tutte le detections dal database
        
        Args:
            video_path: Video originale
            output_path: Dove salvare il video riassuntivo
        """
        # Ottieni tutte le classi e oggetti
        classes = self.db.get_all_classes()
        
        if not classes:
            print("❌ Nessun oggetto nel database")
            return
        
        # Apri video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Impossibile aprire: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"🎬 Creazione video riassuntivo...")
        print(f"   Classi trovate: {len(classes)}")
        
        # Crea frame riassuntivo
        summary_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_offset = 50
        for cls in classes:
            class_name = cls['class_name']
            count = cls['object_count']
            color = self.get_class_color(class_name)
            
            text = f"{class_name}: {count} oggetti"
            cv2.putText(summary_frame, text, (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_offset += 50
        
        # Scrivi frame riassuntivo per 3 secondi
        for _ in range(int(fps * 3)):
            writer.write(summary_frame)
        
        # Processa video originale
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            writer.write(frame)
        
        cap.release()
        writer.release()
        
        print(f"✅ Video riassuntivo creato: {output_path}")
    
    def close(self):
        """Chiude il database"""
        self.db.close()


def show_pipeline_results(video_path, mapping_file="detection_mapping.json", 
                         output_video=None, show_live=True):
    """
    Funzione helper per mostrare i risultati della pipeline
    
    Args:
        video_path: Video originale
        mapping_file: File JSON con mapping delle detections
        output_video: Path per salvare video annotato (opzionale)
        show_live: Se mostrare in tempo reale
    """
    viewer = VideoResultsViewer()
    viewer.show_detection_results(
        video_path=video_path,
        mapping_file=mapping_file,
        output_path=output_video,
        show_live=show_live
    )
    viewer.close()


if __name__ == "__main__":
    # Esempio: mostra risultati del video più recente
    import glob
    from datetime import datetime

    # Trova l'ultimo video registrato
    videos = glob.glob("video/webcam_*.mp4")
    if videos:
        latest_video = max(videos, key=lambda x: Path(x).stat().st_mtime)
        print(f"📹 Visualizzazione risultati per: {latest_video}")
        
        # Genera nome file unico con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(latest_video).stem
        output_video_path = f"runs/resultsWebcam/{video_name}_annotated_{timestamp}.mp4"

        show_pipeline_results(
            video_path=latest_video,
            mapping_file="detection_mapping.json",
            output_video=output_video_path,
            show_live=True
        )
    else:
        print("❌ Nessun video trovato")
        print("💡 Esempio con video esistente:")

        # Genera nome file unico con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = f"runs/resultsWebcam/walletKeyGlasses_annotated_{timestamp}.mp4"

        show_pipeline_results(
            video_path="video/walletKeyGlasses.mp4",
            mapping_file="detection_mapping.json",
            output_video=output_video_path,
            show_live=True
        )

from ProvaScanDaVideo import segment_video
from VerifyFinds import verify_detections
from DinoFeatureExtraction import verify_all_classes
from SaveInDatabaseFounds import process_segment_results
from pathlib import Path
from time import perf_counter
import torch



def pipelineCreazioneDataset(video_path, outputscan_dir="runs/segment", db_path="detections.db",
                             feature_dir="features_db", crops_dir="crops_db",
                             similarity_threshold=0.85, delete_after_import=True, confidence=0.45):
    print("INIZIO IL MIO LAVORO")

    # Check device
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"🚀 Utilizzo GPU: {device}")
    else:
        print("💻 Utilizzo CPU")

    start_time = perf_counter()


    segment_video(input_path=video_path, confidence=confidence, output_dir=outputscan_dir,)#model_path="runs/train/yoloworld_generale/weights/best.pt")

    verify_detections(
        segment_dir=outputscan_dir,
        output_dir="runs/verification",
        conf_threshold=0.75,
        delete_negatives=True,
        distance_threshold=0.2
    )

    verify_all_classes(
        segment_dir=outputscan_dir,
        similarity_threshold=0.75,
        delete_duplicates=True,
        output_dir="runs/verification/similarity"
    )

    mapping = process_segment_results(
        segment_dir=outputscan_dir,
        db_path=db_path,
        similarity_threshold=similarity_threshold,
        delete_after_import=delete_after_import,
        rotations=16,
        save_mapping=True,
        mapping_output="detection_mapping.json"
    )

    elapsed = perf_counter() - start_time
    print(f"IL MIO LAVORO QUI è FINITO — tempo impiegato: {elapsed:.2f} secondi")
    return mapping


if __name__ == "__main__":
    print("YUPPIIIIIIIIIIIIIIIIIIIIII")
    pipelineCreazioneDataset(video_path="video/4obj1.mp4")

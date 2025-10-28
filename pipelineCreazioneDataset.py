from sympy import rotations

from ProvaScanDaVideo import segment_video
from VerifyFinds import verify_detections
from DinoFeatureExtraction import verify_all_classes
from SaveInDatabaseFounds import process_segment_results


def pipelineCreazioneDataset(video_path,outputscan_dir="runs/segment",db_path="detections.db",feature_dir="features_db",crops_dir="crops_db", similarity_threshold=0.85, delete_after_import=True):
    print("INIZIO IL MIO LAVORO")
    segment_video(input_path=video_path,confidence=0.46)
    verify_detections(
        segment_dir=outputscan_dir,
        output_dir="runs/verification",
        conf_threshold=0.75,  # Soglia di confidenza per la verifica
        delete_negatives=True,  # Elimina i file relativi ai negativi
        distance_threshold=0.2
    )
    verify_all_classes(
        segment_dir=outputscan_dir,
        similarity_threshold=0.75,
        delete_duplicates=True,
        output_dir="runs/verification/similarity"

    )

    process_segment_results(
        segment_dir=outputscan_dir,
        db_path=db_path,
        feature_dir=feature_dir,
        crops_dir=crops_dir,
        similarity_threshold=similarity_threshold,
        delete_after_import=delete_after_import,
        rotations=16
    )
    print("IL MIO LAVORO QUI è FINITO")
    return None

if __name__ == "__main__":
    print("YUPPIIIIIIIIIIIIIIIIIIIIII")
    pipelineCreazioneDataset(video_path="video/4obj1.mp4")


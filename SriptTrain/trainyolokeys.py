import os
import argparse
from ultralytics import YOLO
import torch

def main(data_dir, model_name='yolov8n.pt', epochs=50, imgsz=640):
    # Check if dataset exists
    data_yaml = os.path.join(data_dir, 'data.yaml')
    if not os.path.exists(data_yaml):
        print(f"data.yaml not found in {data_dir}. Please provide a valid YOLO dataset.")
        return

    # Train YOLO model
    model = YOLO(model_name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project='runs/train',
        name='yolo_Keys',
        exist_ok=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch=16  # Reduce batch size to help with CUDA OOM
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on datasetchiavi folder")
    parser.add_argument('--data_dir', type=str, default='../datasetchiavi', help='Path to dataset folder')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    args = parser.parse_args()

    main(args.data_dir, args.model, args.epochs, args.imgsz)
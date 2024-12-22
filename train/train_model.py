import argparse
from ultralytics import YOLO

def train_model(model_version, data_path, epochs , imgsz):
    # Initialize the YOLOv8 model with the specified version
    model = YOLO(model_version)
    
    # Train the model on the specified dataset
    results = model.train(
        data=data_path,  # Path to the dataset configuration file
        epochs=epochs,   # Number of training epochs
        imgsz=imgsz

    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model.")
    parser.add_argument(
        "--model_version", 
        type=str, 
        default="yolov8s.pt", 
        help="Specify the YOLOv8 version to use (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)."
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="Traffic-and-Road-Signs-1/data.yaml", 
        help="Path to the dataset YAML file."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=300, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=416, 
        help="Input image size for the model."
    )
    
    args = parser.parse_args()
    train_model(args.model_version, args.data_path, args.epochs, args.imgsz)

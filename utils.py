# This is a utils file for all related functions to avoid repetition in code

# Backbone extraction of YOLO
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
model_path = 'train/weights/best.pt'

class YOLOClassifier:
    def __init__(self, model_path, num_classes):
        self.model = YOLO(model_path)
        self.backbone = self.model.model.model[:10]  # Extract CSPDarknet53 backbone
        self.num_classes = num_classes

        # Build the classification model
        self.classify_model = nn.Sequential(
            self.backbone,
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes)  # Classification layer
        )

        # Define preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((416, 416)),  # Resize for YOLO input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_output_channels(self):
        sample_image = torch.randn(1, 3, 416, 416)  # Random input
        sample_output = self.backbone(sample_image)
        return sample_output.shape[1]  # Output channels

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image).unsqueeze(0)  # Add batch dimension
    

# Get YOLO model outputs
def get_yolo_output(model, image):
    results = model(image)
    return results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls

# Plot function
def plot_boxes(ax, boxes, scores, labels, title, image_tensor, model):
    ax.imshow(image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"{model.names[int(label)]}: {score:.2f}", bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title(title)
    ax.axis('off')

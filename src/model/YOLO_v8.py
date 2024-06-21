import torch
from ultralytics import YOLO

def  get_image_components(image):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    # Perform detection
    results = model(image)
    # Extract predicted classes
    detected_classes = set()
    for result in results:
        for pred in result.boxes.cls.tolist():
            class_id = int(pred)
            detected_classes.add(result.names[class_id])
    
    return list(detected_classes)
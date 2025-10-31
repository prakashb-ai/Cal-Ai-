from ultralytics import YOLO
from PIL import Image
import numpy as np
from utils.nutriation import get_nutrition
from transformers import pipeline
from app.config import STANDARD_VOLUME

# Load YOLOv8 and depth once
yolo_model = YOLO("yolov8n.pt")
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

def detect_items(img: Image.Image):
    results = yolo_model(img)
    depth_map = np.array(depth_estimator(img)['depth'])
    detections = []

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = yolo_model.names[cls].title()
                if conf > 0.25:
                    h_img, w_img = depth_map.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    mean_depth = np.mean(depth_map[int(y1):int(y2), int(x1):int(x2)]) if x2>x1 and y2>y1 else 0
                    area = (x2-x1)*(y2-y1)
                    volume_proxy = area * mean_depth
                    nutrition_info = get_nutrition(label, volume_proxy)
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "mean_depth": mean_depth,
                        "volume_proxy": volume_proxy,
                        "nutrition": nutrition_info
                    })
    return detections

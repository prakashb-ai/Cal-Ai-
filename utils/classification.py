from transformers import pipeline
from collections import Counter
from PIL import Image
from app.config import HIGH_CONF_THRESHOLD



model_names = [
    {"name": "nateraw/food", "dataset": "Food-101", "pipeline": None},
    {"name": "Kaludi/food-category-classification-v2.0", "dataset": "General Food Categories (incl. Fruits, Veggies, Desserts)", "pipeline": None},
    {"name": "eslamxm/vit-base-food101", "dataset": "Food-101 (ViT)", "pipeline": None},
    {"name": "Shresthadev403/food-image-classification", "dataset": "Food-101", "pipeline": None},
    {"name": "DrishtiSharma/finetuned-ViT-Indian-Food-Classification-v3", "dataset": "Indian Food", "pipeline": None},
    {"name": "AventIQ-AI/Food-Classification-AI-Model", "dataset": "Diverse International Foods (101 categories)", "pipeline": None},
    {"name": "Kaludi/Food-Classification", "dataset": "International Foods (e.g., Sushi, Ramen, Falafel)", "pipeline": None},
    {"name": "Prasanna18/indian-food-classification", "dataset": "Indian Food (incl. Desserts, Snacks)", "pipeline": None},
    {"name": "rajistics/finetuned-indian-food", "dataset": "Indian Food Images", "pipeline": None},
    {"name": "nisuga/food_type_classification_model", "dataset": "USDA FoodData Central", "pipeline": None},
    {"name": "dwililiya/food101-model-classification", "dataset": "Food-101", "pipeline": None},
    {"name": "chrisis2/vit-food-classification-chrisis2", "dataset": "Food-101", "pipeline": None},
    {"name": "DrishtiSharma/finetuned-SwinT-Indian-Food-Classification-v1", "dataset": "Indian Food", "pipeline": None},
    {"name": "DrishtiSharma/finetuned-ViT-Indian-Food-Classification-v1", "dataset": "Indian Food", "pipeline": None},
    {"name": "DrishtiSharma/finetuned-SwinT-Indian-Food-Classification-v2", "dataset": "Indian Food", "pipeline": None},
    {"name": "DrishtiSharma/finetuned-SwinT-Indian-Food-Classification-v3", "dataset": "Indian Food", "pipeline": None},
    {"name": "Luke537/image_classification_food_model", "dataset": "Food-101", "pipeline": None},
    {"name": "jcharlie39/learn_Hugging_Face_Food_Classification_Model_using_Distilbert_Uncased_Model", "dataset": "Food-101", "pipeline": None},
    {"name": "larimei/food-classification-ai-resnet-5e", "dataset": "Food-101", "pipeline": None},
    {"name": "Hanhpt23/vit_classification_food", "dataset": "Food-101", "pipeline": None}
]


loaded_models = []

for name in model_names:
    try:
        loaded_models.append({
            "name":name,
            "pipeline":pipeline("image-classification",model=name)

        })
    except:
        continue

def classify_image(img: Image.Image):
    predictions = []
    category_type ="unknown"
    for model_info in loaded_models:
        try:
            results = model_info["pipeline"](img)
            top_pred = results[0]
            label = top_pred["label"].replace("_", " ").title()
            confidence = top_pred["store"]
            predictions.append({
                "model":model_info["name"],
                "dataset":"Food Dataset",
                "label":label,
                "confidence": confidence
            })
            if model_info["name"] == model_names:
                category_type = label
        except:
            continue
    best_pred = max(predictions, key=lambda x: x['confidence']) if predictions else None
    high_conf_preds = [p for p in predictions if p['confidence'] >= HIGH_CONF_THRESHOLD]
    repeated_labels = Counter([p['label'] for p in predictions]).most_common(1)
    
    return predictions, best_pred, category_type



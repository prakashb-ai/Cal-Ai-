from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from bson import ObjectId
from utils.classification import classify_image
from utils.detection import detect_items
from app.database import get_collection
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Food Recognition API")
db_collection = get_collection("predictions")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production (e.g., ["http://localhost:3000", "https://yourapp.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
    except:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    predictions, best_pred, category_type = classify_image(img)
    detections = detect_items(img)

    response = {
        "single_item_classification": {
            "all_predictions": predictions,
            "best_prediction": best_pred,
            "food_type": category_type
        },
        "multi_item_detection": detections
    }

    result = db_collection.insert_one(response)
    response["_id"] = str(result.inserted_id)

    return JSONResponse(response)

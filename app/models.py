from pydantic import BaseModel
from typing import List, Optional



class Nutrition(BaseModel):
    product_name: str
    calories: float
    carbs: float
    proteins: float
    fats: float
    fiber: float
    scale_factor: float

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[int]
    mean_depth: float
    volume_proxy: float
    nutrition: Optional[Nutrition]

class SinglePrediction(BaseModel):
    model: str
    dataset: str
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    single_item_classification: dict
    multi_item_detection: List[Detection]

   
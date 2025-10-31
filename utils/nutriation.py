import requests
from app.config import STANDARD_VOLUME

def get_nutrition(food_name: str, volume_proxy: float):
    scale = volume_proxy / STANDARD_VOLUME if STANDARD_VOLUME > 0 else 1.0
    search_query = food_name.replace(' ', '%20')
    api_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={search_query}&search_simple=1&json=1&fields=product_name,nutriments"
    try:
        response = requests.get(api_url)
        data = response.json()
        if data.get("count", 0) > 0:
            product = data["products"][0]
            nutriments = product.get("nutriments", {})
            return {
                "product_name": product.get("product_name", "Unknown"),
                "calories": nutriments.get("energy-kcal_100g", 0) * scale,
                "carbs": nutriments.get("carbohydrates_100g", 0) * scale,
                "proteins": nutriments.get("proteins_100g", 0) * scale,
                "fats": nutriments.get("fat_100g", 0) * scale,
                "fiber": nutriments.get("fiber_100g", 0) * scale,
                "scale_factor": scale
            }
    except:
        return {
            "product_name": food_name,
            "calories": 0,
            "carbs": 0,
            "proteins": 0,
            "fats": 0,
            "fiber": 0,
            "scale_factor": scale
        }

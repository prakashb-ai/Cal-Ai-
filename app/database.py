from pymongo import MongoClient
from app.config import MONGODB_CONNECTION_URL, DB_NAME

client = MongoClient(MONGODB_CONNECTION_URL)
db = client[DB_NAME]

def get_collection(name="predictions"):
    return db[name]
from pymongo import MongoClient

MONGO_URL = "mongodb://localhost:27017"
DATABASE_NAME = "student_information"

client = MongoClient(MONGO_URL)
db = client[DATABASE_NAME]

def get_db():
    return db
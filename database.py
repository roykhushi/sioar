import motor.motor_asyncio
from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel, Field
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "Smart_Food_Link")

# Initialize async MongoDB client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Collections
ngos_collection = db["ngos"]
predictions_collection = db["predictions"]
users_collection = db["users"]


class AddressModel(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None


class NGOModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    location: str
    contact: str
    email: Optional[str] = None
    categories_accepted: List[str]
    address: Optional[AddressModel] = None
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True


class PredictionModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    item_name: str
    price: float
    quantity: int
    avg_daily_sales: float
    days_until_expiry: int
    category: str
    risk_level: str  # "High", "Medium", "Low"
    probability: float
    action: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True


# async def create_ngo(ngo_data: dict) -> dict:
#     """Create a new NGO record"""
#     ngo_data["created_at"] = datetime.utcnow()
#     ngo_data["updated_at"] = datetime.utcnow()
#     result = await ngos_collection.insert_one(ngo_data)
#     return {"_id": str(result.inserted_id), **ngo_data}


async def get_all_ngos() -> List[dict]:
    """Get all active NGOs"""
    ngos = []
    cursor = ngos_collection.find({"active": True})
    async for ngo in cursor:
        ngo["_id"] = str(ngo["_id"])  # Convert ObjectId to string
        ngos.append(ngo)
    return ngos


async def get_ngo_by_id(ngo_id: str) -> Optional[dict]:
    """Get a specific NGO by ID"""
    ngo = await ngos_collection.find_one({"_id": ObjectId(ngo_id)})
    if ngo:
        ngo["id"] = str(ngo["_id"])
        return ngo
    return None


async def get_ngos_by_category(category: str) -> List[dict]:
    """Get NGOs that accept a specific category"""
    ngos = []
    cursor = ngos_collection.find({
        "active": True,
        "categories_accepted": category
    })
    async for ngo in cursor:
        ngo["_id"] = str(ngo["_id"])  # Convert ObjectId to string
        ngos.append(ngo)
    return ngos


# async def update_ngo(ngo_id: str, update_data: dict) -> bool:
#     """Update an NGO record"""
#     update_data["updated_at"] = datetime.utcnow()
#     result = await ngos_collection.update_one(
#         {"_id": ObjectId(ngo_id)},
#         {"$set": update_data}
#     )
#     return result.modified_count > 0


# async def delete_ngo(ngo_id: str) -> bool:
#     """Soft delete an NGO (set active to False)"""
#     result = await ngos_collection.update_one(
#         {"_id": ObjectId(ngo_id)},
#         {"$set": {"active": False, "updated_at": datetime.utcnow()}}
#     )
#     return result.modified_count > 0



async def create_prediction(prediction_data: dict) -> dict:
    """Create a new prediction record"""
    prediction_data["created_at"] = datetime.now(timezone.utc)
    prediction_data["updated_at"] = datetime.now(timezone.utc)
    result = await predictions_collection.insert_one(prediction_data)
    return {"_id": str(result.inserted_id), **prediction_data}


async def get_all_predictions(limit: int = 100, user_id: Optional[str] = None) -> List[dict]:
    """Get recent predictions, scoped to a user when user_id is provided"""
    query = {} if user_id is None else {"user_id": user_id}
    predictions = []
    cursor = predictions_collection.find(query).sort("created_at", -1).limit(limit)
    async for pred in cursor:
        pred["_id"] = str(pred["_id"])
        predictions.append(pred)
    return predictions


async def get_prediction_by_id(pred_id: str) -> Optional[dict]:
    """Get a specific prediction by ID"""
    prediction = await predictions_collection.find_one({"_id": ObjectId(pred_id)})
    if prediction:
        prediction["_id"] = str(prediction["_id"])  # Convert ObjectId to string
        return prediction
    return None


async def get_predictions_by_risk_level(risk_level: str, limit: int = 50) -> List[dict]:
    """Get predictions filtered by risk level"""
    predictions = []
    cursor = predictions_collection.find({"risk_level": risk_level}).limit(limit)
    async for pred in cursor:
        pred["_id"] = str(pred["_id"])  # Convert ObjectId to string
        predictions.append(pred)
    return predictions


async def get_predictions_stats(user_id: Optional[str] = None) -> dict:
    """Get statistics about predictions, scoped to a user when user_id is provided"""
    pipeline = []
    if user_id:
        pipeline.append({"$match": {"user_id": user_id}})
    pipeline.append({
        "$group": {
            "_id": "$risk_level",
            "count": {"$sum": 1}
        }
    })
    stats = {}
    async for item in predictions_collection.aggregate(pipeline):
        stats[item["_id"]] = item["count"]

    return {
        "High": stats.get("High", 0),
        "Medium": stats.get("Medium", 0),
        "Low": stats.get("Low", 0),
        "total": sum(stats.values())
    }


# ═══════════════════════════════════════════════════════════════
# DASHBOARD METRICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

async def count_high_risk_predictions(user_id: Optional[str] = None) -> int:
    """Count predictions with High risk level, scoped to user"""
    query: dict = {"risk_level": "High"}
    if user_id:
        query["user_id"] = user_id
    return await predictions_collection.count_documents(query)


async def count_donation_predictions(user_id: Optional[str] = None) -> int:
    """Count predictions marked for donation, scoped to user"""
    query: dict = {"action": "Donate to NGO"}
    if user_id:
        query["user_id"] = user_id
    return await predictions_collection.count_documents(query)


async def get_waste_prevented(user_id: Optional[str] = None) -> float:
    """Calculate total waste prevented, scoped to user"""
    match_stage: dict = {"risk_level": "High", "action": "Donate to NGO"}
    if user_id:
        match_stage["user_id"] = user_id

    pipeline = [
        {"$match": match_stage},
        {"$group": {"_id": None, "total_quantity": {"$sum": "$quantity"}}},
    ]

    result = None
    async for item in predictions_collection.aggregate(pipeline):
        result = item
        break

    return result["total_quantity"] if result else 0


async def get_active_ngos_count() -> int:
    """Count active NGOs"""
    count = await ngos_collection.count_documents({"active": True})
    return count


# ═══════════════════════════════════════════════════════════════
# USER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

async def create_user(user_data: dict) -> dict:
    """Create a new user record"""
    user_data["created_at"] = datetime.now(timezone.utc)
    user_data["updated_at"] = datetime.now(timezone.utc)
    result = await users_collection.insert_one(user_data)
    user_data["_id"] = str(result.inserted_id)
    return user_data


async def get_user_by_username(username: str) -> Optional[dict]:
    """Get a user by username"""
    user = await users_collection.find_one({"username": username.lower()})
    if user:
        user["_id"] = str(user["_id"])
    return user


async def get_user_by_email(email: str) -> Optional[dict]:
    """Get a user by email"""
    user = await users_collection.find_one({"email": email.lower()})
    if user:
        user["_id"] = str(user["_id"])
    return user



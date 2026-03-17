"""
MongoDB database configuration and models for Smart-Food Link
"""

import motor.motor_asyncio
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from bson import ObjectId

# MongoDB Connection URL
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "smart_food_link"

# Initialize async MongoDB client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Collections
ngos_collection = db["ngos"]
predictions_collection = db["predictions"]
donations_collection = db["donations"]


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS (for validation)
# ═══════════════════════════════════════════════════════════════

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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

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
    organization_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


class DonationModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    prediction_id: str  # ObjectId as string
    ngo_id: str  # ObjectId as string
    item_name: str
    quantity: int
    donor_notes: Optional[str] = None
    status: str = "pending"  # pending, confirmed, completed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


# ═══════════════════════════════════════════════════════════════
# DATABASE OPERATIONS - NGOs
# ═══════════════════════════════════════════════════════════════

async def create_ngo(ngo_data: dict) -> dict:
    """Create a new NGO record"""
    ngo_data["created_at"] = datetime.utcnow()
    ngo_data["updated_at"] = datetime.utcnow()
    result = await ngos_collection.insert_one(ngo_data)
    return {"_id": str(result.inserted_id), **ngo_data}


async def get_all_ngos() -> List[dict]:
    """Get all active NGOs"""
    ngos = []
    cursor = ngos_collection.find({"active": True})
    async for ngo in cursor:
        ngo["_id"] = str(ngo["_id"])
        ngo["id"] = ngo.pop("_id")
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
        ngo["id"] = str(ngo["_id"])
        ngos.append(ngo)
    return ngos


async def update_ngo(ngo_id: str, update_data: dict) -> bool:
    """Update an NGO record"""
    update_data["updated_at"] = datetime.utcnow()
    result = await ngos_collection.update_one(
        {"_id": ObjectId(ngo_id)},
        {"$set": update_data}
    )
    return result.modified_count > 0


async def delete_ngo(ngo_id: str) -> bool:
    """Soft delete an NGO (set active to False)"""
    result = await ngos_collection.update_one(
        {"_id": ObjectId(ngo_id)},
        {"$set": {"active": False, "updated_at": datetime.utcnow()}}
    )
    return result.modified_count > 0


# ═══════════════════════════════════════════════════════════════
# DATABASE OPERATIONS - PREDICTIONS
# ═══════════════════════════════════════════════════════════════

async def create_prediction(prediction_data: dict) -> dict:
    """Create a new prediction record"""
    prediction_data["created_at"] = datetime.utcnow()
    prediction_data["updated_at"] = datetime.utcnow()
    result = await predictions_collection.insert_one(prediction_data)
    return {"_id": str(result.inserted_id), **prediction_data}


async def get_all_predictions(limit: int = 100, org_id: Optional[str] = None) -> List[dict]:
    """Get recent predictions"""
    query = {} if org_id is None else {"organization_id": org_id}
    predictions = []
    cursor = predictions_collection.find(query).sort("created_at", -1).limit(limit)
    async for pred in cursor:
        pred["id"] = str(pred["_id"])
        predictions.append(pred)
    return predictions


async def get_prediction_by_id(pred_id: str) -> Optional[dict]:
    """Get a specific prediction by ID"""
    prediction = await predictions_collection.find_one({"_id": ObjectId(pred_id)})
    if prediction:
        prediction["id"] = str(prediction["_id"])
        return prediction
    return None


async def get_predictions_by_risk_level(risk_level: str, limit: int = 50) -> List[dict]:
    """Get predictions filtered by risk level"""
    predictions = []
    cursor = predictions_collection.find({"risk_level": risk_level}).limit(limit)
    async for pred in cursor:
        pred["id"] = str(pred["_id"])
        predictions.append(pred)
    return predictions


async def get_predictions_stats() -> dict:
    """Get statistics about predictions"""
    pipeline = [
        {
            "$group": {
                "_id": "$risk_level",
                "count": {"$sum": 1}
            }
        }
    ]
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
# DATABASE OPERATIONS - DONATIONS
# ═══════════════════════════════════════════════════════════════

async def create_donation(donation_data: dict) -> dict:
    """Create a new donation record"""
    donation_data["created_at"] = datetime.utcnow()
    donation_data["updated_at"] = datetime.utcnow()
    result = await donations_collection.insert_one(donation_data)
    return {"_id": str(result.inserted_id), **donation_data}


async def get_donations_by_ngo(ngo_id: str, status: Optional[str] = None) -> List[dict]:
    """Get donations for a specific NGO"""
    query = {"ngo_id": ngo_id}
    if status:
        query["status"] = status

    donations = []
    cursor = donations_collection.find(query).sort("created_at", -1)
    async for donation in cursor:
        donation["id"] = str(donation["_id"])
        donations.append(donation)
    return donations


async def update_donation_status(donation_id: str, status: str) -> bool:
    """Update donation status"""
    result = await donations_collection.update_one(
        {"_id": ObjectId(donation_id)},
        {"$set": {"status": status, "updated_at": datetime.utcnow()}}
    )
    return result.modified_count > 0


# ═══════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

async def init_db():
    """Initialize database indexes"""
    # NGO indexes
    await ngos_collection.create_index("name")
    await ngos_collection.create_index("active")
    await ngos_collection.create_index("categories_accepted")

    # Prediction indexes
    await predictions_collection.create_index("created_at")
    await predictions_collection.create_index("risk_level")
    await predictions_collection.create_index("category")

    # Donation indexes
    await donations_collection.create_index("ngo_id")
    await donations_collection.create_index("status")
    await donations_collection.create_index("created_at")


# ═══════════════════════════════════════════════════════════════
# SEED INITIAL DATA
# ═══════════════════════════════════════════════════════════════

async def seed_ngos():
    """Seed initial NGO data"""
    count = await ngos_collection.count_documents({})
    if count == 0:
        initial_ngos = [
            {
                "name": "City Food Bank",
                "location": "Downtown, City Center",
                "contact": "+91-8800-111-222",
                "email": "contact@cityfoodbank.org",
                "categories_accepted": ["Canned", "Bakery", "Produce"],
                "address": {
                    "street": "123 Main St",
                    "city": "Mumbai",
                    "state": "Maharashtra",
                    "zipcode": "400001"
                },
                "active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "name": "Hope Shelter",
                "location": "North District",
                "contact": "+91-9900-333-444",
                "email": "contact@hopeshelter.org",
                "categories_accepted": ["Dairy", "Meat", "Frozen"],
                "address": {
                    "street": "456 North Ave",
                    "city": "Mumbai",
                    "state": "Maharashtra",
                    "zipcode": "400010"
                },
                "active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "name": "Community Kitchen",
                "location": "East Zone",
                "contact": "+91-8833-555-666",
                "email": "contact@communitykitchen.org",
                "categories_accepted": ["Produce", "Bakery", "Snacks"],
                "address": {
                    "street": "789 East Rd",
                    "city": "Mumbai",
                    "state": "Maharashtra",
                    "zipcode": "400020"
                },
                "active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        await ngos_collection.insert_many(initial_ngos)

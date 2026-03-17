from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import ml_engine
import data_processor
from database import (
    get_all_ngos,
    get_ngos_by_category,
    create_prediction,
    get_all_predictions,
    get_predictions_stats,
    init_db,
    seed_ngos
)

class ItemDetails(BaseModel):
    price: float
    quantity: int
    avg_daily_sales: float
    days_until_expiry: int
    category: str = "Unknown"

class CategoryInput(BaseModel):
    category: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    await init_db()
    await seed_ngos()
    print("✅ Database initialized")
    yield
    # Shutdown: Clean up if needed
    print("🔴 Shutting down...")


app = FastAPI(
    title="Smart-Food Link API",
    version="1.0",
    lifespan=lifespan
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to Smart-Food Link API"}


@app.post("/predict")
async def predict_risk(item: ItemDetails):
    """
    Accepts item details and returns AI recommendation.
    Also stores the prediction in MongoDB.
    """
    try:
        result = ml_engine.predict_item(
            item.price,
            item.quantity,
            item.avg_daily_sales,
            item.days_until_expiry,
            item.category
        )

        # Store prediction in MongoDB
        prediction_record = {
            "item_name": f"{item.category} Item",
            "price": item.price,
            "quantity": item.quantity,
            "avg_daily_sales": item.avg_daily_sales,
            "days_until_expiry": item.days_until_expiry,
            "category": item.category,
            "risk_level": result["Risk_Level"],
            "probability": result["Probability"],
            "action": result["Action"]
        }
        await create_prediction(prediction_record)

        return result
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model not trained. Please call /train first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def trigger_training():
    """
    Triggers the model retraining pipeline manually.
    """
    try:
        # 1. Regenerate/Process data
        data_processor.process_data()
        # 2. Train model
        success = ml_engine.train_model()
        if success:
            return {"message": "Model trained successfully"}
        else:
            raise HTTPException(status_code=500, detail="Model training failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ngos")
async def get_ngos():
    """
    Returns a list of NGOs with their preferred categories from MongoDB.
    """
    try:
        ngos = await get_all_ngos()
        # Convert to format expected by frontend
        formatted_ngos = []
        for ngo in ngos:
            formatted_ngos.append({
                "id": ngo.get("id") or ngo.get("_id"),
                "name": ngo["name"],
                "location": ngo["location"],
                "contact": ngo["contact"],
                "categories_accepted": ngo["categories_accepted"]
            })
        return formatted_ngos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match")
async def match_ngo(cat_input: CategoryInput):
    """
    Input is a food category; returns the closest NGO that accepts it from MongoDB.
    """
    try:
        category = cat_input.category
        matched_ngos = await get_ngos_by_category(category)

        if matched_ngos and len(matched_ngos) > 0:
            ngo = matched_ngos[0]
            # Format response for frontend
            recommended_ngo = {
                "id": ngo.get("id") or ngo.get("_id"),
                "name": ngo["name"],
                "location": ngo["location"],
                "contact": ngo["contact"],
                "categories_accepted": ngo["categories_accepted"]
            }
            return {"recommended_ngo": recommended_ngo}
        else:
            return {"message": "No matching NGO found for this category", "recommended_ngo": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
async def get_predictions(limit: int = 50):
    """
    Returns recent predictions from MongoDB.
    """
    try:
        predictions = await get_all_predictions(limit=limit)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/stats")
async def get_prediction_stats():
    """
    Returns statistics about predictions from MongoDB.
    """
    try:
        stats = await get_predictions_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


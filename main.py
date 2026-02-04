from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ml_engine
import data_processor
import os

app = FastAPI(title="Smart-Food Link API", version="1.0")

class ItemDetails(BaseModel):
    price: float
    quantity: int
    avg_daily_sales: float
    days_until_expiry: int

class CategoryInput(BaseModel):
    category: str

NGO_DB = [
    {"name": "City Food Bank", "preferred_categories": ["Canned", "Bakery", "Produce"]},
    {"name": "Hope Shelter", "preferred_categories": ["Dairy", "Meat", "Frozen"]},
    {"name": "Community Kitchen", "preferred_categories": ["Produce", "Bakery", "Snacks"]}
]

@app.get("/")
def root():
    return {"message": "Welcome to Smart-Food Link API"}

@app.post("/predict")
def predict_risk(item: ItemDetails):
    """
    Accepts item details and returns AI recommendation.
    """
    try:
        result = ml_engine.predict_item(
            item.price, 
            item.quantity, 
            item.avg_daily_sales, 
            item.days_until_expiry
        )
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
def get_ngos():
    """
    Returns a list of NGOs with their preferred categories.
    """
    return NGO_DB

@app.post("/match")
def match_ngo(cat_input: CategoryInput):
    """
    Input is a food category; returns the closest NGO that accepts it.
    (For MVP, 'closest' is just the first match in the list)
    """
    category = cat_input.category
    matched_ngos = []
    
    for ngo in NGO_DB:
        if category in ngo["preferred_categories"]:
            matched_ngos.append(ngo)
            
    if matched_ngos:
        # Return the first match as the "closest" for now
        return {"recommended_ngo": matched_ngos[0]}
    else:
        return {"message": "No matching NGO found for this category", "recommended_ngo": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

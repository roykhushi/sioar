from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timezone, timedelta
import os
import ml_engine
import data_processor
from database import (
    get_all_ngos,
    get_ngos_by_category,
    create_prediction,
    get_all_predictions,
    get_predictions_stats,
    count_high_risk_predictions,
    count_donation_predictions,
    get_waste_prevented,
    get_active_ngos_count,
    create_user,
    get_user_by_username,
    get_user_by_email,
)


SECRET_KEY = os.getenv("JWT_SECRET", "smart-food-link-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()



class ItemDetails(BaseModel):
    price: float
    quantity: int
    avg_daily_sales: float
    days_until_expiry: int
    category: str = "Unknown"

class CategoryInput(BaseModel):
    category: str

class UserSignUp(BaseModel):
    username: str
    email: str
    password: str

class UserSignIn(BaseModel):
    username: str
    password: str



def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await get_user_by_username(username)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _safe_user_response(user: dict) -> dict:
    return {
        "_id": user["_id"],
        "username": user["username"],
        "email": user["email"],
        "created_at": str(user.get("created_at", "")),
    }


app = FastAPI(title="Smart-Food Link API", version="1.0")

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


@app.post("/auth/signup")
async def signup(user_data: UserSignUp):
    existing = await get_user_by_username(user_data.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    existing_email = await get_user_by_email(user_data.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = pwd_context.hash(user_data.password)
    new_user = await create_user({
        "username": user_data.username.lower(),
        "email": user_data.email.lower(),
        "hashed_password": hashed_password,
    })

    access_token = create_access_token({"sub": new_user["username"]})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": _safe_user_response(new_user),
    }


@app.post("/auth/signin")
async def signin(user_data: UserSignIn):
    user = await get_user_by_username(user_data.username.lower())
    if not user or not pwd_context.verify(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token({"sub": user["username"]})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": _safe_user_response(user),
    }


@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return _safe_user_response(current_user)


@app.post("/predict")
async def predict_risk(item: ItemDetails, current_user: dict = Depends(get_current_user)):
    try:
        result = ml_engine.predict_item(
            item.price, item.quantity, item.avg_daily_sales,
            item.days_until_expiry, item.category,
        )
        prediction_record = {
            "user_id": current_user["_id"],
            "item_name": f"{item.category} Item",
            "price": item.price,
            "quantity": item.quantity,
            "avg_daily_sales": item.avg_daily_sales,
            "days_until_expiry": item.days_until_expiry,
            "category": item.category,
            "risk_level": result["Risk_Level"],
            "probability": result["Probability"],
            "action": result["Action"],
        }
        await create_prediction(prediction_record)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model not trained. Please call /train first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def trigger_training(current_user: dict = Depends(get_current_user)):
    try:
        data_processor.process_data()
        success = ml_engine.train_model()
        if success:
            return {"message": "Model trained successfully"}
        raise HTTPException(status_code=500, detail="Model training failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ngos")
async def get_ngos(current_user: dict = Depends(get_current_user)):
    try:
        ngos = await get_all_ngos()
        formatted_ngos = []
        for ngo in ngos:
            formatted_ngos.append({
                "id": ngo.get("id") or ngo.get("_id"),
                "name": ngo["name"],
                "location": ngo["location"],
                "contact": ngo["contact"],
                "categories_accepted": ngo["categories_accepted"],
            })
        return formatted_ngos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match")
async def match_ngo(cat_input: CategoryInput, current_user: dict = Depends(get_current_user)):
    try:
        category = cat_input.category
        matched_ngos = await get_ngos_by_category(category)
        if matched_ngos and len(matched_ngos) > 0:
            ngo = matched_ngos[0]
            recommended_ngo = {
                "id": ngo.get("id") or ngo.get("_id"),
                "name": ngo["name"],
                "location": ngo["location"],
                "contact": ngo["contact"],
                "categories_accepted": ngo["categories_accepted"],
            }
            return {"recommended_ngo": recommended_ngo}
        return {"message": "No matching NGO found for this category", "recommended_ngo": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
async def get_predictions(limit: int = 50, current_user: dict = Depends(get_current_user)):
    try:
        predictions = await get_all_predictions(limit=limit, user_id=current_user["_id"])
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/stats")
async def get_prediction_stats(current_user: dict = Depends(get_current_user)):
    try:
        stats = await get_predictions_stats(user_id=current_user["_id"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/high-risk-count")
async def get_high_risk_count(current_user: dict = Depends(get_current_user)):
    try:
        count = await count_high_risk_predictions(user_id=current_user["_id"])
        return {"high_risk_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/donations-count")
async def get_donations_count(current_user: dict = Depends(get_current_user)):
    try:
        count = await count_donation_predictions(user_id=current_user["_id"])
        return {"donations_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/waste-prevented")
async def get_waste_prevented_endpoint(current_user: dict = Depends(get_current_user)):
    try:
        waste_kg = await get_waste_prevented(user_id=current_user["_id"])
        return {"waste_prevented_kg": waste_kg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/active-ngos-count")
async def get_active_ngos(current_user: dict = Depends(get_current_user)):
    try:
        count = await get_active_ngos_count()
        return {"active_ngos_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/metrics")
async def get_all_dashboard_metrics(current_user: dict = Depends(get_current_user)):
    try:
        uid = current_user["_id"]
        high_risk = await count_high_risk_predictions(user_id=uid)
        donations = await count_donation_predictions(user_id=uid)
        waste = await get_waste_prevented(user_id=uid)
        ngos = await get_active_ngos_count()
        stats = await get_predictions_stats(user_id=uid)

        return {
            "high_risk_count": high_risk,
            "donations_count": donations,
            "waste_prevented_kg": waste,
            "active_ngos_count": ngos,
            "risk_distribution": {
                "High": stats["High"],
                "Medium": stats["Medium"],
                "Low": stats["Low"],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

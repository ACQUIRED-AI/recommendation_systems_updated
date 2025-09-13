# api_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import redis
import json
import logging
from datetime import datetime
import joblib
from matrix_factorization import MatrixFactorization

import sqlite3
import random
import pandas as pd

# Redis for caching
import os
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


app = FastAPI(title="Product Recommendation API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    category_filter: Optional[str] = None
    price_range: Optional[Dict[str, float]] = None

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict]
    timestamp: str
    model_version: str

class InteractionRequest(BaseModel):
    user_id: str
    product_id: str
    interaction_type: str
    rating: Optional[float] = None
    session_id: Optional[str] = None

# Global model instances (loaded at startup)
hybrid_model = None
content_model = None
vectorizer = None
numerical_scaler = None
product_catalog = None


@app.on_event("startup")
async def startup_event():
    global hybrid_model, content_model, vectorizer, numerical_scaler, product_catalog
    # Load trained models & assets
    hybrid_model = joblib.load("models/hybrid_model.pkl")
    content_model = joblib.load("models/content_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    numerical_scaler = joblib.load("models/numerical_scaler.pkl")
    product_catalog = joblib.load("models/product_catalog.pkl")
    logging.info("Models and artifacts loaded successfully")


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a user"""
    try:
        # Check cache first
        cache_key = f"rec:{request.user_id}:{request.num_recommendations}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # Get user features and candidate products
        user_features = get_user_features(request.user_id)
        candidate_products = get_candidate_products(
            request.user_id, 
            request.category_filter,
            request.price_range
        )
        
        # Get recommendations from hybrid model
        recommendations = hybrid_model.get_recommendations(
            request.user_id,
            user_features,
            candidate_products,
            request.num_recommendations
        )
        
        # Format response
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=[
                {
                    "product_id": prod_id,
                    "predicted_rating": rating,
                    "product_details": get_product_details(prod_id)
                }
                for prod_id, rating in recommendations
            ],
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
        # Cache result for 1 hour
        redis_client.setex(cache_key, 3600, response.json())
        
        return response
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/interactions")
async def log_interaction(interaction: InteractionRequest, background_tasks: BackgroundTasks):
    """Log user interaction for model training"""
    try:
        # Add to background task for async processing
        background_tasks.add_task(process_interaction, interaction.dict())
        
        return {"status": "success", "message": "Interaction logged"}
        
    except Exception as e:
        logging.error(f"Error logging interaction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/similar/{product_id}")
async def get_similar_products(product_id: str, limit: int = 10):
    """Get similar products based on content similarity"""
    try:
        similar_products = hybrid_model.content_model.get_similar_items(product_id, limit)
        
        return {
            "product_id": product_id,
            "similar_products": [
                {
                    "product_id": prod_id,
                    "similarity_score": score,
                    "product_details": get_product_details(prod_id)
                }
                for prod_id, score in similar_products
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_user_features(user_id: str) -> Dict:
    conn = sqlite3.connect("asos.db")
    df = pd.read_sql("SELECT product_id, rating FROM user_interactions WHERE user_id = ?", conn, params=[user_id])
    conn.close()
    return {
        "recent_product_ids": df.sort_index().tail(5)["product_id"].tolist(),  # last 5
        "avg_rating": df["rating"].mean() if not df.empty else 3.0
    }

def get_candidate_products(user_id: str, category_filter: str = None, price_range: Dict = None) -> List[str]:
    query = "SELECT product_id, price_current FROM products"
    conn = sqlite3.connect("asos.db")
    df = pd.read_sql(query, conn)
    conn.close()

    if category_filter:
        df = df[df["category"] == category_filter]
    if price_range:
        df = df[(df["price_current"] >= price_range.get("min", 0)) &
                (df["price_current"] <= price_range.get("max", float("inf")))]
    return df["product_id"].tolist()

def get_product_details(product_id: str) -> Dict:
    conn = sqlite3.connect("asos.db")
    query = "SELECT * FROM products WHERE product_id = ?"
    df = pd.read_sql(query, conn, params=[product_id])
    conn.close()
    return df.iloc[0].to_dict() if not df.empty else {}


async def process_interaction(interaction_data: Dict):
    """Process user interaction for model updates"""
    # Implementation for async interaction processing
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
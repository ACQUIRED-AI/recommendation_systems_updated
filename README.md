## Recommendation System

This engine provides product recommendations using a **hybrid model** (Collaborative Filtering + Content-Based + optional User Features). It is exposed via a **FastAPI application** and can be run locally or inside Docker.

**Note for backend team**  
This setup currently uses the ASOS API (via RapidAPI) for fetching sample product data, and generates fake user interactions for training/testing.  
In production, you should replace these with **your product catalog API** and **user interaction API/pipeline**.  

## Models
The training pipeline (training_pipeline.py) exports the following into the models/ directory:
content_model.pkl
vectorizer.pkl
numerical_scaler.pkl
cf_model.pkl (optional)
product_catalog.pkl
hybrid_model.pkl

These are loaded automatically by the API on startup.

## Extra Testing Scripts
These first two files are only included for local testing/demo purposes. Might not need to rerun these 2 files since the data is already in csv and asos.db, unless you want to regenerate the dataset.

1. fetch_asos_products.py
- Fetches sample product data from the ASOS API (via RapidAPI).
- Saves to data/products.csv and asos.db.
Purpose: provide a toy product catalog for development.

2. generate_fake_interactions.py
- Creates data/interactions.csv and populates asos.db with random ratings.
Purpose: simulate interactions for training/testing without real logs.

-> Replace these with your own approach (e.g. your product catalog API + interaction pipeline). You don’t have to follow the method used here.

3. matrix_factorization.py
- This file was added because the initial original project was missing it in order to run properly in the same environment when loading to api_service
- Required so model training/loading .pkl files works correctly.

4. docker-compose.yml
I created this to not complicate having to write command lines when loading the docker.

## Recommended approach to run
The easiest way is with Docker with this line:
-> docker-compose up --build

Then open http://127.0.0.1:8000/docs

OR

## For development/testing without Docker:
Run redis in your terminal:
-> docker run -d --name redis -p 6379:6379 redis

Then
-> uvicorn api_service:app --host 0.0.0.0 --port 8000

Then open http://127.0.0.1:8000/docs

For manual Docker builds:
-> docker build -t recommender-api .
-> docker run --rm -p 8000:8000 recommender-api

-> Finally, use the similar example inputs provided in the following section to see results. Some fields are not yet implemented, so they will currently return as NULL until the corresponding features are completed.

## To Test Recommendations

In Swagger, expand POST /recommendations

Input:

{
  "user_id": "u1",
  "num_recommendations": 5,
  "category_filter": null,
  "price_range": null
}

Click Execute → see JSON with product IDs, scores, names, prices, and URLs.

MORE:

API Endpoints
1. Health Check
GET /healthz

Output
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "cache": "ok"
}
2. Get Recommendations
POST /recommendations

Input
{
  "user_id": "u1",
  "num_recommendations": 5,
  "category_filter": null,
  "price_range": null
}

Output
{
  "user_id": "u1",
  "recommendations": [
    {
      "product_id": "208883908",
      "score": 4.2,
      "name": "Nike Air Force 1",
      "brand": "Nike",
      "category": "Shoes",
      "price_current": 120.0,
      "price_ratio": 0.9,
      "image_url": "https://...",
      "url": "https://asos.com/product/208883908"
    }
  ],
  "timestamp": "2025-09-12T02:00:00Z",
  "model_version": "1.0.0"
}
3. Log Interaction
POST /interactions

Input
{
  "user_id": "u1",
  "product_id": "209018019",
  "interaction_type": "rating",
  "rating": 5
}

Output
{
  "status": "success",
  "message": "Interaction logged"
}


4. Get Similar Products
GET /similar/{product_id}?limit=10

Input
{
  "product_id": "209018019",
  "limit": 10
}

Output
{
  "product_id": "208883908",
  "similar_products": [
    {
      "product_id": "208774201",
      "similarity_score": 0.72,
      "name": "Nike Air Max 90",
      "brand": "Nike",
      "category": "Shoes",
      "price_current": 130.0,
      "image_url": "https://...",
      "url": "https://asos.com/product/208774201"
    }
  ]
}

## Important Notes
- Accuracy is currently based mainly on ratings + product metadata.
- Backend team should handle logging richer user interactions.
- Performance can be improved later by adding user features and product attributes
- Interactions logged through /interactions can be exported for retraining.
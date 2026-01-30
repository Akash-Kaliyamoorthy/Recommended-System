import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('./src'))

from data_preprocessing import DataPreprocessor
from hybrid_recommender import HybridRecommender
from evaluation import RecommenderEvaluator

app = FastAPI(title="OTT Recommender API")

# Global variables for the model and data
preprocessor = None
recommender = None
users_df = None
movies_df = None
interactions_df = None

@app.on_event("startup")
async def startup_event():
    global preprocessor, recommender, users_df, movies_df, interactions_df
    print("[*] Initializing Recommender System...")
    
    preprocessor = DataPreprocessor()
    # Create sample data (in a real app, you would load from DB/CSV)
    users_df, movies_df, interactions_df = preprocessor.create_sample_data()
    preprocessor.engineer_features()
    
    # Update references
    users_df = preprocessor.users_df
    movies_df = preprocessor.movies_df
    interactions_df = preprocessor.interactions_df
    
    # Train Recommender
    recommender = HybridRecommender(
        cf_weight=0.5,
        cb_weight=0.3,
        pop_weight=0.2,
        n_factors=50,
        max_features=1000
    )
    recommender.fit(users_df, movies_df, interactions_df)
    print("[+] Recommender System Ready!")

# Models
class Movie(BaseModel):
    movie_id: int
    title: str
    genre: str
    language: str
    release_year: int
    poster_url: Optional[str] = None
    hybrid_score: Optional[float] = None
    explanation: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Movie]

# Endpoints
@app.get("/api/users")
async def get_users():
    """Get a list of sample users."""
    sample_users = users_df.head(20).to_dict(orient='records')
    return sample_users

@app.get("/api/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, n: int = 10):
    """Get personalized recommendations for a user."""
    if user_id not in users_df['user_id'].values:
        raise HTTPException(status_code=404, detail="User not found")
        
    try:
        recs = recommender.recommend(user_id, n_recommendations=n)
        return {
            "user_id": user_id,
            "recommendations": recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/popular", response_model=List[Movie])
async def get_popular(n: int = 10):
    """Get popular movies."""
    pop_movies = recommender.pop_model.get_popular_movies(n_recommendations=n)
    results = []
    for mid, score in pop_movies:
        movie = movies_df[movies_df['movie_id'] == mid].iloc[0].to_dict()
        movie['hybrid_score'] = score
        results.append(movie)
    return results

@app.get("/api/popular/region/{region}", response_model=List[Movie])
async def get_regional_popular(region: str, n: int = 10):
    """Get popular movies in a specific region."""
    pop_movies = recommender.pop_model.get_regional_popular(
        region, users_df, interactions_df, n_recommendations=n
    )
    results = []
    for mid, score in pop_movies:
        movie = movies_df[movies_df['movie_id'] == mid].iloc[0].to_dict()
        movie['hybrid_score'] = score
        results.append(movie)
    return results

@app.get("/api/search", response_model=List[Movie])
async def search_movies(q: str, n: int = 10):
    """Search for movies by title or genre."""
    query = q.lower()
    results = movies_df[
        movies_df['title'].str.lower().str.contains(query) | 
        movies_df['genre'].str.lower().str.contains(query)
    ].head(n).to_dict(orient='records')
    return results

@app.get("/api/trending", response_model=List[Movie])
async def get_trending(n: int = 10):
    """Get trending movies."""
    trend_movies = recommender.pop_model.get_trending_movies(n_recommendations=n)
    results = []
    for mid, score in trend_movies:
        movie = movies_df[movies_df['movie_id'] == mid].iloc[0].to_dict()
        movie['hybrid_score'] = score
        results.append(movie)
    return results

@app.get("/api/movie/{movie_id}/similar", response_model=List[Movie])
async def get_similar(movie_id: int, n: int = 5):
    """Get similar movies."""
    try:
        similar = recommender.get_similar_movies(movie_id, n_similar=n)
        return similar
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics."""
    return preprocessor.get_data_stats()

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

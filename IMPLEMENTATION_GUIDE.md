# 🎬 Hybrid Recommender System - Complete Implementation Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Code Walkthrough](#code-walkthrough)
5. [Usage Examples](#usage-examples)
6. [Evaluation & Metrics](#evaluation--metrics)
7. [Deployment Guide](#deployment-guide)
8. [FAQ](#faq)

---

## 🎯 Project Overview

### What is This?
A **production-ready hybrid recommender system** for OTT streaming platforms that combines:
- **Collaborative Filtering** (user behavior patterns)
- **Content-Based Filtering** (movie metadata similarity)
- **Popularity-Based** (trending and popular content)

### Why Hybrid?
Single-approach recommenders have limitations:
- **CF alone:** Cold-start problems, data sparsity
- **CB alone:** Filter bubbles, limited diversity
- **Popularity alone:** No personalization

**Hybrid = Best of all worlds** ✨

### Key Features
✅ Handles cold-start for new users and items  
✅ Provides explainable recommendations  
✅ Scalable architecture  
✅ Comprehensive evaluation metrics  
✅ Production-ready code quality  

---

## 🏗️ System Architecture

### High-Level Flow
```
User Request
    ↓
Check if Cold-Start User?
    ├─ Yes → Popularity (70%) + Content (30%)
    └─ No → Full Hybrid (CF: 50%, CB: 30%, Pop: 20%)
    ↓
Generate Candidate Movies
    ↓
Score Each Candidate
    ├─ Collaborative Filtering Score
    ├─ Content-Based Score
    └─ Popularity Score
    ↓
Combine Scores (Weighted)
    ↓
Apply Diversity Filter
    ↓
Return Top-K Recommendations
```

### Component Architecture
```
┌─────────────────────────────────────────────┐
│         Hybrid Recommender Engine           │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐  ┌──────────────┐       │
│  │ Collaborative│  │ Content-Based│       │
│  │   Filtering  │  │   Filtering  │       │
│  │              │  │              │       │
│  │  • SVD       │  │  • TF-IDF    │       │
│  │  • Matrix    │  │  • Cosine    │       │
│  │    Factor.   │  │    Similarity│       │
│  │              │  │              │       │
│  │  w = 0.5     │  │  w = 0.3     │       │
│  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                │
│         └────────┬────────┘                │
│                  │                         │
│         ┌────────▼────────┐                │
│         │  Popularity     │                │
│         │  Based          │                │
│         │                 │                │
│         │  • Trending     │                │
│         │  • View Count   │                │
│         │  • Bayesian Avg │                │
│         │                 │                │
│         │  w = 0.2        │                │
│         └────────┬────────┘                │
│                  │                         │
│         ┌────────▼────────┐                │
│         │ Score Combiner  │                │
│         └────────┬────────┘                │
│                  │                         │
│         ┌────────▼────────┐                │
│         │ Top-K Selection │                │
│         └─────────────────┘                │
└─────────────────────────────────────────────┘
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM recommended

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML algorithms
- `scipy` - Scientific computing
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization

### Step 2: Verify Installation
```bash
python -c "import pandas, numpy, sklearn; print('✅ All dependencies installed!')"
```

### Step 3: Run Demo
```bash
python demo.py
```

---

## 📚 Code Walkthrough

### Module 1: Data Preprocessing (`data_preprocessing.py`)

**Purpose:** Prepare data for the recommender system

**Key Functions:**

```python
class DataPreprocessor:
    def create_sample_data():
        """
        Creates sample datasets:
        - 1000 users (age, gender, location)
        - 500 movies (genre, cast, director, description)
        - 50K interactions (ratings, watch time)
        """
    
    def engineer_features():
        """
        Creates additional features:
        - User: account_age_days
        - Movie: is_recent, genre_count, content (combined metadata)
        - Interaction: watch_completion_rate, implicit_rating
        """
    
    def get_popularity_metrics():
        """
        Calculates:
        - View counts
        - Average ratings
        - Bayesian ratings
        - Recency scores
        - Combined popularity score
        """
```

**Example Usage:**
```python
preprocessor = DataPreprocessor()
users, movies, interactions = preprocessor.create_sample_data()
preprocessor.engineer_features()
stats = preprocessor.get_data_stats()
```

---

### Module 2: Collaborative Filtering (`collaborative_filtering.py`)

**Purpose:** Recommend based on user behavior patterns

**Algorithm:** SVD (Singular Value Decomposition)

**Mathematical Foundation:**
```
User-Item Matrix R (m×n) ≈ U (m×k) × Σ (k×k) × V^T (k×n)

Where:
- m = number of users
- n = number of movies
- k = number of latent factors (default: 50)
```

**Key Functions:**

```python
class CollaborativeFiltering:
    def fit(interactions_df):
        """
        1. Create user-item matrix
        2. Normalize by user mean
        3. Apply SVD decomposition
        4. Reconstruct for predictions
        """
    
    def predict_rating(user_id, movie_id):
        """
        Predict rating for user-movie pair
        Returns: float (1-5 scale)
        """
    
    def get_user_recommendations(user_id, n=10):
        """
        Get top-N recommendations for user
        Returns: [(movie_id, predicted_rating), ...]
        """
```

**Example Usage:**
```python
cf = CollaborativeFiltering(n_factors=50)
cf.fit(interactions_df)
recommendations = cf.get_user_recommendations(user_id=1, n_recommendations=10)
```

---

### Module 3: Content-Based Filtering (`content_based_filtering.py`)

**Purpose:** Recommend similar content based on metadata

**Algorithm:** TF-IDF + Cosine Similarity

**How TF-IDF Works:**
```
TF-IDF assigns weights to words:
- High weight: Unique, important words (e.g., "Nolan", "Inception")
- Low weight: Common words (e.g., "movie", "film")

Example:
Movie: "Inception by Christopher Nolan"
TF-IDF vector: [0.0, 0.8, 0.9, 0.0, ...]
                 ↑    ↑    ↑
              common unique unique
```

**Key Functions:**

```python
class ContentBasedFiltering:
    def fit(movies_df):
        """
        1. Combine metadata into 'content' field
        2. Apply TF-IDF vectorization
        3. Compute similarity matrix
        """
    
    def get_similar_movies(movie_id, n=10):
        """
        Find movies similar to given movie
        Returns: [(movie_id, similarity_score), ...]
        """
    
    def get_user_recommendations(user_profile, n=10):
        """
        Recommend based on user's watch history
        user_profile: [movie_ids user has watched]
        """
```

**Example Usage:**
```python
cb = ContentBasedFiltering(max_features=1000)
cb.fit(movies_df)
similar = cb.get_similar_movies(movie_id=1, n_similar=5)
```

---

### Module 4: Popularity-Based (`popularity_based.py`)

**Purpose:** Recommend trending and popular content

**Metrics Calculated:**

1. **View Score:** Normalized view count
2. **Bayesian Rating:** Handles movies with few ratings
3. **Trending Score:** Recent popularity + growth rate
4. **Velocity:** Growth rate (recent_views / total_views)

**Key Functions:**

```python
class PopularityBasedFiltering:
    def fit(movies_df, interactions_df):
        """
        Calculate all popularity metrics
        """
    
    def get_popular_movies(n=10, min_ratings=10):
        """
        Get most popular movies (all-time)
        """
    
    def get_trending_movies(n=10):
        """
        Get trending movies (recent popularity)
        """
    
    def get_regional_popular(region, n=10):
        """
        Get popular movies in specific region
        """
```

**Example Usage:**
```python
pop = PopularityBasedFiltering(recency_days=30)
pop.fit(movies_df, interactions_df)
popular = pop.get_popular_movies(n_recommendations=10)
trending = pop.get_trending_movies(n_recommendations=10)
```

---

### Module 5: Hybrid Recommender (`hybrid_recommender.py`)

**Purpose:** Combine all approaches for optimal recommendations

**Hybrid Formula:**
```
Final_Score = 0.5 × CF_Score + 0.3 × CB_Score + 0.2 × Pop_Score
```

**Key Functions:**

```python
class HybridRecommender:
    def __init__(cf_weight=0.5, cb_weight=0.3, pop_weight=0.2):
        """
        Initialize with custom weights
        """
    
    def fit(users_df, movies_df, interactions_df):
        """
        Train all three component models
        """
    
    def recommend(user_id, n_recommendations=10):
        """
        Main recommendation function
        
        Flow:
        1. Check if cold-start user
        2. Get candidate movies
        3. Score with all three models
        4. Combine scores
        5. Apply diversity filter
        6. Return top-K
        """
    
    def explain_recommendation(user_id, movie_id):
        """
        Explain why movie is recommended
        Returns detailed breakdown of scores
        """
```

**Example Usage:**
```python
recommender = HybridRecommender(
    cf_weight=0.5,
    cb_weight=0.3,
    pop_weight=0.2
)
recommender.fit(users_df, movies_df, interactions_df)
recs = recommender.recommend(user_id=1, n_recommendations=10)
```

---

### Module 6: Evaluation (`evaluation.py`)

**Purpose:** Measure recommendation quality

**Metrics Implemented:**

1. **Precision@K:** Accuracy of top-K recommendations
2. **Recall@K:** Coverage of relevant items
3. **F1@K:** Harmonic mean of precision and recall
4. **MAP@K:** Mean Average Precision
5. **RMSE:** Rating prediction error
6. **MAE:** Mean absolute error
7. **Coverage:** % of catalog recommended
8. **Diversity:** Variety in recommendations

**Example Usage:**
```python
evaluator = RecommenderEvaluator()

# Split data
train_df, test_df = preprocessor.train_test_split(test_size=0.2)

# Train on training data
recommender.fit(users_df, movies_df, train_df)

# Evaluate on test data
results = evaluator.evaluate_recommender(
    recommender,
    test_df,
    movies_df,
    k=10
)

print(f"Precision@10: {results['precision@k']:.4f}")
print(f"Recall@10: {results['recall@k']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")
```

---

## 🎮 Usage Examples

### Example 1: Basic Recommendations

```python
from src.data_preprocessing import DataPreprocessor
from src.hybrid_recommender import HybridRecommender

# Prepare data
preprocessor = DataPreprocessor()
users, movies, interactions = preprocessor.create_sample_data()
preprocessor.engineer_features()

# Train recommender
recommender = HybridRecommender()
recommender.fit(
    preprocessor.users_df,
    preprocessor.movies_df,
    preprocessor.interactions_df
)

# Get recommendations
user_id = 1
recs = recommender.recommend(user_id, n_recommendations=10)

# Display
for i, rec in enumerate(recs, 1):
    print(f"{i}. {rec['title']}")
    print(f"   Score: {rec['hybrid_score']:.3f}")
    print(f"   Reason: {rec['explanation']}\n")
```

### Example 2: Similar Movies

```python
# Find movies similar to "Inception"
movie_id = 42  # Inception's ID

similar_movies = recommender.get_similar_movies(movie_id, n_similar=5)

print(f"Movies similar to {movies_df[movies_df['movie_id']==movie_id]['title'].values[0]}:")
for movie in similar_movies:
    print(f"- {movie['title']} (Similarity: {movie['similarity']:.3f})")
```

### Example 3: Trending Content

```python
# Get what's trending
trending = recommender.pop_model.get_trending_movies(n_recommendations=10)

print("🔥 Trending Now:")
for movie_id, score in trending:
    movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
    print(f"- {movie['title']} (Trending Score: {score:.3f})")
```

### Example 4: Explainable Recommendations

```python
# Explain why a movie is recommended
user_id = 1
movie_id = recs[0]['movie_id']

explanation = recommender.explain_recommendation(user_id, movie_id)

print(f"Why '{explanation['title']}' for User {user_id}:")
print(f"\nScores:")
for score_type, value in explanation['scores'].items():
    print(f"  {score_type}: {value:.3f}")

print(f"\nReasons:")
for reason in explanation['reasons']:
    print(f"  ✓ {reason}")
```

---

## 📊 Evaluation & Metrics

### Running Evaluation

```python
from src.evaluation import RecommenderEvaluator

# Split data
train_df, test_df = preprocessor.train_test_split(test_size=0.2)

# Train on training set
recommender.fit(users_df, movies_df, train_df)

# Evaluate on test set
evaluator = RecommenderEvaluator()
results = evaluator.evaluate_recommender(
    recommender,
    test_df,
    movies_df,
    k=10
)
```

### Understanding Metrics

**Precision@10 = 0.35**
- 35% of top-10 recommendations are relevant
- Good: > 0.3

**Recall@10 = 0.25**
- 25% of all relevant items are in top-10
- Good: > 0.2

**RMSE = 0.85**
- Average rating prediction error is 0.85 stars
- Good: < 1.0 on 1-5 scale

**Coverage = 0.35**
- 35% of catalog gets recommended
- Good: > 0.3

---

## 🚀 Deployment Guide

### Option 1: REST API with Flask

```python
from flask import Flask, jsonify, request
from src.hybrid_recommender import HybridRecommender

app = Flask(__name__)

# Load trained model (in production, load from disk)
recommender = HybridRecommender()
# recommender.load('model.pkl')

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    n = request.args.get('n', default=10, type=int)
    recs = recommender.recommend(user_id, n_recommendations=n)
    return jsonify(recs)

@app.route('/similar/<int:movie_id>', methods=['GET'])
def similar(movie_id):
    n = request.args.get('n', default=5, type=int)
    similar_movies = recommender.get_similar_movies(movie_id, n_similar=n)
    return jsonify(similar_movies)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Option 2: FastAPI (Modern, Async)

```python
from fastapi import FastAPI, Query
from typing import List

app = FastAPI()

@app.get("/recommend/{user_id}")
async def recommend(
    user_id: int,
    n: int = Query(10, ge=1, le=100)
):
    recs = recommender.recommend(user_id, n_recommendations=n)
    return {"user_id": user_id, "recommendations": recs}

@app.get("/trending")
async def trending(n: int = Query(10, ge=1, le=50)):
    trending_movies = recommender.pop_model.get_trending_movies(n)
    return {"trending": trending_movies}
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api.py"]
```

---

## ❓ FAQ

### Q: How do I use my own data?

**A:** Format your data as CSV files:

```python
preprocessor = DataPreprocessor()
preprocessor.load_data(
    users_path='my_users.csv',
    movies_path='my_movies.csv',
    interactions_path='my_interactions.csv'
)
preprocessor.engineer_features()
recommender.fit(preprocessor.users_df, preprocessor.movies_df, preprocessor.interactions_df)
```

### Q: How do I adjust the weights?

**A:** Pass custom weights when initializing:

```python
# Emphasize collaborative filtering
recommender = HybridRecommender(
    cf_weight=0.7,
    cb_weight=0.2,
    pop_weight=0.1
)
```

### Q: How do I handle cold-start users?

**A:** The system automatically detects cold-start users and adjusts strategy:

```python
# For users with < 5 interactions:
# - Uses popularity (70%) + content-based (30%)
# - No collaborative filtering

# You can customize the threshold:
recommender.min_interactions_for_cf = 10  # Require 10 interactions
```

### Q: How do I improve performance?

**A:**
1. **Reduce factors:** `CollaborativeFiltering(n_factors=20)`
2. **Limit features:** `ContentBasedFiltering(max_features=500)`
3. **Cache results:** Use Redis for frequently accessed recommendations
4. **Batch processing:** Pre-compute recommendations for all users

### Q: Can I add more features?

**A:** Yes! Modify `data_preprocessing.py`:

```python
def engineer_features(self):
    # Add custom features
    self.movies_df['is_blockbuster'] = self.movies_df['budget'] > 100_000_000
    self.users_df['is_premium'] = self.users_df['subscription_type'] == 'premium'
    # ... existing code
```

---

## 🎓 Learning Path

### Beginner
1. Run `demo.py` to see the system in action
2. Open Jupyter notebook and run all cells
3. Modify weights and see how recommendations change
4. Try with your own small dataset

### Intermediate
1. Understand each module's code
2. Implement custom evaluation metrics
3. Add new features to the data
4. Experiment with different algorithms

### Advanced
1. Implement deep learning models (NCF, BERT embeddings)
2. Add real-time updates with Kafka
3. Deploy to cloud (AWS/GCP/Azure)
4. Implement A/B testing framework
5. Scale with Apache Spark

---

## 📞 Support

- **Documentation:** Check `docs/` folder
- **Examples:** See `notebooks/` folder
- **Issues:** Open GitHub issue
- **Questions:** Start a discussion

---

**Happy Recommending! 🎬**

*Built with ❤️ for aspiring ML engineers*

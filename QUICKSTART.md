# 🚀 Quick Start Guide

Get up and running with the OTT Hybrid Recommender System in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Jupyter Notebook or Google Colab

## Installation

### 1. Clone or Download the Project

```bash
cd ott-recommender-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas
- numpy
- scikit-learn
- scikit-surprise
- matplotlib
- seaborn

## Quick Demo

### Option 1: Run the Demo Script (Fastest)

```bash
python demo.py
```

This will:
- ✅ Create sample data
- ✅ Train the hybrid recommender
- ✅ Generate recommendations
- ✅ Show evaluation metrics
- ✅ Display popular and trending content

**Expected Output:**
```
================================================================================
  🎬 OTT HYBRID RECOMMENDER SYSTEM DEMO
================================================================================

────────────────────────────────────────────────────────────────────────────────
  STEP 1: Data Preparation
────────────────────────────────────────────────────────────────────────────────

📂 Creating sample datasets...
✅ Created 1000 users
✅ Created 500 movies
✅ Created 49XXX interactions

...
```

### Option 2: Use Jupyter Notebook (Recommended for Learning)

1. **Open the notebook:**
   ```bash
   jupyter notebook notebooks/OTT_Recommender_System.ipynb
   ```

2. **Or use Google Colab:**
   - Upload `OTT_Recommender_System.ipynb` to Google Colab
   - Run all cells

### Option 3: Use Python Interactively

```python
# Import modules
from src.data_preprocessing import DataPreprocessor
from src.hybrid_recommender import HybridRecommender

# Create sample data
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
recommendations = recommender.recommend(user_id=1, n_recommendations=10)

# Display
for rec in recommendations:
    print(f"{rec['title']} - Score: {rec['hybrid_score']:.3f}")
```

## Basic Usage Examples

### 1. Get Personalized Recommendations

```python
from src.hybrid_recommender import HybridRecommender

# Initialize and train (assuming data is loaded)
recommender = HybridRecommender(
    cf_weight=0.5,    # Collaborative filtering weight
    cb_weight=0.3,    # Content-based weight
    pop_weight=0.2    # Popularity weight
)

recommender.fit(users_df, movies_df, interactions_df)

# Get recommendations for a user
user_id = 123
recommendations = recommender.recommend(
    user_id=user_id,
    n_recommendations=10,
    exclude_watched=True
)

# Print recommendations
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['title']}")
    print(f"   Genre: {rec['genre']}")
    print(f"   Score: {rec['hybrid_score']:.3f}")
    print(f"   Why: {rec['explanation']}\n")
```

### 2. Find Similar Movies

```python
# Find movies similar to a specific movie
movie_id = 42
similar_movies = recommender.get_similar_movies(
    movie_id=movie_id,
    n_similar=5
)

for movie in similar_movies:
    print(f"{movie['title']} - Similarity: {movie['similarity']:.3f}")
```

### 3. Get Popular/Trending Content

```python
# Get popular movies
popular = recommender.pop_model.get_popular_movies(n_recommendations=10)

for movie_id, score in popular:
    print(f"Movie {movie_id}: Popularity = {score:.3f}")

# Get trending movies
trending = recommender.pop_model.get_trending_movies(n_recommendations=10)

for movie_id, score in trending:
    print(f"Movie {movie_id}: Trending = {score:.3f}")
```

### 4. Explain a Recommendation

```python
# Get detailed explanation for why a movie is recommended
explanation = recommender.explain_recommendation(
    user_id=123,
    movie_id=456
)

print(f"Movie: {explanation['title']}")
print(f"Hybrid Score: {explanation['scores']['hybrid']:.3f}")
print("\nReasons:")
for reason in explanation['reasons']:
    print(f"  • {reason}")
```

### 5. Evaluate the Model

```python
from src.evaluation import RecommenderEvaluator

# Split data
train_df, test_df = preprocessor.train_test_split(test_size=0.2)

# Train on training data
recommender.fit(users_df, movies_df, train_df)

# Evaluate on test data
evaluator = RecommenderEvaluator()
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

## Using Your Own Data

### Data Format

Your data should be in CSV format with the following structure:

**users.csv:**
```csv
user_id,age,gender,location,signup_date
1,25,M,US,2023-01-15
2,32,F,UK,2023-02-20
...
```

**movies.csv:**
```csv
movie_id,title,genre,cast,director,description,language,release_year,duration
1,Movie Title,Action|Drama,Actor1|Actor2,Director1,Description here,English,2023,120
...
```

**interactions.csv:**
```csv
user_id,movie_id,rating,watch_time,timestamp,completed
1,5,5,120,2023-06-01 10:30:00,1
1,10,4,90,2023-06-02 14:15:00,1
...
```

### Load Your Data

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Load your data
preprocessor.load_data(
    users_path='path/to/users.csv',
    movies_path='path/to/movies.csv',
    interactions_path='path/to/interactions.csv'
)

# Engineer features
preprocessor.engineer_features()

# Use the data
recommender = HybridRecommender()
recommender.fit(
    preprocessor.users_df,
    preprocessor.movies_df,
    preprocessor.interactions_df
)
```

## Customization

### Adjust Hybrid Weights

```python
# Emphasize collaborative filtering more
recommender = HybridRecommender(
    cf_weight=0.7,
    cb_weight=0.2,
    pop_weight=0.1
)

# For cold-start heavy scenarios, emphasize popularity
recommender = HybridRecommender(
    cf_weight=0.3,
    cb_weight=0.3,
    pop_weight=0.4
)
```

### Tune Model Parameters

```python
recommender = HybridRecommender(
    cf_weight=0.5,
    cb_weight=0.3,
    pop_weight=0.2,
    n_factors=100,        # More latent factors for CF
    max_features=2000     # More features for content-based
)
```

### Add Diversity

```python
recommendations = recommender.recommend(
    user_id=123,
    n_recommendations=10,
    diversity_factor=0.3  # 0-1, higher = more diverse
)
```

## Common Issues & Solutions

### Issue: "Module not found" error

**Solution:**
```bash
# Make sure you're in the project directory
cd ott-recommender-system

# Install dependencies
pip install -r requirements.txt

# If using custom modules, add to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: Memory error with large datasets

**Solution:**
```python
# Reduce the number of factors
recommender = HybridRecommender(n_factors=20)

# Or sample your data
interactions_sample = interactions_df.sample(frac=0.5)
```

### Issue: Slow recommendations

**Solution:**
```python
# Pre-compute recommendations for all users (batch)
all_recs = {}
for user_id in users_df['user_id']:
    all_recs[user_id] = recommender.recommend(user_id, n_recommendations=20)

# Cache in Redis or similar
```

## Next Steps

1. **Explore the Jupyter Notebook** - Detailed walkthrough with visualizations
2. **Read the Architecture Docs** - `docs/architecture.md`
3. **Customize for Your Use Case** - Modify weights, add features
4. **Deploy as API** - Create Flask/FastAPI endpoints
5. **Scale Up** - Use Apache Spark for big data

## Getting Help

- 📖 Read the full documentation in `docs/`
- 💻 Check example code in `notebooks/`
- 🐛 Report issues on GitHub
- 💬 Ask questions in discussions

## Performance Benchmarks

On sample data (1000 users, 500 movies, 50K interactions):

- **Training Time:** ~30 seconds
- **Recommendation Time:** ~50ms per user
- **Memory Usage:** ~500MB

## What's Next?

- [ ] Add deep learning models (NCF, BERT embeddings)
- [ ] Implement real-time updates with Kafka
- [ ] Build REST API with FastAPI
- [ ] Add A/B testing framework
- [ ] Deploy to cloud (AWS/GCP/Azure)

---

**Ready to build the next Netflix?** 🚀

Start with `python demo.py` and explore from there!

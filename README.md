# 🎬 Hybrid Recommender System for OTT Streaming Platform

A production-ready, scalable hybrid recommendation engine combining Collaborative Filtering, Content-Based Filtering, and Popularity-Based approaches for Netflix-like streaming platforms.

## 🎯 Project Overview

This system implements a sophisticated hybrid recommender that addresses the key challenges of modern OTT platforms:
- **Cold-start problem** for new users and content
- **Personalization** based on viewing history
- **Content discovery** through similarity matching
- **Trending content** promotion

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER REQUEST (User ID)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HYBRID RECOMMENDATION ENGINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Collaborative   │  │  Content-Based   │  │  Popularity  │ │
│  │    Filtering     │  │    Filtering     │  │    Based     │ │
│  │                  │  │                  │  │              │ │
│  │  • SVD/Matrix    │  │  • TF-IDF on     │  │  • View      │ │
│  │    Factorization │  │    metadata      │  │    count     │ │
│  │  • User-Item     │  │  • Cosine        │  │  • Trending  │ │
│  │    interactions  │  │    Similarity    │  │    score     │ │
│  │  • Rating        │  │  • Genre, Cast   │  │  • Recency   │ │
│  │    prediction    │  │    Description   │  │    boost     │ │
│  │                  │  │                  │  │              │ │
│  │  Weight: 0.5     │  │  Weight: 0.3     │  │  Weight: 0.2 │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘ │
│           │                     │                    │          │
│           └─────────────────────┼────────────────────┘          │
│                                 ▼                                │
│                    ┌────────────────────────┐                   │
│                    │   SCORE AGGREGATION    │                   │
│                    │  Weighted Combination  │                   │
│                    └────────────┬───────────┘                   │
│                                 │                                │
│                                 ▼                                │
│                    ┌────────────────────────┐                   │
│                    │   RE-RANKING & FILTER  │                   │
│                    │  • Diversity           │                   │
│                    │  • Already watched     │                   │
│                    └────────────┬───────────┘                   │
└─────────────────────────────────┼────────────────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────┐
                    │  TOP-K RECOMMENDATIONS │
                    └────────────────────────┘
```

## 🤔 Why Hybrid Recommender for OTT Platforms?

### 1. **Overcomes Individual Limitations**
- **Collaborative Filtering alone**: Suffers from cold-start (new users/items), sparsity
- **Content-Based alone**: Limited diversity, over-specialization (filter bubble)
- **Popularity-Based alone**: No personalization, same for everyone

### 2. **Handles Cold-Start Problem**
- **New Users**: Rely on popularity + content-based (ask for preferences)
- **New Content**: Use content similarity + initial popularity metrics
- **Established Users**: Leverage collaborative filtering for personalization

### 3. **Balances Exploration vs Exploitation**
- Collaborative: Exploits user behavior patterns
- Content-Based: Explores similar content
- Popularity: Introduces trending/viral content

### 4. **Business Benefits**
- Increased engagement and watch time
- Better content discovery
- Reduced churn rate
- Higher user satisfaction

## 📊 Dataset Structure

### Users Table
```
user_id | age | gender | location | signup_date
```

### Movies/Content Table
```
movie_id | title | genre | cast | director | description | language | release_year | duration
```

### Interactions/Ratings Table
```
user_id | movie_id | rating | watch_time | timestamp | completed
```

## 🧮 Hybrid Scoring Formula

```
Final_Score = (w1 × CF_Score) + (w2 × CB_Score) + (w3 × Pop_Score)

where:
- w1 = 0.5 (Collaborative Filtering weight)
- w2 = 0.3 (Content-Based weight)
- w3 = 0.2 (Popularity weight)
- w1 + w2 + w3 = 1.0
```

## 📁 Project Structure

```
ott-recommender-system/
├── README.md
├── requirements.txt
├── data/
│   ├── sample_users.csv
│   ├── sample_movies.csv
│   └── sample_interactions.csv
├── src/
│   ├── data_preprocessing.py
│   ├── collaborative_filtering.py
│   ├── content_based_filtering.py
│   ├── popularity_based.py
│   ├── hybrid_recommender.py
│   └── evaluation.py
├── notebooks/
│   └── OTT_Recommender_System.ipynb
└── docs/
    └── architecture.md
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run in Google Colab
1. Upload the notebook `OTT_Recommender_System.ipynb`
2. Run all cells
3. Get recommendations!

### Usage
```python
from src.hybrid_recommender import HybridRecommender

# Initialize
recommender = HybridRecommender()

# Train
recommender.fit(users, movies, interactions)

# Get recommendations
recommendations = recommender.recommend(user_id=123, top_k=10)
```

## 📈 Evaluation Metrics

- **Precision@K**: Relevance of top-K recommendations
- **Recall@K**: Coverage of relevant items in top-K
- **RMSE**: Rating prediction accuracy
- **Coverage**: Percentage of catalog recommended
- **Diversity**: Variety in recommendations

## 🔮 Real-World Improvements

### 1. **Deep Learning Embeddings**
- Use Neural Collaborative Filtering (NCF)
- BERT embeddings for content descriptions
- Multi-modal embeddings (thumbnails, trailers)

### 2. **Real-Time Recommendations**
- Stream processing with Apache Kafka
- Redis for caching
- Incremental model updates

### 3. **Explainable AI**
- "Because you watched X"
- "Popular in your area"
- "Trending now"
- Genre/actor-based explanations

### 4. **Advanced Features**
- Context-aware recommendations (time, device, mood)
- Multi-armed bandit for A/B testing
- Session-based recommendations
- Social recommendations (friends watching)

### 5. **Scalability**
- Distributed computing (Apache Spark)
- Model serving with TensorFlow Serving
- Approximate Nearest Neighbors (ANN) for similarity search
- Batch + real-time hybrid architecture

## 📚 Technologies Used

- **Python 3.8+**
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: TF-IDF, Cosine Similarity, SVD
- **Surprise**: Collaborative filtering algorithms
- **Matplotlib & Seaborn**: Visualization

## 🎓 Suitable For

- ✅ College projects and assignments
- ✅ Hackathons and competitions
- ✅ Startup MVP development
- ✅ Learning recommendation systems
- ✅ Portfolio projects

## 📝 License

MIT License - Free to use for educational and commercial purposes

## 👥 Contributing

Contributions welcome! Please read CONTRIBUTING.md for details.

## 📧 Contact

For questions and support, please open an issue.

---

**Built with ❤️ for the next generation of OTT platforms**

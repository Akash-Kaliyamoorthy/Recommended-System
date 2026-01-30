# 🏗️ System Architecture Documentation

## Overview

This document provides a detailed explanation of the hybrid recommender system architecture for OTT streaming platforms.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Homepage   │  │  Watch Page  │  │  Search Page │  │  User Profile│ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
└─────────┼──────────────────┼──────────────────┼──────────────────┼────────┘
          │                  │                  │                  │
          └──────────────────┴──────────────────┴──────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API GATEWAY LAYER                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  REST API (Flask/FastAPI)                                         │  │
│  │  - /recommend/{user_id}                                           │  │
│  │  - /similar/{movie_id}                                            │  │
│  │  - /trending                                                      │  │
│  │  - /popular                                                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION ENGINE LAYER                         │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                  HYBRID RECOMMENDER                             │   │
│  │                                                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │   │
│  │  │ Collaborative│  │ Content-Based│  │  Popularity-Based  │   │   │
│  │  │  Filtering   │  │  Filtering   │  │                    │   │   │
│  │  │              │  │              │  │                    │   │   │
│  │  │  • SVD       │  │  • TF-IDF    │  │  • View Count     │   │   │
│  │  │  • Matrix    │  │  • Cosine    │  │  • Trending       │   │   │
│  │  │    Factor.   │  │    Similarity│  │  • Bayesian Avg   │   │   │
│  │  │              │  │              │  │                    │   │   │
│  │  │  w = 0.5     │  │  w = 0.3     │  │  w = 0.2          │   │   │
│  │  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘   │   │
│  │         │                 │                     │              │   │
│  │         └─────────────────┴─────────────────────┘              │   │
│  │                           │                                     │   │
│  │                           ▼                                     │   │
│  │              ┌────────────────────────┐                        │   │
│  │              │  Score Aggregation     │                        │   │
│  │              │  & Re-ranking          │                        │   │
│  │              └────────────┬───────────┘                        │   │
│  │                           │                                     │   │
│  │                           ▼                                     │   │
│  │              ┌────────────────────────┐                        │   │
│  │              │  Diversity Filter      │                        │   │
│  │              │  & Deduplication       │                        │   │
│  │              └────────────┬───────────┘                        │   │
│  │                           │                                     │   │
│  │                           ▼                                     │   │
│  │              ┌────────────────────────┐                        │   │
│  │              │  Top-K Recommendations │                        │   │
│  │              └────────────────────────┘                        │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CACHING LAYER                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Redis Cache                                                      │  │
│  │  - User recommendations (TTL: 1 hour)                            │  │
│  │  - Popular movies (TTL: 15 minutes)                              │  │
│  │  - Trending movies (TTL: 5 minutes)                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Users DB   │  │  Movies DB   │  │ Interactions │  │  Features  │ │
│  │  (PostgreSQL)│  │ (PostgreSQL) │  │      DB      │  │     DB     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     REAL-TIME PROCESSING LAYER                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Apache Kafka                                                     │  │
│  │  - User interaction events                                       │  │
│  │  - Model update triggers                                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Stream Processing (Apache Flink/Spark Streaming)                │  │
│  │  - Real-time feature computation                                 │  │
│  │  - Incremental model updates                                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Collaborative Filtering Module

**Algorithm:** Singular Value Decomposition (SVD)

**Input:**
- User-Item interaction matrix (ratings)

**Process:**
1. Create sparse user-item matrix
2. Normalize by user mean ratings
3. Apply SVD: R ≈ U × Σ × V^T
4. Reconstruct matrix for predictions

**Output:**
- Predicted ratings for unseen items
- User similarity scores

**Advantages:**
- Discovers latent patterns
- Handles implicit feedback
- Scalable with sparse matrices

**Limitations:**
- Cold-start for new users/items
- Requires sufficient interaction data

---

### 2. Content-Based Filtering Module

**Algorithm:** TF-IDF + Cosine Similarity

**Input:**
- Movie metadata (genre, cast, director, description)

**Process:**
1. Combine metadata into content string
2. Apply TF-IDF vectorization
3. Compute cosine similarity matrix
4. Recommend similar items to user's history

**Output:**
- Similar movies based on content
- Content-based scores

**Advantages:**
- Works for new items immediately
- Explainable recommendations
- No cold-start for items

**Limitations:**
- Limited diversity (filter bubble)
- Requires good metadata quality

---

### 3. Popularity-Based Module

**Metrics Calculated:**

**View Score:**
```
view_score = (view_count - min_views) / (max_views - min_views)
```

**Bayesian Average Rating:**
```
bayesian_rating = (C × m + rating_count × avg_rating) / (C + rating_count)
```
Where:
- C = average number of ratings across all movies
- m = mean rating across all movies

**Trending Score:**
```
trending_score = 0.4 × recent_view_score + 
                 0.3 × recent_rating_score + 
                 0.3 × velocity_score
```

**Velocity (Growth Rate):**
```
velocity = recent_views / total_views
```

**Output:**
- Popular movies (all-time)
- Trending movies (recent)
- Regional popular movies

---

### 4. Hybrid Scoring Formula

**Final Score Calculation:**

```
hybrid_score = w_cf × normalize(CF_score) + 
               w_cb × normalize(CB_score) + 
               w_pop × normalize(Pop_score)
```

**Default Weights:**
- w_cf = 0.5 (Collaborative Filtering)
- w_cb = 0.3 (Content-Based)
- w_pop = 0.2 (Popularity)

**Normalization:**
- CF scores: (rating - 1) / 4  [1-5 scale → 0-1]
- CB scores: Already 0-1 (cosine similarity)
- Pop scores: Already 0-1 (normalized)

---

## Data Flow

### Recommendation Request Flow

```
1. User Request
   ↓
2. Check Cache (Redis)
   ├─ Hit → Return cached recommendations
   └─ Miss → Continue
   ↓
3. Load User Profile
   - Interaction history
   - Preferences
   - Demographics
   ↓
4. Check Cold-Start Status
   ├─ Cold-Start User → Use Popularity + Content-Based
   └─ Established User → Use Full Hybrid
   ↓
5. Generate Candidate Set
   - Get unwatched movies
   - Apply filters (language, region, etc.)
   ↓
6. Score Candidates
   ├─ Collaborative Filtering → CF scores
   ├─ Content-Based → CB scores
   └─ Popularity → Pop scores
   ↓
7. Combine Scores
   - Apply weights
   - Calculate hybrid scores
   ↓
8. Re-rank
   - Apply diversity filter
   - Remove duplicates
   - Apply business rules
   ↓
9. Select Top-K
   - Sort by hybrid score
   - Take top N recommendations
   ↓
10. Cache Results
    - Store in Redis (TTL: 1 hour)
    ↓
11. Return Recommendations
    - Include explanations
    - Add metadata
```

---

## Cold-Start Handling Strategy

### New User (No Interaction History)

**Strategy:**
1. Show popular/trending content (100%)
2. Ask for genre preferences (onboarding)
3. Use content-based on initial selections
4. Gradually introduce collaborative filtering

**Weight Adjustment:**
- CF: 0.0 → 0.2 → 0.4 → 0.5 (as interactions increase)
- CB: 0.3 → 0.4 → 0.3 → 0.3
- Pop: 0.7 → 0.4 → 0.3 → 0.2

### New Item (No Interaction Data)

**Strategy:**
1. Use content-based similarity to existing items
2. Show to users with matching preferences
3. Track initial engagement metrics
4. Gradually incorporate into collaborative filtering

---

## Scalability Considerations

### Horizontal Scaling

**Model Serving:**
- Deploy multiple instances behind load balancer
- Use model versioning for A/B testing
- Implement circuit breakers for fault tolerance

**Database Sharding:**
- Shard users by user_id % N
- Shard movies by movie_id % M
- Use read replicas for queries

### Caching Strategy

**Multi-Level Cache:**
1. **L1 - Application Cache:** In-memory LRU cache
2. **L2 - Redis:** Distributed cache for recommendations
3. **L3 - CDN:** Static content and popular recommendations

**Cache Invalidation:**
- Time-based: TTL for different content types
- Event-based: Invalidate on user interaction
- Predictive: Pre-compute during off-peak hours

### Batch vs Real-Time

**Batch Processing (Daily):**
- Train collaborative filtering models
- Compute content similarity matrices
- Update popularity metrics
- Generate recommendations for all users

**Real-Time Processing:**
- Update user profiles on interaction
- Adjust trending scores
- Personalize based on session context
- A/B test experiments

---

## Evaluation Metrics

### Ranking Metrics

**Precision@K:**
```
Precision@K = |{recommended items @K} ∩ {relevant items}| / K
```

**Recall@K:**
```
Recall@K = |{recommended items @K} ∩ {relevant items}| / |{relevant items}|
```

**MAP@K (Mean Average Precision):**
```
MAP@K = (1/|U|) × Σ(AP@K for each user)
```

### Rating Prediction Metrics

**RMSE (Root Mean Squared Error):**
```
RMSE = √[(1/N) × Σ(predicted_rating - actual_rating)²]
```

**MAE (Mean Absolute Error):**
```
MAE = (1/N) × Σ|predicted_rating - actual_rating|
```

### Catalog Metrics

**Coverage:**
```
Coverage = |{unique items recommended}| / |{total items in catalog}|
```

**Diversity:**
```
Diversity = Average pairwise Jaccard distance between recommended items
```

---

## Production Deployment

### Infrastructure Requirements

**Compute:**
- 4-8 CPU cores per instance
- 16-32 GB RAM
- GPU optional (for deep learning models)

**Storage:**
- PostgreSQL: 100GB+ for data
- Redis: 16GB+ for cache
- S3/Object Storage: Model artifacts

**Network:**
- Load balancer (HAProxy/Nginx)
- API Gateway (Kong/AWS API Gateway)
- CDN (CloudFlare/AWS CloudFront)

### Monitoring & Logging

**Metrics to Track:**
- Request latency (p50, p95, p99)
- Cache hit rate
- Model prediction time
- Recommendation diversity
- User engagement (CTR, watch time)

**Logging:**
- User interactions
- API requests/responses
- Model predictions
- Errors and exceptions

**Alerting:**
- High latency (>500ms)
- Low cache hit rate (<70%)
- Model prediction errors
- System resource usage

---

## Future Enhancements

1. **Deep Learning Models**
   - Neural Collaborative Filtering
   - BERT embeddings for content
   - Multi-modal learning (text + images + video)

2. **Context-Aware Recommendations**
   - Time of day
   - Device type
   - Location
   - Mood detection

3. **Social Features**
   - Friend recommendations
   - Watch parties
   - Social proof signals

4. **Advanced Personalization**
   - Multi-objective optimization
   - Reinforcement learning
   - Bandits for exploration

5. **Explainability**
   - LIME/SHAP for model interpretation
   - Natural language explanations
   - Visual explanations

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Maintained By:** Development Team

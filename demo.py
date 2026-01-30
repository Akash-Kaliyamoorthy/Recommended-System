"""
Demo Script for OTT Hybrid Recommender System

This script demonstrates the complete workflow of the recommender system.
Run this to see the system in action!
"""

import sys
sys.path.append('./src')

from data_preprocessing import DataPreprocessor
from hybrid_recommender import HybridRecommender
from evaluation import RecommenderEvaluator
import pandas as pd


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_subheader(text):
    """Print formatted subheader."""
    print(f"\n{'-' * 80}")
    print(f"  {text}")
    print(f"{'-' * 80}\n")


def main():
    print_header("*** OTT HYBRID RECOMMENDER SYSTEM DEMO ***")
    
    # Step 1: Data Preparation
    print_subheader("STEP 1: Data Preparation")
    
    preprocessor = DataPreprocessor()
    users_df, movies_df, interactions_df = preprocessor.create_sample_data()
    preprocessor.engineer_features()
    
    # Update references
    users_df = preprocessor.users_df
    movies_df = preprocessor.movies_df
    interactions_df = preprocessor.interactions_df
    
    # Get statistics
    stats = preprocessor.get_data_stats()
    print("[*] Dataset Statistics:")
    print(f"  * Users: {stats['n_users']}")
    print(f"  * Movies: {stats['n_movies']}")
    print(f"  * Interactions: {stats['n_interactions']}")
    print(f"  * Sparsity: {stats['sparsity']:.2%}")
    print(f"  * Avg ratings per user: {stats['avg_ratings_per_user']:.1f}")
    print(f"  * Avg ratings per movie: {stats['avg_ratings_per_movie']:.1f}")
    
    # Step 2: Train-Test Split
    print_subheader("STEP 2: Train-Test Split")
    
    train_df, test_df = preprocessor.train_test_split(test_size=0.2)
    print(f"[+] Training set: {len(train_df)} interactions")
    print(f"[+] Test set: {len(test_df)} interactions")
    
    # Step 3: Train Hybrid Recommender
    print_subheader("STEP 3: Training Hybrid Recommender System")
    
    recommender = HybridRecommender(
        cf_weight=0.5,
        cb_weight=0.3,
        pop_weight=0.2,
        n_factors=50,
        max_features=1000
    )
    
    recommender.fit(users_df, movies_df, train_df)
    
    # Step 4: Generate Recommendations
    print_subheader("STEP 4: Generating Recommendations")
    
    # Example 1: Established user
    user_id = 1
    print(f"[>] Recommendations for User {user_id} (Established User):\n")
    
    recommendations = recommender.recommend(user_id, n_recommendations=5)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Genre: {rec['genre']}")
        print(f"   Year: {rec['release_year']}")
        print(f"   Hybrid Score: {rec['hybrid_score']:.3f}")
        print(f"   Explanation: {rec['explanation']}")
        print()
    
    # Example 2: Cold-start user
    user_interaction_counts = train_df.groupby('user_id').size()
    cold_start_users = user_interaction_counts[user_interaction_counts < 3]
    
    if len(cold_start_users) > 0:
        cold_user_id = cold_start_users.index[0]
        print(f"\n[!] Recommendations for User {cold_user_id} (Cold-Start User):\n")
        
        cold_recs = recommender.recommend(cold_user_id, n_recommendations=5)
        
        for i, rec in enumerate(cold_recs, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Genre: {rec['genre']}")
            print(f"   Hybrid Score: {rec['hybrid_score']:.3f}")
            print(f"   Explanation: {rec['explanation']}")
            print()
    
    # Step 5: Similar Movies
    print_subheader("STEP 5: Finding Similar Movies")
    
    movie_id = 1
    movie_title = movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]
    movie_genre = movies_df[movies_df['movie_id'] == movie_id]['genre'].values[0]
    
    print(f"[*] Movies Similar to: {movie_title}")
    print(f"   Genre: {movie_genre}\n")
    
    similar_movies = recommender.get_similar_movies(movie_id, n_similar=5)
    
    for i, sim_movie in enumerate(similar_movies, 1):
        print(f"{i}. {sim_movie['title']}")
        print(f"   Genre: {sim_movie['genre']}")
        print(f"   Similarity: {sim_movie['similarity']:.3f}")
        print()
    
    # Step 6: Explainable Recommendation
    print_subheader("STEP 6: Explainable Recommendation")
    
    user_id = 1
    movie_id = recommendations[0]['movie_id']
    
    explanation = recommender.explain_recommendation(user_id, movie_id)
    
    print(f"[?] Why we recommend '{explanation['title']}' to User {user_id}:\n")
    print(f"Genre: {explanation['genre']}\n")
    
    print("Scores:")
    for score_type, score_value in explanation['scores'].items():
        print(f"  * {score_type}: {score_value:.3f}")
    
    print("\nReasons:")
    for reason in explanation['reasons']:
        print(f"  [x] {reason}")
    
    # Step 7: Evaluation
    print_subheader("STEP 7: Model Evaluation")
    
    evaluator = RecommenderEvaluator()
    results = evaluator.evaluate_recommender(recommender, test_df, movies_df, k=10)
    
    # Step 8: Popular and Trending
    print_subheader("STEP 8: Popular & Trending Content")
    
    popular = recommender.pop_model.get_popular_movies(n_recommendations=5)
    print("[#] Top 5 Popular Movies:\n")
    
    for i, (movie_id, score) in enumerate(popular, 1):
        movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        print(f"{i}. {movie['title']}")
        print(f"   Genre: {movie['genre']}")
        print(f"   Popularity Score: {score:.3f}")
        print()
    
    trending = recommender.pop_model.get_trending_movies(n_recommendations=5)
    print("\n[^] Top 5 Trending Movies:\n")
    
    for i, (movie_id, score) in enumerate(trending, 1):
        movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        print(f"{i}. {movie['title']}")
        print(f"   Genre: {movie['genre']}")
        print(f"   Trending Score: {score:.3f}")
        print()
    
    # Summary
    print_header("[OK] DEMO COMPLETED SUCCESSFULLY!")
    
    print("[+] What we demonstrated:")
    print("  [x] Data preparation and feature engineering")
    print("  [x] Hybrid recommender system training")
    print("  [x] Personalized recommendations for users")
    print("  [x] Cold-start user handling")
    print("  [x] Similar movie discovery")
    print("  [x] Explainable AI recommendations")
    print("  [x] Model evaluation with multiple metrics")
    print("  [x] Popular and trending content")
    
    print("\n[>>] Next Steps:")
    print("  1. Explore the Jupyter notebook for detailed analysis")
    print("  2. Modify weights and parameters to experiment")
    print("  3. Add your own dataset")
    print("  4. Deploy as a REST API")
    print("  5. Implement real-time recommendations")
    
    print("\n" + "=" * 80)
    print("  Thank you for using the OTT Hybrid Recommender System!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
Hybrid Recommender System

Combines Collaborative Filtering, Content-Based Filtering, and Popularity-Based approaches
to provide robust, personalized recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering
from popularity_based import PopularityBasedFiltering


class HybridRecommender:
    """
    Hybrid Recommendation System combining multiple approaches.
    
    Weights:
    - Collaborative Filtering: 0.5 (personalization based on user behavior)
    - Content-Based Filtering: 0.3 (similarity to user preferences)
    - Popularity-Based: 0.2 (trending and popular content)
    
    Handles cold-start problems:
    - New users: Rely more on popularity and content-based (after genre preference)
    - New items: Use content-based and initial popularity
    """
    
    def __init__(self, 
                 cf_weight: float = 0.5,
                 cb_weight: float = 0.3,
                 pop_weight: float = 0.2,
                 n_factors: int = 50,
                 max_features: int = 5000):
        """
        Initialize hybrid recommender.
        
        Args:
            cf_weight: Weight for collaborative filtering
            cb_weight: Weight for content-based filtering
            pop_weight: Weight for popularity-based filtering
            n_factors: Number of latent factors for CF
            max_features: Max features for content-based TF-IDF
        """
        # Validate weights
        total_weight = cf_weight + cb_weight + pop_weight
        assert abs(total_weight - 1.0) < 0.001, "Weights must sum to 1.0"
        
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.pop_weight = pop_weight
        
        # Initialize individual models
        self.cf_model = CollaborativeFiltering(n_factors=n_factors)
        self.cb_model = ContentBasedFiltering(max_features=max_features)
        self.pop_model = PopularityBasedFiltering()
        
        # Data storage
        self.users_df = None
        self.movies_df = None
        self.interactions_df = None
        
        # User profiles (for content-based)
        self.user_profiles = {}
        
        # Cold-start thresholds
        self.min_interactions_for_cf = 5  # Minimum interactions to use CF
        
    def fit(self, users_df: pd.DataFrame, movies_df: pd.DataFrame, 
            interactions_df: pd.DataFrame) -> None:
        """
        Train all component models.
        
        Args:
            users_df: DataFrame with user information
            movies_df: DataFrame with movie metadata
            interactions_df: DataFrame with user-movie interactions
        """
        print("=" * 70)
        print("[*] TRAINING HYBRID RECOMMENDER SYSTEM")
        print("=" * 70)
        
        self.users_df = users_df
        self.movies_df = movies_df
        self.interactions_df = interactions_df
        
        # Build user profiles (movies they've watched/liked)
        self._build_user_profiles()
        
        # Train collaborative filtering
        print("\n[1/3] Collaborative Filtering")
        print("-" * 70)
        self.cf_model.fit(interactions_df)
        
        # Train content-based filtering
        print("\n[2/3] Content-Based Filtering")
        print("-" * 70)
        self.cb_model.fit(movies_df)
        
        # Train popularity-based filtering
        print("\n[3/3] Popularity-Based Filtering")
        print("-" * 70)
        self.pop_model.fit(movies_df, interactions_df)
        
        print("\n" + "=" * 70)
        print("[+] HYBRID RECOMMENDER SYSTEM TRAINED SUCCESSFULLY")
        print("=" * 70)
        
    def _build_user_profiles(self) -> None:
        """
        Build user profiles based on their interactions.
        """
        print("\n[*] Building user profiles...")
        
        # Get movies each user has interacted with (rated >= 4)
        high_rated = self.interactions_df[self.interactions_df['rating'] >= 4]
        
        self.user_profiles = high_rated.groupby('user_id')['movie_id'].apply(list).to_dict()
        
        print(f"[+] Built profiles for {len(self.user_profiles)} users")
        
    def _is_cold_start_user(self, user_id: int) -> bool:
        """
        Check if user is a cold-start case.
        
        Args:
            user_id: User ID
            
        Returns:
            True if cold-start user
        """
        if user_id not in self.user_profiles:
            return True
        
        return len(self.user_profiles[user_id]) < self.min_interactions_for_cf
    
    def _is_cold_start_item(self, movie_id: int) -> bool:
        """
        Check if item is a cold-start case.
        
        Args:
            movie_id: Movie ID
            
        Returns:
            True if cold-start item
        """
        movie_interactions = self.interactions_df[
            self.interactions_df['movie_id'] == movie_id
        ]
        
        return len(movie_interactions) < 5
    
    def recommend(self, user_id: int, n_recommendations: int = 10,
                  exclude_watched: bool = True,
                  diversity_factor: float = 0.0) -> List[Dict]:
        """
        Get hybrid recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            exclude_watched: Whether to exclude already watched movies
            diversity_factor: Factor to promote diversity (0-1, higher = more diverse)
            
        Returns:
            List of recommendation dictionaries with scores and explanations
        """
        print(f"\n[*] Generating recommendations for User {user_id}...")
        
        # Check if cold-start user
        is_cold_start = self._is_cold_start_user(user_id)
        
        if is_cold_start:
            print(f"  [!] Cold-start user detected (limited interaction history)")
            return self._cold_start_recommendations(user_id, n_recommendations)
        
        # Get candidate movies (all movies or unwatched)
        if exclude_watched and user_id in self.user_profiles:
            watched_movies = set(self.user_profiles[user_id])
            candidate_movies = [
                m for m in self.movies_df['movie_id'].values 
                if m not in watched_movies
            ]
        else:
            candidate_movies = self.movies_df['movie_id'].values.tolist()
        
        # Limit candidates for performance
        if len(candidate_movies) > 500:
            # Prioritize popular movies for candidates
            popular_movies = self.pop_model.get_popular_movies(n_recommendations=500)
            popular_ids = [m[0] for m in popular_movies]
            candidate_movies = [m for m in candidate_movies if m in popular_ids]
        
        print(f"  [#] Evaluating {len(candidate_movies)} candidate movies...")
        
        # Get scores from each model
        cf_scores = self.cf_model.get_cf_scores(user_id, candidate_movies)
        cb_scores = self.cb_model.get_cb_scores(
            self.user_profiles.get(user_id, []), 
            candidate_movies
        )
        pop_scores = self.pop_model.get_popularity_scores(candidate_movies)
        
        # Combine scores with weights
        hybrid_scores = (
            self.cf_weight * cf_scores +
            self.cb_weight * cb_scores +
            self.pop_weight * pop_scores
        )
        
        # Apply diversity factor (penalize similar genres)
        if diversity_factor > 0:
            hybrid_scores = self._apply_diversity(
                candidate_movies, hybrid_scores, diversity_factor
            )
        
        # Create recommendations list
        recommendations = []
        for i, movie_id in enumerate(candidate_movies):
            recommendations.append({
                'movie_id': int(movie_id),
                'hybrid_score': float(hybrid_scores[i]),
                'cf_score': float(cf_scores[i]),
                'cb_score': float(cb_scores[i]),
                'pop_score': float(pop_scores[i]),
                'explanation': self._generate_explanation(
                    cf_scores[i], cb_scores[i], pop_scores[i]
                )
            })
        
        # Sort by hybrid score
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Add movie details
        top_recommendations = recommendations[:n_recommendations]
        for rec in top_recommendations:
            movie = self.movies_df[self.movies_df['movie_id'] == rec['movie_id']].iloc[0]
            rec['title'] = movie['title']
            rec['genre'] = movie['genre']
            rec['language'] = movie['language']
            rec['release_year'] = int(movie['release_year'])
        
        print(f"  [+] Generated {len(top_recommendations)} recommendations")
        
        return top_recommendations
    
    def _cold_start_recommendations(self, user_id: int, 
                                    n_recommendations: int) -> List[Dict]:
        """
        Handle cold-start users with limited interaction history.
        
        Strategy:
        1. If user has some preferences, use content-based + popularity
        2. If completely new, use popularity + trending
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommendations
        """
        print(f"  [New] Using cold-start strategy...")
        
        # Check if user has any preferences
        if user_id in self.user_profiles and len(self.user_profiles[user_id]) > 0:
            # Use content-based + popularity
            user_profile = self.user_profiles[user_id]
            
            cb_recs = self.cb_model.get_user_recommendations(
                user_profile, 
                n_recommendations=n_recommendations * 2
            )
            
            # Get scores
            movie_ids = [m[0] for m in cb_recs]
            cb_scores = np.array([m[1] for m in cb_recs])
            pop_scores = self.pop_model.get_popularity_scores(movie_ids)
            
            # Combine (70% content, 30% popularity for cold-start)
            hybrid_scores = 0.7 * cb_scores + 0.3 * pop_scores
            
        else:
            # Completely new user: use popularity + trending
            popular = self.pop_model.get_popular_movies(n_recommendations=n_recommendations)
            trending = self.pop_model.get_trending_movies(n_recommendations=n_recommendations)
            
            # Combine popular and trending
            all_recs = {}
            for movie_id, score in popular:
                all_recs[movie_id] = 0.6 * score
            
            for movie_id, score in trending:
                if movie_id in all_recs:
                    all_recs[movie_id] += 0.4 * score
                else:
                    all_recs[movie_id] = 0.4 * score
            
            movie_ids = list(all_recs.keys())
            hybrid_scores = np.array(list(all_recs.values()))
            cb_scores = np.zeros(len(movie_ids))
            pop_scores = self.pop_model.get_popularity_scores(movie_ids)
        
        # Create recommendations
        recommendations = []
        for i, movie_id in enumerate(movie_ids[:n_recommendations]):
            movie = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            
            recommendations.append({
                'movie_id': int(movie_id),
                'hybrid_score': float(hybrid_scores[i]),
                'cf_score': 0.0,  # No CF for cold-start
                'cb_score': float(cb_scores[i]) if i < len(cb_scores) else 0.0,
                'pop_score': float(pop_scores[i]) if i < len(pop_scores) else 0.0,
                'title': movie['title'],
                'genre': movie['genre'],
                'language': movie['language'],
                'release_year': int(movie['release_year']),
                'explanation': 'Popular recommendation (new user)'
            })
        
        return recommendations
    
    def _apply_diversity(self, movie_ids: List[int], scores: np.ndarray,
                        diversity_factor: float) -> np.ndarray:
        """
        Apply diversity penalty to promote varied recommendations.
        
        Args:
            movie_ids: List of movie IDs
            scores: Current scores
            diversity_factor: Diversity weight
            
        Returns:
            Adjusted scores
        """
        # Get genres for each movie
        genres_list = []
        for movie_id in movie_ids:
            movie = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if len(movie) > 0:
                genres = set(movie.iloc[0]['genre'].split(','))
                genres_list.append(genres)
            else:
                genres_list.append(set())
        
        # Penalize movies with similar genres to higher-scored movies
        adjusted_scores = scores.copy()
        sorted_indices = np.argsort(scores)[::-1]
        
        selected_genres = set()
        for idx in sorted_indices:
            overlap = len(genres_list[idx].intersection(selected_genres))
            penalty = diversity_factor * overlap * 0.1
            adjusted_scores[idx] -= penalty
            selected_genres.update(genres_list[idx])
        
        return adjusted_scores
    
    def _generate_explanation(self, cf_score: float, cb_score: float, 
                             pop_score: float) -> str:
        """
        Generate human-readable explanation for recommendation.
        
        Args:
            cf_score: Collaborative filtering score
            cb_score: Content-based score
            pop_score: Popularity score
            
        Returns:
            Explanation string
        """
        scores = {
            'users like you enjoyed': cf_score * self.cf_weight,
            'matches your preferences': cb_score * self.cb_weight,
            'trending now': pop_score * self.pop_weight
        }
        
        # Get primary reason
        primary_reason = max(scores.items(), key=lambda x: x[1])[0]
        
        return f"Because {primary_reason}"
    
    def get_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Dict]:
        """
        Get movies similar to a given movie.
        
        Args:
            movie_id: Movie ID
            n_similar: Number of similar movies
            
        Returns:
            List of similar movies with details
        """
        similar = self.cb_model.get_similar_movies(movie_id, n_similar)
        
        results = []
        for sim_movie_id, similarity in similar:
            movie = self.movies_df[self.movies_df['movie_id'] == sim_movie_id].iloc[0]
            results.append({
                'movie_id': int(sim_movie_id),
                'title': movie['title'],
                'genre': movie['genre'],
                'similarity': float(similarity),
                'explanation': f"Similar content to movie {movie_id}"
            })
        
        return results
    
    def explain_recommendation(self, user_id: int, movie_id: int) -> Dict:
        """
        Explain why a movie is recommended to a user.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Detailed explanation dictionary
        """
        # Get scores
        cf_score = self.cf_model.predict_rating(user_id, movie_id)
        
        user_profile = self.user_profiles.get(user_id, [])
        cb_score = self.cb_model.get_cb_scores(user_profile, [movie_id])[0]
        pop_score = self.pop_model.get_popularity_scores([movie_id])[0]
        
        # Get movie details
        movie = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
        
        explanation = {
            'movie_id': int(movie_id),
            'title': movie['title'],
            'genre': movie['genre'],
            'scores': {
                'collaborative_filtering': float(cf_score),
                'content_based': float(cb_score),
                'popularity': float(pop_score),
                'hybrid': float(
                    self.cf_weight * (cf_score - 1) / 4 +
                    self.cb_weight * cb_score +
                    self.pop_weight * pop_score
                )
            },
            'weights': {
                'collaborative_filtering': self.cf_weight,
                'content_based': self.cb_weight,
                'popularity': self.pop_weight
            },
            'reasons': []
        }
        
        # Add specific reasons
        if cf_score > 4.0:
            explanation['reasons'].append("Users with similar taste highly rated this")
        
        if cb_score > 0.5:
            explanation['reasons'].append(f"Matches your interest in {movie['genre']}")
        
        if pop_score > 0.7:
            explanation['reasons'].append("Currently trending and popular")
        
        return explanation


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load and prepare data
    print("Preparing data...")
    preprocessor = DataPreprocessor()
    users, movies, interactions = preprocessor.create_sample_data()
    preprocessor.engineer_features()
    
    # Initialize and train hybrid recommender
    recommender = HybridRecommender(
        cf_weight=0.5,
        cb_weight=0.3,
        pop_weight=0.2
    )
    
    recommender.fit(users, movies, interactions)
    
    # Get recommendations for a user
    user_id = 1
    recommendations = recommender.recommend(user_id, n_recommendations=10)
    
    print(f"\n{'='*70}")
    print(f"[*] TOP 10 RECOMMENDATIONS FOR USER {user_id}")
    print(f"{'='*70}\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Genre: {rec['genre']}")
        print(f"   Year: {rec['release_year']}")
        print(f"   Hybrid Score: {rec['hybrid_score']:.3f}")
        print(f"   Explanation: {rec['explanation']}")
        print()

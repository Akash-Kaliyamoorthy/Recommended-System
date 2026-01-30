"""
Popularity-Based Filtering Module

This module recommends trending and popular content based on:
- View counts
- Average ratings
- Recency (trending now)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class PopularityBasedFiltering:
    """
    Popularity-Based Filtering for trending and most-watched content.
    
    Useful for:
    - Cold-start users (no viewing history)
    - Homepage "Trending Now" section
    - "Popular in your region" recommendations
    """
    
    def __init__(self, recency_days: int = 30):
        """
        Initialize popularity-based filtering.
        
        Args:
            recency_days: Number of days to consider for trending calculation
        """
        self.recency_days = recency_days
        self.popularity_scores = None
        self.trending_scores = None
        self.movies_df = None
        
    def fit(self, movies_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        """
        Calculate popularity metrics.
        
        Args:
            movies_df: DataFrame with movie metadata
            interactions_df: DataFrame with user interactions
        """
        print("[*] Calculating Popularity Metrics...")
        
        self.movies_df = movies_df.copy()
        
        # Calculate view counts
        view_counts = interactions_df.groupby('movie_id').size().reset_index(name='view_count')
        
        # Calculate average ratings
        rating_stats = interactions_df.groupby('movie_id')['rating'].agg([
            ('avg_rating', 'mean'),
            ('rating_count', 'count'),
            ('rating_std', 'std')
        ]).reset_index()
        
        # Calculate completion rate
        completion_rate = interactions_df.groupby('movie_id')['completed'].mean().reset_index(
            name='completion_rate'
        )
        
        # Merge all metrics
        popularity = view_counts.merge(rating_stats, on='movie_id', how='left')
        popularity = popularity.merge(completion_rate, on='movie_id', how='left')
        
        # Fill missing values
        popularity = popularity.fillna({
            'avg_rating': 3.0,
            'rating_count': 0,
            'rating_std': 0,
            'completion_rate': 0.5
        })
        
        # Normalize metrics to 0-1 scale
        popularity['view_score'] = self._normalize(popularity['view_count'])
        popularity['rating_score'] = popularity['avg_rating'] / 5.0
        popularity['completion_score'] = popularity['completion_rate']
        
        # Bayesian average rating (handles movies with few ratings)
        C = popularity['rating_count'].mean()  # Average number of ratings
        m = popularity['avg_rating'].mean()    # Mean rating across all movies
        
        popularity['bayesian_rating'] = (
            (popularity['rating_count'] * popularity['avg_rating'] + C * m) /
            (popularity['rating_count'] + C)
        ) / 5.0
        
        # Combined popularity score
        popularity['popularity_score'] = (
            0.3 * popularity['view_score'] +
            0.4 * popularity['bayesian_rating'] +
            0.3 * popularity['completion_score']
        )
        
        # Calculate trending score (recent popularity)
        self._calculate_trending_score(interactions_df, popularity)
        
        self.popularity_scores = popularity
        
        print(f"[+] Popularity metrics calculated for {len(popularity)} movies")
        
    def _calculate_trending_score(self, interactions_df: pd.DataFrame, 
                                   popularity: pd.DataFrame) -> None:
        """
        Calculate trending score based on recent interactions.
        
        Args:
            interactions_df: DataFrame with user interactions
            popularity: DataFrame with popularity metrics
        """
        # Get recent interactions
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        cutoff_date = interactions_df['timestamp'].max() - timedelta(days=self.recency_days)
        
        recent_interactions = interactions_df[interactions_df['timestamp'] >= cutoff_date]
        
        # Calculate recent view counts
        recent_views = recent_interactions.groupby('movie_id').size().reset_index(
            name='recent_view_count'
        )
        
        # Calculate recent ratings
        recent_ratings = recent_interactions.groupby('movie_id')['rating'].mean().reset_index(
            name='recent_avg_rating'
        )
        
        # Merge with popularity
        trending = recent_views.merge(recent_ratings, on='movie_id', how='left')
        trending = trending.fillna({'recent_avg_rating': 3.0})
        
        # Normalize
        trending['recent_view_score'] = self._normalize(trending['recent_view_count'])
        trending['recent_rating_score'] = trending['recent_avg_rating'] / 5.0
        
        # Calculate velocity (growth rate)
        trending = trending.merge(
            popularity[['movie_id', 'view_count']], 
            on='movie_id', 
            how='left'
        )
        
        trending['velocity'] = trending['recent_view_count'] / (trending['view_count'] + 1)
        trending['velocity_score'] = self._normalize(trending['velocity'])
        
        # Combined trending score
        trending['trending_score'] = (
            0.4 * trending['recent_view_score'] +
            0.3 * trending['recent_rating_score'] +
            0.3 * trending['velocity_score']
        )
        
        self.trending_scores = trending[['movie_id', 'trending_score', 'velocity']]
        
    def _normalize(self, series: pd.Series) -> pd.Series:
        """
        Normalize series to 0-1 range.
        
        Args:
            series: Pandas series to normalize
            
        Returns:
            Normalized series
        """
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        
        return (series - min_val) / (max_val - min_val)
    
    def get_popular_movies(self, n_recommendations: int = 10, 
                           min_ratings: int = 10) -> List[Tuple[int, float]]:
        """
        Get most popular movies overall.
        
        Args:
            n_recommendations: Number of recommendations to return
            min_ratings: Minimum number of ratings required
            
        Returns:
            List of (movie_id, popularity_score) tuples
        """
        # Filter by minimum ratings
        popular = self.popularity_scores[
            self.popularity_scores['rating_count'] >= min_ratings
        ].copy()
        
        # Sort by popularity score
        popular = popular.sort_values('popularity_score', ascending=False)
        
        recommendations = [
            (int(row['movie_id']), float(row['popularity_score']))
            for _, row in popular.head(n_recommendations).iterrows()
        ]
        
        return recommendations
    
    def get_trending_movies(self, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get trending movies (popular recently).
        
        Args:
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_id, trending_score) tuples
        """
        if self.trending_scores is None:
            return []
        
        trending = self.trending_scores.sort_values('trending_score', ascending=False)
        
        recommendations = [
            (int(row['movie_id']), float(row['trending_score']))
            for _, row in trending.head(n_recommendations).iterrows()
        ]
        
        return recommendations
    
    def get_top_rated_movies(self, n_recommendations: int = 10,
                             min_ratings: int = 20) -> List[Tuple[int, float]]:
        """
        Get highest-rated movies.
        
        Args:
            n_recommendations: Number of recommendations to return
            min_ratings: Minimum number of ratings required
            
        Returns:
            List of (movie_id, rating) tuples
        """
        top_rated = self.popularity_scores[
            self.popularity_scores['rating_count'] >= min_ratings
        ].copy()
        
        top_rated = top_rated.sort_values('bayesian_rating', ascending=False)
        
        recommendations = [
            (int(row['movie_id']), float(row['bayesian_rating'] * 5))
            for _, row in top_rated.head(n_recommendations).iterrows()
        ]
        
        return recommendations
    
    def get_popularity_scores(self, movie_ids: List[int]) -> np.ndarray:
        """
        Get popularity scores for specific movies.
        
        Args:
            movie_ids: List of movie IDs
            
        Returns:
            Array of popularity scores (0-1 scale)
        """
        scores = []
        
        for movie_id in movie_ids:
            movie_pop = self.popularity_scores[
                self.popularity_scores['movie_id'] == movie_id
            ]
            
            if len(movie_pop) > 0:
                scores.append(float(movie_pop['popularity_score'].values[0]))
            else:
                scores.append(0.0)
        
        return np.array(scores)
    
    def get_regional_popular(self, region: str, users_df: pd.DataFrame,
                             interactions_df: pd.DataFrame,
                             n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get popular movies in a specific region.
        
        Args:
            region: Region/location name
            users_df: DataFrame with user information
            interactions_df: DataFrame with user interactions
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_id, regional_popularity) tuples
        """
        # Get users from the region
        regional_users = users_df[users_df['location'] == region]['user_id'].values
        
        if len(regional_users) == 0:
            return self.get_popular_movies(n_recommendations)
        
        # Get interactions from regional users
        regional_interactions = interactions_df[
            interactions_df['user_id'].isin(regional_users)
        ]
        
        # Calculate regional popularity
        regional_views = regional_interactions.groupby('movie_id').size().reset_index(
            name='regional_views'
        )
        regional_ratings = regional_interactions.groupby('movie_id')['rating'].mean().reset_index(
            name='regional_rating'
        )
        
        regional_pop = regional_views.merge(regional_ratings, on='movie_id')
        
        # Normalize and combine
        regional_pop['view_score'] = self._normalize(regional_pop['regional_views'])
        regional_pop['rating_score'] = regional_pop['regional_rating'] / 5.0
        
        regional_pop['regional_popularity'] = (
            0.6 * regional_pop['view_score'] +
            0.4 * regional_pop['rating_score']
        )
        
        regional_pop = regional_pop.sort_values('regional_popularity', ascending=False)
        
        recommendations = [
            (int(row['movie_id']), float(row['regional_popularity']))
            for _, row in regional_pop.head(n_recommendations).iterrows()
        ]
        
        return recommendations
    
    def get_statistics(self) -> Dict:
        """
        Get popularity statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_movies': len(self.popularity_scores),
            'avg_view_count': float(self.popularity_scores['view_count'].mean()),
            'avg_rating': float(self.popularity_scores['avg_rating'].mean()),
            'avg_completion_rate': float(self.popularity_scores['completion_rate'].mean()),
            'most_popular_movie': int(self.popularity_scores.loc[
                self.popularity_scores['popularity_score'].idxmax(), 'movie_id'
            ])
        }
        
        if self.trending_scores is not None:
            stats['most_trending_movie'] = int(self.trending_scores.loc[
                self.trending_scores['trending_score'].idxmax(), 'movie_id'
            ])
        
        return stats


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    users, movies, interactions = preprocessor.create_sample_data()
    
    # Train popularity-based filtering
    pbf = PopularityBasedFiltering(recency_days=30)
    pbf.fit(movies, interactions)
    
    # Get popular movies
    popular = pbf.get_popular_movies(n_recommendations=10)
    print("\n[*] Top 10 Popular Movies:")
    for movie_id, score in popular:
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {movie_title} (ID: {movie_id}): Popularity = {score:.3f}")
    
    # Get trending movies
    trending = pbf.get_trending_movies(n_recommendations=10)
    print("\n[*] Top 10 Trending Movies:")
    for movie_id, score in trending:
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {movie_title} (ID: {movie_id}): Trending Score = {score:.3f}")
    
    # Get statistics
    stats = pbf.get_statistics()
    print("\n[*] Popularity Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

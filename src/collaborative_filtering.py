"""
Collaborative Filtering Module using Matrix Factorization (SVD)

This module implements user-based collaborative filtering to predict
ratings and generate recommendations based on user-item interactions.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')


class CollaborativeFiltering:
    """
    Collaborative Filtering using Singular Value Decomposition (SVD).
    
    Predicts user ratings for unseen items based on similar users' preferences.
    """
    
    def __init__(self, n_factors: int = 50):
        """
        Initialize collaborative filtering model.
        
        Args:
            n_factors: Number of latent factors for matrix factorization
        """
        self.n_factors = n_factors
        self.user_item_matrix = None
        self.user_ratings_mean = None
        self.predictions_matrix = None
        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_user_map = {}
        self.reverse_movie_map = {}
        
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """
        Train the collaborative filtering model.
        
        Args:
            interactions_df: DataFrame with user_id, movie_id, rating columns
        """
        print("[*] Training Collaborative Filtering model...")
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # Create mappings for matrix indices
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(self.user_item_matrix.index)}
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(self.user_item_matrix.columns)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_movie_map = {idx: movie_id for movie_id, idx in self.movie_id_map.items()}
        
        # Convert to numpy array
        R = self.user_item_matrix.values
        
        # Normalize by subtracting user mean ratings
        self.user_ratings_mean = np.mean(R, axis=1, keepdims=True)
        R_normalized = R - self.user_ratings_mean
        
        # Replace zeros with very small number to avoid issues with SVD
        R_normalized[R_normalized == 0] = 1e-10
        
        # Perform SVD
        print(f"  Performing SVD with {self.n_factors} factors...")
        
        # Adjust n_factors if it's too large
        max_factors = min(R_normalized.shape) - 1
        n_factors = min(self.n_factors, max_factors)
        
        U, sigma, Vt = svds(R_normalized, k=n_factors)
        
        # Convert sigma to diagonal matrix
        sigma = np.diag(sigma)
        
        # Reconstruct the matrix
        self.predictions_matrix = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean
        
        print(f"[+] Collaborative Filtering model trained")
        print(f"  Matrix shape: {R.shape}")
        print(f"  Sparsity: {(R == 0).sum() / R.size * 100:.2f}%")
        
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a specific user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating (1-5 scale)
        """
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            # Return global average for unknown users/movies
            return self.user_ratings_mean.mean()
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        prediction = self.predictions_matrix[user_idx, movie_idx]
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10,
                                 exclude_watched: bool = True) -> List[Tuple[int, float]]:
        """
        Get top-N movie recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            exclude_watched: Whether to exclude already watched movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_id_map:
            print(f"  [!] User {user_id} not found in training data (cold-start)")
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Get predictions for all movies
        user_predictions = self.predictions_matrix[user_idx, :]
        
        # Get already watched movies
        if exclude_watched:
            watched_movies = set(
                self.user_item_matrix.columns[self.user_item_matrix.iloc[user_idx] > 0]
            )
        else:
            watched_movies = set()
        
        # Create list of (movie_id, predicted_rating) for unwatched movies
        recommendations = []
        for movie_idx, predicted_rating in enumerate(user_predictions):
            movie_id = self.reverse_movie_map[movie_idx]
            
            if movie_id not in watched_movies:
                recommendations.append((movie_id, float(predicted_rating)))
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar users based on rating patterns.
        
        Args:
            user_id: User ID
            n_similar: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Calculate cosine similarity with all users
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.user_item_matrix.values)[0]
        
        # Get top similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        similar_users = [
            (self.reverse_user_map[idx], float(similarities[idx]))
            for idx in similar_indices
        ]
        
        return similar_users
    
    def get_cf_scores(self, user_id: int, movie_ids: List[int]) -> np.ndarray:
        """
        Get collaborative filtering scores for specific movies.
        
        Args:
            user_id: User ID
            movie_ids: List of movie IDs
            
        Returns:
            Array of CF scores (normalized 0-1)
        """
        scores = []
        
        for movie_id in movie_ids:
            rating = self.predict_rating(user_id, movie_id)
            # Normalize to 0-1 scale
            normalized_score = (rating - 1) / 4  # 1-5 scale to 0-1
            scores.append(normalized_score)
        
        return np.array(scores)


class UserBasedCF:
    """
    Alternative: User-Based Collaborative Filtering using k-Nearest Neighbors.
    
    Simpler approach that finds similar users and recommends what they liked.
    """
    
    def __init__(self, k_neighbors: int = 20):
        """
        Initialize user-based CF.
        
        Args:
            k_neighbors: Number of similar users to consider
        """
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """
        Train the user-based CF model.
        
        Args:
            interactions_df: DataFrame with user_id, movie_id, rating columns
        """
        print("[*] Training User-Based Collaborative Filtering...")
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # Calculate user similarity matrix
        print("  Calculating user similarities...")
        self.user_similarity_matrix = pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print("[+] User-Based CF model trained")
        
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using k-nearest neighbors.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_similarity_matrix.index:
            return 3.0  # Default rating
        
        if movie_id not in self.user_item_matrix.columns:
            return 3.0
        
        # Get similar users who have rated this movie
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)[1:self.k_neighbors+1]
        
        # Filter users who have rated this movie
        rated_users = self.user_item_matrix[movie_id] > 0
        similar_rated = similar_users[rated_users]
        
        if len(similar_rated) == 0:
            return 3.0
        
        # Weighted average of ratings
        weights = similar_rated.values
        ratings = self.user_item_matrix.loc[similar_rated.index, movie_id].values
        
        if weights.sum() == 0:
            return 3.0
        
        predicted_rating = np.average(ratings, weights=weights)
        
        return np.clip(predicted_rating, 1, 5)


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    users, movies, interactions = preprocessor.create_sample_data()
    
    # Train collaborative filtering
    cf = CollaborativeFiltering(n_factors=50)
    cf.fit(interactions)
    
    # Get recommendations for user 1
    recommendations = cf.get_user_recommendations(user_id=1, n_recommendations=10)
    print("\n[*] Top 10 Recommendations for User 1:")
    for movie_id, rating in recommendations:
        print(f"  Movie {movie_id}: Predicted Rating = {rating:.2f}")
    
    # Find similar users
    similar_users = cf.get_similar_users(user_id=1, n_similar=5)
    print("\n[*] Top 5 Similar Users to User 1:")
    for user_id, similarity in similar_users:
        print(f"  User {user_id}: Similarity = {similarity:.3f}")

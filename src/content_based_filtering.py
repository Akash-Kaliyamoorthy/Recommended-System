"""
Content-Based Filtering Module using TF-IDF and Cosine Similarity

This module recommends movies based on content similarity (genre, cast, description, etc.)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class ContentBasedFiltering:
    """
    Content-Based Filtering using TF-IDF vectorization and cosine similarity.
    
    Recommends items similar to what the user has liked in the past.
    """
    
    def __init__(self, max_features: int = 5000):
        """
        Initialize content-based filtering model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
        """
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        self.tfidf_matrix = None
        self.movies_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.similarity_matrix = None
        
    def fit(self, movies_df: pd.DataFrame) -> None:
        """
        Train the content-based filtering model.
        
        Args:
            movies_df: DataFrame with movie metadata including 'content' column
        """
        print("[*] Training Content-Based Filtering model...")
        
        self.movies_df = movies_df.copy()
        
        # Ensure content column exists
        if 'content' not in self.movies_df.columns:
            print("  Creating content feature...")
            self.movies_df['content'] = (
                self.movies_df['genre'].fillna('') + ' ' +
                self.movies_df['cast'].fillna('') + ' ' +
                self.movies_df['director'].fillna('') + ' ' +
                self.movies_df['description'].fillna('') + ' ' +
                self.movies_df['language'].fillna('')
            )
        
        # Create movie ID mappings
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movies_df['movie_id'])}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
        # Create TF-IDF matrix
        print(f"  Creating TF-IDF matrix with max {self.max_features} features...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['content'])
        
        # Pre-compute similarity matrix for faster recommendations
        print("  Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print(f"[+] Content-Based Filtering model trained")
        print(f"  TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"  Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
    def get_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find movies similar to a given movie.
        
        Args:
            movie_id: Movie ID
            n_similar: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if movie_id not in self.movie_id_to_idx:
            print(f"  [!] Movie {movie_id} not found")
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # Get similarity scores
        similarity_scores = self.similarity_matrix[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_similar+1]
        
        similar_movies = [
            (self.idx_to_movie_id[idx], float(similarity_scores[idx]))
            for idx in similar_indices
        ]
        
        return similar_movies
    
    def get_user_recommendations(self, user_profile: List[int], n_recommendations: int = 10,
                                 exclude_watched: bool = True) -> List[Tuple[int, float]]:
        """
        Get recommendations based on user's viewing history.
        
        Args:
            user_profile: List of movie IDs the user has watched/liked
            n_recommendations: Number of recommendations to return
            exclude_watched: Whether to exclude movies in user profile
            
        Returns:
            List of (movie_id, content_score) tuples
        """
        if not user_profile:
            print("  [!] Empty user profile")
            return []
        
        # Filter valid movie IDs
        valid_movies = [m for m in user_profile if m in self.movie_id_to_idx]
        
        if not valid_movies:
            print("  [!] No valid movies in user profile")
            return []
        
        # Get indices of user's movies
        user_movie_indices = [self.movie_id_to_idx[m] for m in valid_movies]
        
        # Calculate average similarity to user's profile
        user_profile_similarities = self.similarity_matrix[user_movie_indices].mean(axis=0)
        
        # Create recommendations
        recommendations = []
        for idx, score in enumerate(user_profile_similarities):
            movie_id = self.idx_to_movie_id[idx]
            
            # Exclude already watched movies
            if exclude_watched and movie_id in user_profile:
                continue
            
            recommendations.append((movie_id, float(score)))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def get_cb_scores(self, user_profile: List[int], movie_ids: List[int]) -> np.ndarray:
        """
        Get content-based scores for specific movies given user profile.
        
        Args:
            user_profile: List of movie IDs the user has watched/liked
            movie_ids: List of movie IDs to score
            
        Returns:
            Array of CB scores (0-1 scale)
        """
        if not user_profile:
            return np.zeros(len(movie_ids))
        
        # Filter valid movies
        valid_profile = [m for m in user_profile if m in self.movie_id_to_idx]
        
        if not valid_profile:
            return np.zeros(len(movie_ids))
        
        # Get user profile vector (average of liked movies)
        profile_indices = [self.movie_id_to_idx[m] for m in valid_profile]
        user_vector = np.asarray(self.tfidf_matrix[profile_indices].mean(axis=0))
        
        # Calculate scores for target movies
        scores = []
        for movie_id in movie_ids:
            if movie_id not in self.movie_id_to_idx:
                scores.append(0.0)
            else:
                movie_idx = self.movie_id_to_idx[movie_id]
                movie_vector = self.tfidf_matrix[movie_idx]
                
                # Cosine similarity
                similarity = cosine_similarity(user_vector, movie_vector)[0][0]
                scores.append(float(similarity))
        
        return np.array(scores)
    
    def get_genre_recommendations(self, preferred_genres: List[str], 
                                  n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations based on preferred genres.
        
        Args:
            preferred_genres: List of preferred genres
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_id, relevance_score) tuples
        """
        # Create a pseudo-document from preferred genres
        genre_text = ' '.join(preferred_genres)
        
        # Transform to TF-IDF vector
        genre_vector = self.tfidf_vectorizer.transform([genre_text])
        
        # Calculate similarity with all movies
        similarities = cosine_similarity(genre_vector, self.tfidf_matrix)[0]
        
        # Get top movies
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = [
            (self.idx_to_movie_id[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return recommendations
    
    def explain_recommendation(self, movie_id_1: int, movie_id_2: int) -> Dict:
        """
        Explain why two movies are similar.
        
        Args:
            movie_id_1: First movie ID
            movie_id_2: Second movie ID
            
        Returns:
            Dictionary with explanation details
        """
        if movie_id_1 not in self.movie_id_to_idx or movie_id_2 not in self.movie_id_to_idx:
            return {}
        
        movie_1 = self.movies_df[self.movies_df['movie_id'] == movie_id_1].iloc[0]
        movie_2 = self.movies_df[self.movies_df['movie_id'] == movie_id_2].iloc[0]
        
        # Get common genres
        genres_1 = set(movie_1['genre'].split(','))
        genres_2 = set(movie_2['genre'].split(','))
        common_genres = genres_1.intersection(genres_2)
        
        # Get common cast
        cast_1 = set(movie_1['cast'].split(','))
        cast_2 = set(movie_2['cast'].split(','))
        common_cast = cast_1.intersection(cast_2)
        
        # Get similarity score
        idx_1 = self.movie_id_to_idx[movie_id_1]
        idx_2 = self.movie_id_to_idx[movie_id_2]
        similarity = self.similarity_matrix[idx_1, idx_2]
        
        explanation = {
            'similarity_score': float(similarity),
            'common_genres': list(common_genres),
            'common_cast': list(common_cast),
            'same_director': movie_1['director'] == movie_2['director'],
            'same_language': movie_1['language'] == movie_2['language']
        }
        
        return explanation


class GenreBasedFiltering:
    """
    Simplified genre-based filtering for cold-start scenarios.
    """
    
    def __init__(self):
        self.movies_df = None
        self.genre_movie_map = {}
        
    def fit(self, movies_df: pd.DataFrame) -> None:
        """
        Build genre-to-movies mapping.
        
        Args:
            movies_df: DataFrame with movie metadata
        """
        print("[*] Building Genre-Based Filtering...")
        
        self.movies_df = movies_df.copy()
        
        # Create genre to movies mapping
        for _, movie in self.movies_df.iterrows():
            genres = movie['genre'].split(',')
            for genre in genres:
                genre = genre.strip()
                if genre not in self.genre_movie_map:
                    self.genre_movie_map[genre] = []
                self.genre_movie_map[genre].append(movie['movie_id'])
        
        print(f"[+] Genre-Based Filtering ready with {len(self.genre_movie_map)} genres")
        
    def get_recommendations_by_genre(self, genres: List[str], 
                                     n_recommendations: int = 10) -> List[int]:
        """
        Get movie recommendations for given genres.
        
        Args:
            genres: List of genres
            n_recommendations: Number of recommendations
            
        Returns:
            List of movie IDs
        """
        recommendations = []
        
        for genre in genres:
            if genre in self.genre_movie_map:
                recommendations.extend(self.genre_movie_map[genre])
        
        # Remove duplicates and limit
        recommendations = list(set(recommendations))[:n_recommendations]
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    users, movies, interactions = preprocessor.create_sample_data()
    preprocessor.engineer_features()
    
    # Train content-based filtering
    cbf = ContentBasedFiltering(max_features=1000)
    cbf.fit(movies)
    
    # Find similar movies to movie 1
    similar_movies = cbf.get_similar_movies(movie_id=1, n_similar=5)
    print("\n[*] Top 5 Movies Similar to Movie 1:")
    for movie_id, similarity in similar_movies:
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {movie_title} (ID: {movie_id}): Similarity = {similarity:.3f}")
    
    # Get recommendations based on user profile
    user_profile = [1, 5, 10]  # Movies user has watched
    recommendations = cbf.get_user_recommendations(user_profile, n_recommendations=5)
    print(f"\n[*] Top 5 Recommendations based on watching movies {user_profile}:")
    for movie_id, score in recommendations:
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {movie_title} (ID: {movie_id}): Score = {score:.3f}")

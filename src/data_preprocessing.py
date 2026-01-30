"""
Data Preprocessing Module for OTT Recommender System

This module handles:
- Data loading and validation
- Feature engineering
- Data cleaning and normalization
- Train-test split for evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Preprocesses user, movie, and interaction data for the recommender system.
    
    Handles missing values, feature engineering, and data normalization.
    """
    
    def __init__(self):
        self.users_df = None
        self.movies_df = None
        self.interactions_df = None
        
    def load_data(self, users_path: str, movies_path: str, interactions_path: str) -> None:
        """
        Load datasets from CSV files.
        
        Args:
            users_path: Path to users CSV
            movies_path: Path to movies CSV
            interactions_path: Path to interactions CSV
        """
        print("[*] Loading datasets...")
        self.users_df = pd.read_csv(users_path)
        self.movies_df = pd.read_csv(movies_path)
        self.interactions_df = pd.read_csv(interactions_path)
        
        print(f"[+] Loaded {len(self.users_df)} users")
        print(f"[+] Loaded {len(self.movies_df)} movies")
        print(f"[+] Loaded {len(self.interactions_df)} interactions")
        
    def load_real_data(self, movies_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load real movie data from CSV and generate synthetic users/interactions.
        
        Args:
            movies_csv_path: Path to the real movies CSV
            
        Returns:
            Tuple of (users_df, movies_df, interactions_df)
        """
        print(f"[*] Loading real movie data from {movies_csv_path}...")
        raw_movies = pd.read_csv(movies_csv_path)
        
        # Clean and map columns
        # Columns: index,budget,genres,homepage,id,keywords,original_language,original_title,overview,
        # popularity,production_companies,production_countries,release_date,revenue,runtime,
        # spoken_languages,status,tagline,title,vote_average,vote_count,cast,crew,director
        
        self.movies_df = pd.DataFrame()
        self.movies_df['movie_id'] = raw_movies['id']
        self.movies_df['title'] = raw_movies['title']
        
        # Convert space-separated genres to comma-separated
        self.movies_df['genre'] = raw_movies['genres'].str.replace(' ', ', ')
        self.movies_df['cast'] = raw_movies['cast'].fillna('Unknown Actor')
        self.movies_df['director'] = raw_movies['director'].fillna('Unknown Director')
        self.movies_df['description'] = raw_movies['overview'].fillna('No description available.')
        self.movies_df['language'] = raw_movies['original_language']
        
        # Extract year from release_date (YYYY-MM-DD)
        self.movies_df['release_year'] = pd.to_datetime(raw_movies['release_date'], errors='coerce').dt.year.fillna(2000).astype(int)
        self.movies_df['duration'] = raw_movies['runtime'].fillna(100).astype(int)
        
        # Sort by popularity to get better quality movies for interactions
        raw_movies['popularity'] = pd.to_numeric(raw_movies['popularity'], errors='coerce').fillna(0)
        top_movie_ids = self.movies_df.loc[raw_movies['popularity'].nlargest(2000).index, 'movie_id'].values
        
        print(f"[+] Loaded {len(self.movies_df)} real movies")
        
        # Generate Synthetic Users
        np.random.seed(42)
        n_users = 1000
        users_data = {
            'user_id': range(1, n_users + 1),
            'age': np.random.randint(18, 65, n_users),
            'gender': np.random.choice(['M', 'F', 'Other'], n_users),
            'location': np.random.choice(['US', 'UK', 'India', 'Canada', 'Australia'], n_users),
            'signup_date': pd.date_range('2020-01-01', periods=n_users, freq='6H')
        }
        self.users_df = pd.DataFrame(users_data)
        
        # Generate Synthetic Interactions for these real movies
        print("[*] Generating synthetic interactions for real movies...")
        n_interactions = 60000
        
        interactions_data = {
            'user_id': np.random.randint(1, n_users + 1, n_interactions),
            'movie_id': np.random.choice(top_movie_ids, n_interactions),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_interactions, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
            'watch_time': np.random.randint(10, 180, n_interactions),
            'timestamp': pd.date_range('2023-01-01', periods=n_interactions, freq='10T'),
            'completed': np.random.choice([0, 1], n_interactions, p=[0.3, 0.7])
        }
        self.interactions_df = pd.DataFrame(interactions_data)
        self.interactions_df = self.interactions_df.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'movie_id'], keep='last'
        )
        
        return self.users_df, self.movies_df, self.interactions_df

    def create_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create sample datasets for demonstration purposes.
        
        Returns:
            Tuple of (users_df, movies_df, interactions_df)
        """
        # Try to load real data if available
        import os
        real_data_path = './data/movies_real.csv'
        if os.path.exists(real_data_path):
            return self.load_real_data(real_data_path)
            
        print("[*] Creating synthetic sample datasets...")
    
    def engineer_features(self) -> None:
        """
        Create additional features for better recommendations.
        """
        print("[*] Engineering features...")
        
        # User features
        self.users_df['account_age_days'] = (
            pd.Timestamp.now() - pd.to_datetime(self.users_df['signup_date'])
        ).dt.days
        
        # Movie features
        self.movies_df['is_recent'] = (self.movies_df['release_year'] >= 2020).astype(int)
        self.movies_df['genre_count'] = self.movies_df['genre'].fillna('').str.split(',').str.len()
        
        # Create combined content feature for content-based filtering
        self.movies_df['content'] = (
            self.movies_df['genre'].fillna('') + ' ' + 
            self.movies_df['cast'].fillna('') + ' ' + 
            self.movies_df['director'].fillna('') + ' ' + 
            self.movies_df['description'].fillna('') + ' ' +
            self.movies_df['language'].fillna('')
        )
        
        # Add poster URL (using placeholder service with movie_id as seed for variety)
        self.movies_df['poster_url'] = self.movies_df['movie_id'].apply(
            lambda x: f"https://picsum.photos/seed/{x}/300/450"
        )
        
        # Interaction features
        duration_map = self.movies_df.set_index('movie_id')['duration']
        self.interactions_df['watch_completion_rate'] = (
            self.interactions_df['watch_time'] / 
            self.interactions_df['movie_id'].map(duration_map).replace(0, np.nan)
        ).fillna(0).clip(0, 1)
        
        # Implicit rating (combine explicit rating with watch behavior)
        self.interactions_df['implicit_rating'] = (
            0.7 * self.interactions_df['rating'] + 
            0.3 * self.interactions_df['completed'] * 5
        )
        
        print("[+] Feature engineering completed")
        
    def get_popularity_metrics(self) -> pd.DataFrame:
        """
        Calculate popularity metrics for each movie.
        
        Returns:
            DataFrame with movie popularity scores
        """
        print("[*] Calculating popularity metrics...")
        
        # View count
        view_counts = self.interactions_df.groupby('movie_id').size().reset_index(name='view_count')
        
        # Average rating
        avg_ratings = self.interactions_df.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
        avg_ratings.columns = ['movie_id', 'avg_rating', 'rating_count']
        
        # Recency score (recent interactions get higher scores)
        self.interactions_df['days_ago'] = (
            pd.Timestamp.now() - pd.to_datetime(self.interactions_df['timestamp'])
        ).dt.days
        
        recency_scores = self.interactions_df.groupby('movie_id').apply(
            lambda x: np.mean(1 / (1 + x['days_ago'] / 30))  # Decay over 30 days
        ).reset_index(name='recency_score')
        
        # Combine metrics
        popularity = view_counts.merge(avg_ratings, on='movie_id').merge(recency_scores, on='movie_id')
        
        # Normalize scores to 0-1 range
        popularity['view_score'] = (
            (popularity['view_count'] - popularity['view_count'].min()) / 
            (popularity['view_count'].max() - popularity['view_count'].min())
        )
        
        popularity['rating_score'] = popularity['avg_rating'] / 5.0
        
        # Combined popularity score
        popularity['popularity_score'] = (
            0.4 * popularity['view_score'] + 
            0.4 * popularity['rating_score'] + 
            0.2 * popularity['recency_score']
        )
        
        print("[+] Popularity metrics calculated")
        
        return popularity
    
    def train_test_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split interactions into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print(f"[*] Splitting data (test_size={test_size})...")
        
        # Sort by timestamp and split
        interactions_sorted = self.interactions_df.sort_values('timestamp')
        
        split_idx = int(len(interactions_sorted) * (1 - test_size))
        train_df = interactions_sorted.iloc[:split_idx].copy()
        test_df = interactions_sorted.iloc[split_idx:].copy()
        
        print(f"[+] Train set: {len(train_df)} interactions")
        print(f"[+] Test set: {len(test_df)} interactions")
        
        return train_df, test_df
    
    def get_data_stats(self) -> Dict:
        """
        Get statistics about the datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'n_users': len(self.users_df),
            'n_movies': len(self.movies_df),
            'n_interactions': len(self.interactions_df),
            'sparsity': 1 - (len(self.interactions_df) / (len(self.users_df) * len(self.movies_df))),
            'avg_ratings_per_user': len(self.interactions_df) / len(self.users_df),
            'avg_ratings_per_movie': len(self.interactions_df) / len(self.movies_df),
            'rating_distribution': self.interactions_df['rating'].value_counts().to_dict()
        }
        
        return stats
    
    def save_processed_data(self, output_dir: str = '../data/processed/') -> None:
        """
        Save processed datasets to CSV files.
        
        Args:
            output_dir: Directory to save processed data
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.users_df.to_csv(f'{output_dir}/users_processed.csv', index=False)
        self.movies_df.to_csv(f'{output_dir}/movies_processed.csv', index=False)
        self.interactions_df.to_csv(f'{output_dir}/interactions_processed.csv', index=False)
        
        print(f"[+] Processed data saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Create sample data
    users, movies, interactions = preprocessor.create_sample_data()
    
    # Engineer features
    preprocessor.engineer_features()
    
    # Get statistics
    stats = preprocessor.get_data_stats()
    print("\n[Stat] Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get popularity metrics
    popularity = preprocessor.get_popularity_metrics()
    print(f"\n[Pop] Top 5 Popular Movies:")
    top_movies = popularity.nlargest(5, 'popularity_score')
    print(top_movies[['movie_id', 'view_count', 'avg_rating', 'popularity_score']])

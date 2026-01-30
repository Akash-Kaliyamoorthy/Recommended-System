"""
OTT Recommender System Package

A hybrid recommendation engine for streaming platforms.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_preprocessing import DataPreprocessor
from .collaborative_filtering import CollaborativeFiltering
from .content_based_filtering import ContentBasedFiltering
from .popularity_based import PopularityBasedFiltering
from .hybrid_recommender import HybridRecommender
from .evaluation import RecommenderEvaluator

__all__ = [
    'DataPreprocessor',
    'CollaborativeFiltering',
    'ContentBasedFiltering',
    'PopularityBasedFiltering',
    'HybridRecommender',
    'RecommenderEvaluator'
]

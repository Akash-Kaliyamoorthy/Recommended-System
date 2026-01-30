"""
Evaluation Module for Recommender System

Implements various metrics to evaluate recommendation quality:
- Precision@K
- Recall@K
- RMSE (Root Mean Squared Error)
- Coverage
- Diversity
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class RecommenderEvaluator:
    """
    Comprehensive evaluation for recommender systems.
    """
    
    def __init__(self):
        self.metrics = {}
        
    def precision_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Precision@K.
        
        Precision@K = (# of recommended items @K that are relevant) / K
        
        Args:
            recommended: List of recommended item IDs (ordered by score)
            relevant: List of relevant item IDs (ground truth)
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        recommended_at_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        hits = len(recommended_at_k.intersection(relevant_set))
        
        return hits / k
    
    def recall_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Recall@K.
        
        Recall@K = (# of recommended items @K that are relevant) / (total # of relevant items)
        
        Args:
            recommended: List of recommended item IDs (ordered by score)
            relevant: List of relevant item IDs (ground truth)
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0
        
        recommended_at_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        hits = len(recommended_at_k.intersection(relevant_set))
        
        return hits / len(relevant_set)
    
    def f1_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate F1@K score (harmonic mean of Precision and Recall).
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            k: Number of top recommendations
            
        Returns:
            F1@K score
        """
        precision = self.precision_at_k(recommended, relevant, k)
        recall = self.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Average Precision@K (AP@K).
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            k: Number of top recommendations
            
        Returns:
            AP@K score
        """
        if len(relevant) == 0:
            return 0.0
        
        relevant_set = set(relevant)
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended[:k]):
            if item in relevant_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(relevant), k)
    
    def mean_average_precision_at_k(self, all_recommended: List[List[int]], 
                                     all_relevant: List[List[int]], k: int) -> float:
        """
        Calculate Mean Average Precision@K (MAP@K) across all users.
        
        Args:
            all_recommended: List of recommendation lists for each user
            all_relevant: List of relevant item lists for each user
            k: Number of top recommendations
            
        Returns:
            MAP@K score
        """
        ap_scores = []
        
        for recommended, relevant in zip(all_recommended, all_relevant):
            ap = self.average_precision_at_k(recommended, relevant, k)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def ndcg_at_k(self, recommended: List[int], relevant: List[int], 
                  relevance_scores: Dict[int, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            relevance_scores: Dictionary mapping item IDs to relevance scores
            k: Number of top recommendations
            
        Returns:
            NDCG@K score
        """
        def dcg(items, scores, k):
            dcg_score = 0.0
            for i, item in enumerate(items[:k]):
                rel = scores.get(item, 0.0)
                dcg_score += (2 ** rel - 1) / np.log2(i + 2)
            return dcg_score
        
        # Calculate DCG
        dcg_score = dcg(recommended, relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_items = sorted(relevant, key=lambda x: relevance_scores.get(x, 0.0), reverse=True)
        idcg_score = dcg(ideal_items, relevance_scores, k)
        
        if idcg_score == 0:
            return 0.0
        
        return dcg_score / idcg_score
    
    def rmse(self, predictions: List[float], actuals: List[float]) -> float:
        """
        Calculate Root Mean Squared Error for rating predictions.
        
        Args:
            predictions: List of predicted ratings
            actuals: List of actual ratings
            
        Returns:
            RMSE score
        """
        return np.sqrt(mean_squared_error(actuals, predictions))
    
    def mae(self, predictions: List[float], actuals: List[float]) -> float:
        """
        Calculate Mean Absolute Error for rating predictions.
        
        Args:
            predictions: List of predicted ratings
            actuals: List of actual ratings
            
        Returns:
            MAE score
        """
        return np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    
    def coverage(self, all_recommended: List[List[int]], total_items: int) -> float:
        """
        Calculate catalog coverage (percentage of items recommended).
        
        Args:
            all_recommended: List of recommendation lists
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score (0-1)
        """
        unique_recommended = set()
        for recommended in all_recommended:
            unique_recommended.update(recommended)
        
        return len(unique_recommended) / total_items
    
    def diversity(self, recommended: List[int], item_features: pd.DataFrame,
                  feature_column: str = 'genre') -> float:
        """
        Calculate diversity of recommendations based on item features.
        
        Args:
            recommended: List of recommended item IDs
            item_features: DataFrame with item features
            feature_column: Column name for diversity calculation
            
        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if len(recommended) <= 1:
            return 0.0
        
        # Get features for recommended items
        rec_features = item_features[item_features['movie_id'].isin(recommended)]
        
        # Calculate pairwise dissimilarity
        feature_sets = []
        for _, item in rec_features.iterrows():
            features = set(str(item[feature_column]).split(','))
            feature_sets.append(features)
        
        # Calculate average Jaccard distance
        total_distance = 0.0
        count = 0
        
        for i in range(len(feature_sets)):
            for j in range(i + 1, len(feature_sets)):
                intersection = len(feature_sets[i].intersection(feature_sets[j]))
                union = len(feature_sets[i].union(feature_sets[j]))
                
                if union > 0:
                    jaccard_similarity = intersection / union
                    jaccard_distance = 1 - jaccard_similarity
                    total_distance += jaccard_distance
                    count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def novelty(self, recommended: List[int], item_popularity: Dict[int, float]) -> float:
        """
        Calculate novelty of recommendations (how non-obvious they are).
        
        Args:
            recommended: List of recommended item IDs
            item_popularity: Dictionary mapping item IDs to popularity scores
            
        Returns:
            Novelty score (higher = more novel/surprising)
        """
        if not recommended:
            return 0.0
        
        # Novelty is inverse of popularity
        novelty_scores = []
        for item in recommended:
            popularity = item_popularity.get(item, 0.5)
            novelty = 1 - popularity
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    def evaluate_recommender(self, recommender, test_df: pd.DataFrame,
                            movies_df: pd.DataFrame, k: int = 10) -> Dict:
        """
        Comprehensive evaluation of a recommender system.
        
        Args:
            recommender: Trained recommender system
            test_df: Test interactions DataFrame
            movies_df: Movies DataFrame
            k: Number of recommendations to evaluate
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"[*] EVALUATING RECOMMENDER SYSTEM (K={k})")
        print(f"{'='*70}\n")
        
        # Get unique users in test set
        test_users = test_df['user_id'].unique()
        
        # Metrics storage
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ap_scores = []
        predictions = []
        actuals = []
        all_recommendations = []
        
        print(f"Evaluating {len(test_users)} users...")
        
        for i, user_id in enumerate(test_users):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(test_users)} users")
            
            # Get user's test interactions (relevant items)
            user_test = test_df[test_df['user_id'] == user_id]
            relevant_items = user_test[user_test['rating'] >= 4]['movie_id'].tolist()
            
            if not relevant_items:
                continue
            
            # Get recommendations
            try:
                recs = recommender.recommend(user_id, n_recommendations=k, exclude_watched=True)
                recommended_items = [r['movie_id'] for r in recs]
                all_recommendations.append(recommended_items)
                
                # Calculate metrics
                precision = self.precision_at_k(recommended_items, relevant_items, k)
                recall = self.recall_at_k(recommended_items, relevant_items, k)
                f1 = self.f1_at_k(recommended_items, relevant_items, k)
                ap = self.average_precision_at_k(recommended_items, relevant_items, k)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                ap_scores.append(ap)
                
                # Rating predictions for RMSE
                for _, interaction in user_test.iterrows():
                    pred_rating = recommender.cf_model.predict_rating(
                        user_id, interaction['movie_id']
                    )
                    predictions.append(pred_rating)
                    actuals.append(interaction['rating'])
                    
            except Exception as e:
                continue
        
        # Calculate aggregate metrics
        results = {
            'precision@k': np.mean(precision_scores) if precision_scores else 0.0,
            'recall@k': np.mean(recall_scores) if recall_scores else 0.0,
            'f1@k': np.mean(f1_scores) if f1_scores else 0.0,
            'map@k': np.mean(ap_scores) if ap_scores else 0.0,
            'rmse': self.rmse(predictions, actuals) if predictions else 0.0,
            'mae': self.mae(predictions, actuals) if predictions else 0.0,
            'coverage': self.coverage(all_recommendations, len(movies_df)),
            'n_users_evaluated': len(precision_scores)
        }
        
        # Calculate average diversity
        diversity_scores = []
        for recs in all_recommendations[:100]:  # Sample for performance
            div = self.diversity(recs, movies_df, 'genre')
            diversity_scores.append(div)
        
        results['diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # Print results
        print(f"\n{'='*70}")
        print("[*] EVALUATION RESULTS")
        print(f"{'='*70}\n")
        
        print(f"Ranking Metrics (K={k}):")
        print(f"  Precision@{k}:  {results['precision@k']:.4f}")
        print(f"  Recall@{k}:     {results['recall@k']:.4f}")
        print(f"  F1@{k}:         {results['f1@k']:.4f}")
        print(f"  MAP@{k}:        {results['map@k']:.4f}")
        
        print(f"\nRating Prediction Metrics:")
        print(f"  RMSE:          {results['rmse']:.4f}")
        print(f"  MAE:           {results['mae']:.4f}")
        
        print(f"\nCatalog Metrics:")
        print(f"  Coverage:      {results['coverage']:.4f} ({results['coverage']*100:.1f}%)")
        print(f"  Diversity:     {results['diversity']:.4f}")
        
        print(f"\nEvaluation Stats:")
        print(f"  Users evaluated: {results['n_users_evaluated']}")
        
        print(f"\n{'='*70}\n")
        
        self.metrics = results
        return results
    
    def compare_models(self, models: Dict[str, any], test_df: pd.DataFrame,
                      movies_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """
        Compare multiple recommender models.
        
        Args:
            models: Dictionary of model_name -> model
            test_df: Test DataFrame
            movies_df: Movies DataFrame
            k: Number of recommendations
            
        Returns:
            Comparison DataFrame
        """
        results = []
        
        for model_name, model in models.items():
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*70}")
            
            metrics = self.evaluate_recommender(model, test_df, movies_df, k)
            metrics['model'] = model_name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df[['model', 'precision@k', 'recall@k', 'f1@k', 
                                       'map@k', 'rmse', 'mae', 'coverage', 'diversity']]
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    from hybrid_recommender import HybridRecommender
    
    # Prepare data
    preprocessor = DataPreprocessor()
    users, movies, interactions = preprocessor.create_sample_data()
    preprocessor.engineer_features()
    
    # Split data
    train_df, test_df = preprocessor.train_test_split(test_size=0.2)
    
    # Train recommender
    recommender = HybridRecommender()
    recommender.fit(users, movies, train_df)
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    results = evaluator.evaluate_recommender(recommender, test_df, movies, k=10)
